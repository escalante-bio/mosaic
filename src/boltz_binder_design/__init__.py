### Binder design using hallucination (following ColabFold)


from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from boltz.main import (
    BoltzInferenceDataModule,
    BoltzProcessedInput,
    BoltzWriter,
    Manifest,
    check_inputs,
    process_inputs,
)
from jaxtyping import PyTree


class StructureWriter:
    """
        Hacky class to write predicted structures to disk using a BoltzWriter
    """
    writer: BoltzWriter
    atom_pad_mask: torch.Tensor
    record: any
    out_dir: str

    def __init__(
        self,
        *,
        features_dict,
        target_dir: Path,
        output_dir: Path,
        output_format: str = "pdb",
    ):
        self.writer = BoltzWriter(
            data_dir=target_dir,
            output_dir=output_dir,
            output_format=output_format,
        )
        self.atom_pad_mask = features_dict["atom_pad_mask"].unsqueeze(0)
        self.record = features_dict["record"][0]
        self.out_dir = output_dir

    def __call__(self, sample_atom_coords):
        confidence = torch.ones(1)

        pred_dict = {
            "exception": False,
            "coords": torch.tensor(np.array(sample_atom_coords)).unsqueeze(0),
            "masks": self.atom_pad_mask,
            "confidence_score": confidence,
        }
        self.writer.write_on_batch_end(
            None,
            None,
            pred_dict,
            None,
            {"record": [self.record]},
            None,
            None,
        )
        # TODO: return path to output structure
        return (Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.pdb"


def binder_fasta_seq(binder_len: int) -> str:
    return f">A|protein|empty\n{'X'*binder_len}\n"


def target_fasta_seq(
    sequence: str, polymer_type: str = "protein", use_msa=True, chain="B"
):
    return f">{chain}|{polymer_type}{'|empty' if not use_msa else ''}\n{sequence}\n"


def make_binder_features(
    binder_len: int,
    target_sequence: str,
    target_polymer_type: str = "protein",
    use_msa=True,
    out_dir: Path | None = None,
):
    if out_dir is None:
        out_dir = Path(target_sequence)

    out_dir.mkdir(exist_ok=True, parents=True)
    fasta_path = out_dir / "protein.fasta"
    fasta_path.write_text(
        binder_fasta_seq(binder_len)
        + target_fasta_seq(target_sequence, target_polymer_type, use_msa)
    )
    return load_features_and_structure_writer(fasta_path, out_dir)


def make_binder_monomer_features(monomer_len: int, out_dir: Path | None = None):
    return make_monomer_features(
        "X" * monomer_len, out_dir, use_msa=False, polymer_type="protein"
    )


def make_monomer_features(
    seq: str, out_dir: Path | None = None, use_msa=True, polymer_type: str = "protein"
):
    if out_dir is None:
        out_dir = Path(f"monomer_{seq}")

    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "protein.fasta").write_text(
        target_fasta_seq(seq, polymer_type=polymer_type, use_msa=use_msa, chain="A")
    )
    return load_features_and_structure_writer(out_dir / "protein.fasta", out_dir)


def load_features_and_structure_writer(
    input_fasta_path: Path,
    out_dir: Path,
    cache=Path("~/.boltz/").expanduser(),
) -> tuple[PyTree, StructureWriter]:
    print("Loading data")
    data = check_inputs(input_fasta_path, out_dir, override=True)
    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
    )
    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
    )

    # Load the features for the single example
    features_dict = list(data_module.predict_dataloader())[0]

    # convert features to numpy arrays
    features = {
        k: jnp.array(v).astype(jnp.float32)
        for k, v in features_dict.items()
        if k != "record"
    }
    # set up structure writer
    writer = StructureWriter(
        features_dict=features_dict,
        target_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
    )
    return features, writer


# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
@eqx.filter_jit
def _eval_loss_and_grad(loss_function, x, key):
    return eqx.filter_value_and_grad(loss_function, has_aux=True)(
        x,
        key=key,
    )


def _bregman_step_optax(*, optim, opt_state, x, loss_function, key):
    (v, aux), g = _eval_loss_and_grad(
        loss_function=loss_function, x=jax.nn.softmax(x), key=key
    )

    updates, opt_state = optim.update(g, opt_state, x)
    x = optax.apply_updates(x, updates)

    x = jax.nn.log_softmax(x)  # do we need this?
    return x, v, aux, opt_state


def design_bregman_optax(
    *,
    loss_function,
    x,
    n_steps: int,
    optim=optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1E-1))
):


    opt_state = optim.init(x)
    for _iter in range(n_steps):
        x, v, aux, opt_state = _bregman_step_optax(
            x=x,
            loss_function=loss_function,

            key=jax.random.key(np.random.randint(0, 10000)),
            optim=optim,
            opt_state=opt_state,
        )


        
        entropy = -(jax.nn.log_softmax(x) * jax.nn.softmax(x)).sum(-1).mean()
        _print_iter(_iter, aux, entropy, v)

    return x

def _print_iter(iter, aux, entropy, v):
    print(
        iter,
        f"loss: {v :0.2f} [entropy: {entropy :0.2f}]",
        " ".join(f"{k}:{v : 0.2f}" for (k, v) in aux.items() if hasattr(v, "item")),
    )
