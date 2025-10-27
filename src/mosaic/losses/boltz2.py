from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import boltz.data.const as const
import boltz.main as boltz_main
import equinox as eqx
import gemmi
import jax
import joltz
import numpy as np
import torch
from boltz.model.models.boltz2 import Boltz2
from boltz.data.const import ref_atoms
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree
from joltz import TrunkState


from ..common import LinearCombination, LossTerm
from .structure_prediction import AbstractStructureOutput


def load_boltz2(checkpoint_path=Path("~/.boltz/boltz2_conf.ckpt").expanduser()):
    if not checkpoint_path.exists():
        print(f"Downloading Boltz checkpoint to {checkpoint_path}")
        cache = checkpoint_path.parent
        cache.mkdir(parents=True, exist_ok=True)
        boltz_main.download_boltz2(cache)

    torch_model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location="cpu",
        # Note: these args ARE NOT USED during prediction, but are needed to load the model
        predict_args={
            "recycling_steps": 0,
            "sampling_steps": 25,
            "diffusion_samples": 1,
        },
        diffusion_process_args=asdict(boltz_main.Boltz2DiffusionParams()),
        # ema=False,
        msa_args=asdict(
            boltz_main.MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
        ),
        pairformer_args=asdict(boltz_main.PairformerArgsV2()),
    ).eval()

    model = joltz.from_torch(torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(_model_params), _model_static)


## Duplicated code
class StructureWriter:
    writer: boltz_main.BoltzWriter
    atom_pad_mask: torch.Tensor
    record: any
    out_dir: str
    temp_dir_handle: TemporaryDirectory

    def __init__(
        self,
        *,
        features_dict,
        target_dir: Path,
        output_dir: Path,
        temp_dir_handle: TemporaryDirectory,
    ):
        self.writer = boltz_main.BoltzWriter(
            data_dir=target_dir,
            output_dir=output_dir,
            output_format="mmcif",
            boltz2=True,
        )
        self.atom_pad_mask = features_dict["atom_pad_mask"].unsqueeze(0)
        self.record = features_dict["record"][0]
        self.out_dir = output_dir
        self.temp_dir_handle = temp_dir_handle

    def __call__(self, sample_atom_coords) -> gemmi.Structure:
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
        return gemmi.read_structure(
            str((Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.cif")
        )


def load_features_and_structure_writer(
    input_yaml_str: str,
    cache=Path("~/.boltz/").expanduser(),
) -> PyTree:
    print("Loading data")
    out_dir_handle = (
        TemporaryDirectory()
    )  # this is sketchy -- we have to remember not to let this get garbage collected
    out_dir = Path(out_dir_handle.name)
    # dump the yaml to a file
    input_data_path = out_dir / "input.yaml"
    input_data_path.write_text(input_yaml_str)
    data = boltz_main.check_inputs(input_data_path)
    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    manifest = boltz_main.process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    if manifest is None:
        print("Something odd happened with manifest, trying to reload.")
        manifest = boltz_main.Manifest.load(processed_dir / "manifest.json")

    processed = boltz_main.BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Create data module
    data_module = boltz_main.Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
        mol_dir=mol_dir,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
    )

    # Load the features for the single example
    features_dict = list(data_module.predict_dataloader())[0]

    # convert features to numpy arrays
    features = {k: np.array(v) for k, v in features_dict.items() if k != "record"}

    ## one-hot the MSA

    features["msa"] = jax.nn.one_hot(features["msa"], const.num_tokens)
    # fix up some dtypes
    # features["method_feature"] = features["method_feature"].astype(np.int32)

    writer = StructureWriter(
        features_dict=features_dict,
        target_dir=processed.targets_dir,
        output_dir=out_dir / "output",
        temp_dir_handle=out_dir_handle,
    )

    return jax.tree.map(jnp.array, features), writer


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree,
):
    """Replace features related to first N tokens with `new_sequence.` Used for hallucination/binder design."""
    # features = jax.tree.map(lambda v: v.astype(jnp.float32), features)
    features["res_type"] = features["res_type"].astype(jnp.float32)
    features["msa"] = features["msa"].astype(jnp.float32)
    features["profile"] = features["profile"].astype(jnp.float32)
    assert len(new_sequence.shape) == 2
    assert new_sequence.shape[1] == 20
    binder_len = new_sequence.shape[0]

    # We only use the standard 20 amino acids, but boltz has 33 total tokens.
    # zero out non-standard AA types
    zero_padded_sequence = jnp.pad(new_sequence, ((0, 0), (2, 11)))
    n_msa = features["msa"].shape[1]
    print("n_msa", n_msa)

    # We assume there are no MSA hits for the binder sequence
    binder_profile = jnp.zeros_like(features["profile"][0, :binder_len])
    binder_profile = binder_profile.at[:binder_len].set(zero_padded_sequence) / n_msa
    binder_profile = binder_profile.at[:, 1].set((n_msa - 1) / n_msa)

    return features | {
        "res_type": features["res_type"]
        .at[0, :binder_len, :]
        .set(zero_padded_sequence),
        "msa": features["msa"].at[0, 0, :binder_len, :].set(zero_padded_sequence),
        "profile": features["profile"].at[0, :binder_len].set(binder_profile),
    }


# TODO: remove some batch dimensions
@dataclass
class Boltz2Output(AbstractStructureOutput):
    joltz2: joltz.Joltz2
    features: PyTree
    deterministic: bool
    key: jax.Array
    recycling_steps: int = 0
    num_sampling_steps: int = 25
    initial_recycling_state: TrunkState | None = None

    @property
    def full_sequence(self):
        return self.features["res_type"][0][:, 2:22]

    @property
    def asym_id(self):
        return self.features["asym_id"][0]

    @property
    def residue_idx(self):
        return self.features["residue_index"][0]

    @cached_property
    def initial_embedding(self):
        return self.joltz2.embed_inputs(self.features)

    @cached_property
    def trunk_state(self):
        print("JIT compiling trunk module...")

        def body_fn(carry, _):
            trunk_state, key = carry
            trunk_state = jax.tree.map(jax.lax.stop_gradient, trunk_state)
            trunk_state, key = self.joltz2.trunk_iteration(
                trunk_state,
                self.initial_embedding,
                self.features,
                key=key,
                deterministic=self.deterministic,
            )
            return (trunk_state, key), None

        if self.initial_recycling_state is None:
            state = TrunkState(
                s=jnp.zeros_like(self.initial_embedding.s_init),
                z=jnp.zeros_like(self.initial_embedding.z_init),
            )
        else:
            state = self.initial_recycling_state

        (final_state, _), _ = jax.lax.scan(
            body_fn,
            (state, self.key),
            None,
            length=self.recycling_steps,
        )
        return final_state

    @property
    def distogram_bins(self) -> Float[Array, "64"]:
        return np.linspace(start=2.0, stop=22.0, num=64)

    @cached_property
    def distogram_logits(self) -> Float[Array, "N N 64"]:
        return self.joltz2.distogram_module(self.trunk_state.z)[0, :, :, 0, :]

    @cached_property
    def structure_coordinates(self):
        print("JIT compiling structure module...")
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.joltz2.diffusion_conditioning(
                self.trunk_state.s,
                self.trunk_state.z,
                self.initial_embedding.relative_position_encoding,
                self.features,
            )
        )
        with jax.default_matmul_precision("float32"):
            return self.joltz2.structure_module.sample(
                s_trunk=self.trunk_state.s,
                s_inputs=self.initial_embedding.s_inputs,
                feats=self.features,
                num_sampling_steps=self.num_sampling_steps,
                atom_mask=self.features["atom_pad_mask"],
                multiplicity=1,
                diffusion_conditioning={
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                },
                key=jax.random.fold_in(self.key, 2),
            )

    @cached_property
    def confidence_metrics(self) -> joltz.ConfidenceMetrics:
        print("JIT compiling confidence module...")
        return self.joltz2.confidence_module(
            s_inputs=self.initial_embedding.s_inputs,
            s=self.trunk_state.s,
            z=self.trunk_state.z,
            x_pred=self.structure_coordinates,
            feats=self.features,
            pred_distogram_logits=self.distogram_logits[None],
            key=jax.random.fold_in(self.key, 5),
            deterministic=self.deterministic,
        )

    @property
    def plddt(self) -> Float[Array, "N"]:
        """PLDDT *normalized* to between 0 and 1."""
        return self.confidence_metrics.plddt[0]

    @property
    def pae(self) -> Float[Array, "N N"]:
        return self.confidence_metrics.pae[0]

    @property
    def pae_logits(self) -> Float[Array, "N N Bins"]:
        return self.confidence_metrics.pae_logits[0]

    @property
    def pae_bins(self) -> Float[Array, "Bins"]:
        end = 32.0
        num_bins = 64
        bin_width = end / num_bins
        return np.arange(start=0.5 * bin_width, stop=end, step=bin_width)

    @property
    def backbone_coordinates(self) -> Float[Array, "N 4"]:
        features = jax.tree.map(lambda x: x[0], self.features)
        # In order these are N, C-alpha, C, O
        assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
        # first step, which is a bit cryptic, is to get the first atom for each token
        first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
            features["atom_to_token"].T
        )
        # NOTE: this will completely (and silently) fail if any tokens are non-protein!
        all_atom_coords = self.structure_coordinates[0]
        coords = jnp.stack([all_atom_coords[first_atom_idx + i] for i in range(4)], -2)
        return coords


class Boltz2Loss(LossTerm):
    joltz2: joltz.Joltz2
    features: PyTree
    loss: LossTerm | LinearCombination
    deterministic: bool = True
    recycling_steps: int = 0
    sampling_steps: int = 25
    name: str = "boltz2"
    initial_recycling_state: TrunkState | None = None

    def __call__(self, sequence: Float[Array, "N 20"], key=None):
        """Compute the loss for a given sequence."""
        # Set the binder sequence in the features
        features = set_binder_sequence(sequence, self.features)

        # initialize lazy output object
        output = Boltz2Output(
            joltz2=self.joltz2,
            features=features,
            deterministic=self.deterministic,
            key=key,
            recycling_steps=self.recycling_steps,
            num_sampling_steps=self.sampling_steps,
            initial_recycling_state=self.initial_recycling_state,
        )

        v, aux = self.loss(
            sequence=sequence,
            output=output,
            key=key,
        )

        return v, {self.name: aux}
