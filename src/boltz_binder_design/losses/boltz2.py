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
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree

from boltz_binder_design.common import LinearCombination, LossTerm


def load_boltz2(cache=Path("~/.boltz").expanduser()):
    boltz_main.download_boltz2(cache)
    torch_model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt",
        strict=True,
        map_location="cpu",
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
    templates: dict[str, gemmi.Structure] = {}
) -> PyTree:
    assert len(templates) == 0, "Templates are not supported yet."
    print("Loading data")
    out_dir_handle = (
        TemporaryDirectory()
    )  # this is sketchy -- we have to remember not to let this get garbage collected
    out_dir = Path(out_dir_handle.name)
    # dump the yaml to a file
    input_data_path = out_dir / "input.yaml"
    input_data_path.write_text(input_yaml_str)
    for template_id, template in templates.items():
        template_path = out_dir / f"{template_id}.cif"
        template.make_mmcif_document().write_file(str(template_path))
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
class Boltz2Output:
    joltz2: joltz.Joltz2
    features: PyTree
    deterministic: bool
    key: jax.Array
    recycling_steps: int = 0
    num_sampling_steps: int = 25

    @cached_property
    def initial_embedding(self):
        return self.joltz2.embed_inputs(self.features)

    @cached_property
    def trunk_state(self):
        print("running trunk module")
        return self.joltz2.recycle(
            initial_embedding=self.initial_embedding,
            recycling_steps=self.recycling_steps,
            feats=self.features,
            key=self.key,
            deterministic=self.deterministic,
        )[0]

    @cached_property
    def distogram_logits(self) -> Float[Array, "N N 64"]:
        return self.joltz2.distogram_module(self.trunk_state.z)[0, :, :, 0, :] 

    @cached_property
    def structure_coordinates(self):
        print("running structure module")
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
                key = jax.random.fold_in(self.key, 2),
            )
        
    @cached_property
    def confidence_metrics(self) -> joltz.ConfidenceMetrics:
        return self.joltz2.confidence_module(
            s_inputs=self.initial_embedding.s_inputs,
            s=self.trunk_state.s,
            z=self.trunk_state.z,
            x_pred=self.structure_coordinates,
            feats=self.features,
            pred_distogram_logits=self.distogram_logits[None],
            key = jax.random.fold_in(self.key, 5),
            deterministic=self.deterministic,
        )
        
    
class Boltz2Loss(LossTerm):
    joltz2: joltz.Joltz2
    features: PyTree
    loss: LossTerm | LinearCombination
    deterministic: bool = True
    recycling_steps: int = 0

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
        )

        return self.loss(
            sequence = sequence,
            output=output,
            key = key,
        )

def contact_cross_entropy(
    distogram_logits: Float[Array, "... N N 64"],
    contact_dist: float,
    min_dist=2.0,
    max_dist=22.0,
) -> Float[Array, "... N N"]:
    """Compute partial entropy (under distogram) that D_ij < contact_dist."""
    distogram_logits = jax.nn.log_softmax(distogram_logits)
    contact_idx = np.searchsorted(
        np.linspace(start=min_dist, stop=max_dist, num=distogram_logits.shape[-1]),
        contact_dist,
    )

    px_ = jax.nn.softmax(distogram_logits[..., :contact_idx], axis=-1)

    return (px_ * distogram_logits[..., :contact_idx]).sum(-1)


class WithinBinderContact(LossTerm):
    """Encourages contacts between residues."""

    max_contact_distance: float = 14.0
    min_sequence_separation: int = 8
    num_contacts_per_residue: int = 25

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: Boltz2Output,
        key,
    ):
        binder_len = sequence.shape[0]
        log_contact_intra = contact_cross_entropy(
            output.distogram_logits[:binder_len, :binder_len], self.max_contact_distance
        )
        # only count binder-binder contacts with sequence sep > min_sequence_separation
        within_binder_mask = (
            jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :])
            > self.min_sequence_separation
        )
        # for each position in binder find positions most likely to make contact
        binder_binder_max_p, _ = jax.vmap(
            lambda lcp: jax.lax.top_k(lcp, self.num_contacts_per_residue)
        )(log_contact_intra + (1 - within_binder_mask) * -30)
        average_log_prob = binder_binder_max_p.mean()

        return -average_log_prob, {"intra_contact": average_log_prob}


class BinderTargetContact(LossTerm):
    """Encourages contacts between binder and target."""

    paratope_idx: list[int] | None = None
    paratope_size: int | None = None
    contact_distance: float = 20.0
    epitope_idx: list[int] | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: Boltz2Output,
        key=None,
    ):
        binder_len = sequence.shape[0]
        log_contact_inter = contact_cross_entropy(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
        )
        if self.epitope_idx is not None:
            log_contact_inter = log_contact_inter[:, self.epitope_idx]

        # binder_target_max_p = log_contact_inter[:binder_len, binder_len:].max(-1)
        binder_target_max_p = jax.vmap(lambda v: jax.lax.top_k(v, 3)[0])(
            log_contact_inter
        ).mean(-1)
        # log probability of contacting target for each position in binder

        if self.paratope_idx is not None:
            binder_target_max_p = binder_target_max_p[self.paratope_idx]
        if self.paratope_size is not None:
            binder_target_max_p = jax.lax.top_k(
                binder_target_max_p, self.paratope_size
            )[0]

        average_log_prob = binder_target_max_p.mean()
        return -average_log_prob, {"target_contact": average_log_prob}
