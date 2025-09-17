from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import jax
import jax.numpy as jnp
import joltz
import numpy as np
import torch
from boltz.data.const import ref_atoms
from boltz.main import (
    BoltzDiffusionParams,
    BoltzInferenceDataModule,
    BoltzProcessedInput,
    BoltzWriter,
    Manifest,
    check_inputs,
    process_inputs,
    download_boltz1 as download,
)
from boltz.model.models.boltz1 import Boltz1
from jax import tree
from jaxtyping import Array, Float, PyTree

from ..common import LinearCombination, LossTerm

from .structure_prediction import AbstractStructureOutput


def load_boltz(
    checkpoint_path: Path = Path("~/.boltz/boltz1_conf.ckpt").expanduser(),
):
    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": 25,
        "diffusion_samples": 1,
    }
    if not checkpoint_path.exists():
        print(f"Downloading Boltz checkpoint to {checkpoint_path}")
        cache = checkpoint_path.parent
        cache.mkdir(parents=True, exist_ok=True)
        download(cache)

    _torch_model = Boltz1.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
    )

    model = joltz.from_torch(_torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    model = eqx.combine(jax.device_put(_model_params), _model_static)

    return model


class StructureWriter:
    """
    Hacky class to write predicted structures to disk using a BoltzWriter
    """

    writer: BoltzWriter
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
        self.writer = BoltzWriter(
            data_dir=target_dir,
            output_dir=output_dir,
            output_format="mmcif",
        )
        self.atom_pad_mask = features_dict["atom_pad_mask"].unsqueeze(0)
        self.record = features_dict["record"][0]
        self.out_dir = output_dir
        self.temp_dir_handle = temp_dir_handle

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
        return (Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.cif"



def load_features_and_structure_writer(
    input_yaml_str: str,
    cache=Path("~/.boltz/").expanduser(),
) -> tuple[PyTree, StructureWriter]:
    print("Loading data")
    out_dir_handle = (
        TemporaryDirectory()
    )  # this is sketchy -- we have to remember not to let this get garbage collected
    out_dir = Path(out_dir_handle.name)
    # dump the yaml to a file
    input_data_path = out_dir / "input.yaml"
    input_data_path.write_text(input_yaml_str)
    data = check_inputs(input_data_path)
    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=cache / "mols",
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
        temp_dir_handle=out_dir_handle,
    )
    return features, writer


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree,
):
    """Replace features related to first N tokens with `new_sequence.` Used for hallucination/binder design."""
    features = tree.map(lambda v: v.astype(jnp.float32), features)
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


@dataclass
class Boltz1Output(AbstractStructureOutput):
    joltz: joltz.Joltz1
    features: PyTree
    deterministic: bool
    key: jax.Array
    recycling_steps: int = 0
    num_sampling_steps: int = 25

    @property
    def full_sequence(self):
        return self.features["res_type"][0][:, 2:22]

    @property
    def asym_id(self):
        return self.features["asym_id"][0]

    @property
    def residue_idx(self):
        return self.features["residue_index"][0]

    @property
    def distogram_bins(self) -> Float[Array, "64"]:
        return np.linspace(start=2.0, stop=22.0, num=64)

    @cached_property
    def trunk_outputs(self) -> joltz.TrunkOutputs:
        return self.joltz.trunk(
            self.features,
            recycling_steps=self.recycling_steps,
            key=self.key,
            deterministic=self.deterministic,
        )

    @cached_property
    def distogram_logits(self) -> Float[Array, "N N 64"]:
        return self.trunk_outputs.pdistogram[0]  # strip batch dim

    @cached_property
    def structure_outputs(self) -> joltz.StructureModuleOutputs:
        print("JIT compiling boltz1 structure module...")
        return self.joltz.sample_structure(
            self.features,
            self.trunk_outputs,
            num_sampling_steps=self.num_sampling_steps,
            key=self.key,
        )

    @cached_property
    def confidence_outputs(self) -> PyTree:
        print("JIT compiling boltz1 confidence module...")
        return self.joltz.predict_confidence(
            self.features,
            self.trunk_outputs,
            self.structure_outputs,
            key=self.key,
            deterministic=self.deterministic,
        )

    @property
    def plddt(self) -> Float[Array, "N"]:
        return self.confidence_outputs["plddt"][0]

    @property
    def pae(self) -> Float[Array, "N N"]:
        return self.confidence_outputs["pae"][0]

    @property
    def pae_logits(self) -> Float[Array, "N N 64"]:
        return self.confidence_outputs["pae_logits"][0]

    @property
    def pae_bins(self) -> Float[Array, "64"]:
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
        all_atom_coords = self.structure_output.sample_atom_coords[0]
        coords = jnp.stack([all_atom_coords[first_atom_idx + i] for i in range(4)], -2)
        return coords


class Boltz1Loss(LossTerm):
    joltz1: joltz.Joltz1
    features: PyTree
    loss: LossTerm | LinearCombination
    deterministic: bool = True
    recycling_steps: int = 0
    sampling_steps: int = 25
    name: str = "boltz1"

    def __call__(self, sequence: Float[Array, "N 20"], key=None):
        """Compute the loss for a given sequence."""
        # Set the binder sequence in the features
        features = set_binder_sequence(sequence, self.features)

        # initialize lazy output object
        output = Boltz1Output(
            joltz=self.joltz1,
            features=features,
            deterministic=self.deterministic,
            key=key,
            recycling_steps=self.recycling_steps,
            num_sampling_steps=self.sampling_steps,
        )

        v, aux = self.loss(
            sequence=sequence,
            output=output,
            key=key,
        )
        return v, {self.name: aux}
