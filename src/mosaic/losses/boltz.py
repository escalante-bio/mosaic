from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml

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

from .structure_prediction import PAE_BINS, StructureModelOutput


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

class ListFlowStyle(list):
    """Used to copy Boltz's specific yaml style"""

    pass


def represent_list_flowstyle(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(ListFlowStyle, represent_list_flowstyle)


def get_binder_yaml(
    binder_sequence: str | None = None,
    binder_len: int | None = None,
    use_msa: bool = False,
    chain: str = "A",
) -> list[dict]:
    """msa is usually "empty" (use_msa=False) during optimization"""

    if binder_sequence is None and binder_len is None:
        raise ValueError("Either binder_sequence or binder_len must be provided")

    binder_yaml = [
        {
            "protein": {
                "id": chain,
                "sequence": binder_sequence or "X" * binder_len,
            }
        }
    ]

    if use_msa is False:
        binder_yaml[-1]["protein"]["msa"] = "empty"

    return binder_yaml


def get_targets_yaml(
    sequence: str | list[str],
    entity_type: str | list[str] = "protein",
    use_msa: bool | list[bool] = False,
    chain: str = "B",
) -> list[dict]:
    """Assuming that usually the target is one protein or a list of proteins,
    flexibly allow entity_type and use_msa to be string/bool or a list.
    """

    ALL_CHAINS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Convert the inputs into a standardized list to iterate over
    if isinstance(sequence, str):
        if isinstance(entity_type, list) or isinstance(use_msa, list):
            raise ValueError(f"{entity_type=} and {use_msa=} must be str/bool")

        sequences = [sequence]
        entity_types = [entity_type]
        use_msas = [use_msa]
    else:
        sequences = sequence

        if isinstance(entity_type, list):
            assert len(entity_type) == len(sequences), f"wrong {len(entity_type)=}"
            entity_types = entity_type
        else:
            entity_types = [entity_type for _ in range(len(sequences))]

        if isinstance(use_msa, list):
            assert len(use_msa) == len(sequences), f"wrong {len(use_msa)=}"
            use_msas = use_msa
        else:
            use_msas = [use_msa for _ in range(len(sequences))]

    chains = ALL_CHAINS[ALL_CHAINS.index(chain) :]
    assert len(chains) >= len(sequences), "not enough chains available!"

    targets_yaml = []
    for sequence, entity_type, use_msa, chain in zip(
        sequences, entity_types, use_msas, chains
    ):
        targets_yaml.append({entity_type: {"id": chain, "sequence": sequence}})
        if use_msa is False:
            targets_yaml[-1][entity_type] |= {"msa": "empty"}

    return targets_yaml


def get_pocket_constraints_yaml(
    pocket_constraints: list[tuple[str, int]], binder_chain: str = "A"
) -> list[dict]:
    return [
        {
            "pocket": {
                "binder": binder_chain,
                "contacts": ListFlowStyle([list(c) for c in pocket_constraints]),
            }
        }
    ]


def get_bond_constraints_yaml(bond_constraints: list[dict]) -> list[dict]:
    if any(set(bond.keys()) != {"atom1", "atom2"} for bond in bond_constraints):
        raise ValueError("bond_constraints must have keys 'atom1' and 'atom2'")

    return [
        {
            "bond": {
                "atom1": ListFlowStyle(list(bond["atom1"])),
                "atom2": ListFlowStyle(list(bond["atom2"])),
            }
        }
        for bond in bond_constraints
    ]


def get_input_yaml(
    binder_sequence: str | None = None,
    binder_len: int | None = None,
    binder_use_msa: bool = False,
    binder_chain: str = "A",
    targets_sequence: str | list | None = None,
    targets_entity_type: str | list = "protein",
    targets_use_msa: bool | list = True,
    targets_chain: str = "B",
    pocket_constraints: list | None = None,
    bond_constraints: list | None = None,
) -> str:
    """Create a yaml file that includes binder and target sequences,
    plus optionally pocket constraints."""

    sequences = get_binder_yaml(
        binder_sequence, binder_len, binder_use_msa, binder_chain
    )

    sequences += get_targets_yaml(
        targets_sequence, targets_entity_type, targets_use_msa, targets_chain
    )

    constraints = []

    if pocket_constraints is not None:
        constraints += get_pocket_constraints_yaml(pocket_constraints, binder_chain)

    if bond_constraints is not None:
        constraints += get_bond_constraints_yaml(bond_constraints)

    boltz_yaml = {"sequences": sequences}
    boltz_yaml |= {"constraints": constraints} if constraints else {}

    return yaml.dump(boltz_yaml, indent=4, sort_keys=False, default_flow_style=False)


def make_binder_features(
    binder_len: int,
    target_sequence: str,
    target_polymer_type: str = "protein",
    use_msa=True,
    pocket_constraints=None,
    bond_constraints=None,
):
    return load_features_and_structure_writer(
        get_input_yaml(
            binder_len=binder_len,
            targets_sequence=target_sequence,
            targets_entity_type=target_polymer_type,
            targets_use_msa=use_msa,
            pocket_constraints=pocket_constraints,
            bond_constraints=bond_constraints,
        )
    )


def make_binder_monomer_features(monomer_len: int, out_dir: Path | None = None):
    return make_monomer_features(
        "X" * monomer_len, out_dir, use_msa=False, polymer_type="protein"
    )


def make_monomer_features(seq: str, use_msa=True, polymer_type: str = "protein"):
    return load_features_and_structure_writer(
        """
version: 1
sequences:
- {polymer_type}:
    id: [A]
    sequence: {seq}
    {msa}""".format(
            polymer_type=polymer_type,
            seq=seq,
            msa="msa: empty" if not use_msa else "",
        )
    )



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


BOLTZ1_DISTOGRAM_BINS = np.linspace(start=2.0, stop=22.0, num=64)


def boltz1_trunk(
    model: joltz.Joltz1,
    features: PyTree,
    *,
    recycling_steps: int,
    deterministic: bool,
    key: jax.Array,
) -> joltz.TrunkOutputs:
    """Run embedding + trunk recycling. Returns TrunkOutputs."""
    return model.trunk(
        features,
        recycling_steps=recycling_steps,
        key=key,
        deterministic=deterministic,
    )


def boltz1_forward_from_trunk(
    model: joltz.Joltz1,
    features: PyTree,
    trunk_outputs: joltz.TrunkOutputs,
    *,
    num_sampling_steps: int,
    deterministic: bool,
    key: jax.Array,
) -> StructureModelOutput:
    """Run distogram, structure, and confidence from pre-computed trunk output."""
    distogram_logits = trunk_outputs.pdistogram[0]  # strip batch dim

    structure_outputs = model.sample_structure(
        features,
        trunk_outputs,
        num_sampling_steps=num_sampling_steps,
        key=key,
    )

    confidence = model.predict_confidence(
        features,
        trunk_outputs,
        structure_outputs,
        key=key,
        deterministic=deterministic,
    )

    # Backbone coordinates (N, CA, C, O)
    features_unbatched = jax.tree.map(lambda x: x[0], features)
    assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
    first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
        features_unbatched["atom_to_token"].T
    )
    all_atom_coords = structure_outputs.sample_atom_coords[0]
    backbone_coordinates = jnp.stack(
        [all_atom_coords[first_atom_idx + i] for i in range(4)], -2
    )

    return StructureModelOutput(
        distogram_logits=distogram_logits,
        distogram_bins=BOLTZ1_DISTOGRAM_BINS,
        plddt=confidence["plddt"][0],
        pae=confidence["pae"][0],
        pae_logits=confidence["pae_logits"][0],
        pae_bins=PAE_BINS,
        structure_coordinates=structure_outputs.sample_atom_coords,
        backbone_coordinates=backbone_coordinates,
        full_sequence=features["res_type"][0][:, 2:22],
        asym_id=features["asym_id"][0],
        residue_idx=features["residue_index"][0],
    )


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
        features = set_binder_sequence(sequence, self.features)

        trunk_outputs = boltz1_trunk(
            self.joltz1, features,
            recycling_steps=self.recycling_steps,
            deterministic=self.deterministic,
            key=key,
        )
        output = boltz1_forward_from_trunk(
            self.joltz1, features, trunk_outputs,
            num_sampling_steps=self.sampling_steps,
            deterministic=self.deterministic,
            key=key,
        )

        v, aux = self.loss(sequence=sequence, output=output, key=key)
        return v, {self.name: aux}
