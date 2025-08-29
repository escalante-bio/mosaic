from protenix.config import parse_configs
from protenix.configs.configs_base import configs as configs_base
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.protenij import ConfidenceMetrics, InitialEmbedding, TrunkEmbedding
from protenix.protenij import Protenix as Protenij
from protenix.utils.torch_utils import dict_to_tensor

from mosaic.common import TOKENS
from mosaic.losses.structure_prediction import AbstractStructureOutput
from mosaic.common import LossTerm, LinearCombination, StateIndex
import copy
import json
from pathlib import Path
import equinox as eqx
from tempfile import TemporaryDirectory

import gemmi
import jax.numpy as jnp
import numpy as np
import protenix
import protenix.inference
import torch
from jaxtyping import Array, Float, PyTree
import jax
from dataclasses import dataclass
from functools import cached_property
from ml_collections.config_dict import ConfigDict
from protenix.data.constants import PRO_STD_RESIDUES
from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.model.protenix import Protenix
from protenix.protenij import from_torch
from protenix.runner import msa_search

# set "PROTENIX_DATA_ROOT_DIR" env variable

import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = str(Path("~/.protenix").expanduser())


def _load_model(name="protenix_mini_default_v0.5.0", cache_path = Path("~/.protenix")):
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=f"--model_name {name}",
        fill_required_with_null=True,
    )
    configs.update({"load_checkpoint_dir" : str(cache_path)})
    configs["data"]["pdb_cluster_file"] = str(cache_path / "clusters-by-entity-40.txt")
    model_specfics_configs = ConfigDict(model_configs[configs.model_name])
    configs.update(model_specfics_configs)
    protenix.inference.download_infercence_cache(configs)
    checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint = torch.load(checkpoint_path)
    sample_key = [k for k in checkpoint["model"].keys()][0]
    print(f"Sampled key: {sample_key}")
    if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
        checkpoint["model"] = {
            k[len("module.") :]: v for k, v in checkpoint["model"].items()
        }
    model = Protenix(configs)
    model.load_state_dict(state_dict=checkpoint["model"], strict=configs.load_strict)
    return from_torch(model)


def load_protenix_mini(cache_path = Path("~/.protenix")):
    return _load_model(name="protenix_mini_default_v0.5.0", cache_path=cache_path)


def load_protenix_tiny(cache_path = Path("~/.protenix")):
    return _load_model(name="protenix_tiny_default_v0.5.0", cache_path=cache_path)


def _process_one(single_sample_dict: dict[str], use_msa: bool = True):
    """
    Processes a single sample from the input JSON to generate features and statistics.

    Args:
        single_sample_dict: A dictionary containing the sample data.

    Returns:
        A tuple containing:
            - A dictionary of features.
            - An AtomArray object.
            - A dictionary of time tracking statistics.
    """
    # general features
    sample2feat = SampleDictToFeatures(
        single_sample_dict,
    )
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    entity_poly_type = sample2feat.entity_poly_type

    # Msa features
    entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
    msa_features = (
        InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=single_sample_dict["sequences"],
            entity_to_asym_id=entity_to_asym_id,
            token_array=token_array,
            atom_array=atom_array,
        )
        if use_msa
        else {}
    )

    # Make dummy features for not implemented features
    dummy_feats = ["template"]
    if len(msa_features) == 0:
        dummy_feats.append("msa")
    else:
        msa_features = dict_to_tensor(msa_features)
        features_dict.update(msa_features)
    features_dict = make_dummy_feature(
        features_dict=features_dict,
        dummy_feats=dummy_feats,
    )

    # Transform to right data type
    feat = data_type_transform(feat_or_label_dict=features_dict)

    data = {}
    data["input_feature_dict"] = feat

    # Add dimension related items
    N_token = feat["token_index"].shape[0]
    N_atom = feat["atom_to_token_idx"].shape[0]
    N_msa = feat["msa"].shape[0]

    stats = {}
    for mol_type in ["ligand", "protein", "dna", "rna"]:
        mol_type_mask = feat[f"is_{mol_type}"].bool()
        stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
        stats[f"{mol_type}/token"] = len(
            torch.unique(feat["atom_to_token_idx"][mol_type_mask])
        )

    N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
    data.update(
        {
            "N_asym": torch.tensor([N_asym]),
            "N_token": torch.tensor([N_token]),
            "N_atom": torch.tensor([N_atom]),
            "N_msa": torch.tensor([N_msa]),
        }
    )

    def formatted_key(key):
        type_, unit = key.split("/")
        if type_ == "protein":
            type_ = "prot"
        elif type_ == "ligand":
            type_ = "lig"
        else:
            pass
        return f"N_{type_}_{unit}"

    data.update(
        {
            formatted_key(k): torch.tensor([stats[k]])
            for k in [
                "protein/atom",
                "ligand/atom",
                "dna/atom",
                "rna/atom",
                "protein/token",
                "ligand/token",
                "dna/token",
                "rna/token",
            ]
        }
    )
    data.update({"entity_poly_type": entity_poly_type})

    return data, atom_array


def load_features_from_json(json_data: dict, add_msa=True):
    with TemporaryDirectory() as _d:
        d = Path(_d)
        name = json_data["name"]
        if add_msa:
            p = d / (name + ".json")
            p.write_text(json.dumps([json_data]))
            msa_search.update_infer_json(str(p), str(d))
            json_data = json.loads((d / f"{name}-add-msa.json").read_text())[0]

        features, biotite_array = _process_one(json_data)
        features = from_torch(features["input_feature_dict"]) | {
            "atom_rep_atom_idx": np.array(
                features["input_feature_dict"]["distogram_rep_atom_mask"]
            ).nonzero()[0]
        }
        return features, biotite_array


def biotite_atom_to_gemmi_atom(atom):
    ga = gemmi.Atom()
    ga.pos = gemmi.Position(*atom.coord)
    ga.element = gemmi.Element(atom.element)
    ga.name = atom.atom_name
    return ga


def new_gemmi_residue(atom):
    r = gemmi.Residue()
    r.name = atom.res_name
    r.seqid = gemmi.SeqId(atom.res_id, " ")
    r.entity_type = gemmi.EntityType.Polymer
    return r


def biotite_array_to_gemmi_struct(atom_array, pred_coord=None, per_atom_plddt=None):
    if pred_coord is not None:
        atom_array = copy.deepcopy(atom_array)
        atom_array.coord = pred_coord
    structure = gemmi.Structure()
    model = gemmi.Model("0")
    chains = {}
    for atom_idx, atom in enumerate(atom_array):
        chain = chains.setdefault(atom.chain_id, {})
        residue = chain.setdefault(int(atom.res_id), new_gemmi_residue(atom))
        gemmi_atom = biotite_atom_to_gemmi_atom(atom)
        if per_atom_plddt is not None:
            gemmi_atom.b_iso = per_atom_plddt[atom_idx]
        residue.add_atom(gemmi_atom)
    for k in chains:
        chain = gemmi.Chain(k)
        chain.append_residues(list(chains[k].values()))
        model.add_chain(chain)
    structure.add_model(model)
    return structure


def boltz_to_protenix_matrix():
    T = np.zeros((len(TOKENS), 32))
    for i, tok in enumerate(TOKENS):
        protenix_idx = PRO_STD_RESIDUES[
            gemmi.expand_one_letter(tok, gemmi.ResidueKind.AA)
        ]
        T[i, protenix_idx] = 1
    return T


def set_binder_sequence(new_sequence: Float[Array, "N 20"], features: PyTree):
    binder_len = new_sequence.shape[0]
    protenix_sequence = new_sequence @ boltz_to_protenix_matrix()
    n_msa = features["msa"].shape[0]
    print("n_msa", n_msa)

    zero_msa_idx = 20  # GAP #31#20
    n_fake_seq = 1

    # TODO: we may need to be more aggressive here and upweight the profile
    # We assume there are no MSA hits for the binder sequence
    binder_profile = jnp.zeros_like(features["profile"][:binder_len])
    binder_profile = (
        binder_profile.at[:binder_len].set(protenix_sequence) * n_fake_seq / n_msa
    )
    binder_profile = binder_profile.at[:, zero_msa_idx].set(
        (n_msa - n_fake_seq) / n_msa
    )
    # binder_profile = protenix_sequence
    return features | {
        "restype": features["restype"].at[:binder_len, :].set(protenix_sequence),
        # "msa": features["msa"].at[:, :binder_len].set(protenix_sequence.argmax(-1)),
        "profile": features["profile"].at[:binder_len].set(binder_profile),
    }


@dataclass
class ProtenixOutput(AbstractStructureOutput):
    model: Protenij
    features: PyTree
    key: jax.Array
    initial_recycling_state: TrunkEmbedding | None = None
    recycling_steps: int = 0
    sampling_steps: int = 2
    n_structures: int = 1

    @property
    def full_sequence(self):
        return self.features["restype"] @ boltz_to_protenix_matrix().T

    @property
    def asym_id(self):
        return self.features["asym_id"]

    @property
    def residue_idx(self):
        return self.features["residue_index"]

    @cached_property
    def initial_embedding(self) -> InitialEmbedding:
        return self.model.embed_inputs(input_feature_dict=self.features)

    @cached_property
    def trunk_state(self) -> TrunkEmbedding:
        print("JIT compiling protenix trunk module...")
        return self.model.recycle(
            initial_embedding=self.initial_embedding,
            recycling_steps=self.recycling_steps,
            input_feature_dict=self.features,
            key=self.key,
            state=self.initial_recycling_state,
        )

    @property
    def distogram_bins(self) -> Float[Array, "64"]:
        return np.linspace(start=2.3125, stop=21.6875, num=64)

    @cached_property
    def distogram_logits(self) -> Float[Array, "N N 64"]:
        return self.model.distogram_head(self.trunk_state.z)

    @cached_property
    def structure_coordinates(self):
        print("JIT compiling structure module...")
        return self.model.sample_structures(
            initial_embedding=self.initial_embedding,
            trunk_embedding=self.trunk_state,
            input_feature_dict=self.features,
            N_samples=self.n_structures,
            N_steps=self.sampling_steps,
            key=self.key,
        )

    @cached_property
    def confidence_metrics(self) -> ConfidenceMetrics:
        print("JIT compiling confidence module...")
        return self.model.confidence_metrics(
            initial_embedding=self.initial_embedding,
            trunk_embedding=self.trunk_state,
            input_feature_dict=self.features,
            coordinates=self.structure_coordinates,
            key=self.key,
        )

    @property
    def plddt(self) -> Float[Array, "N"]:
        """PLDDT *normalized* to between 0 and 1."""
        return (
            jax.nn.softmax(
                self.confidence_metrics.plddt_logits[0][
                    self.features["atom_rep_atom_idx"]
                ]
            )
            * jnp.linspace(0, 1, 50)[None, :]
        ).sum(-1)

    @property
    def pae(self) -> Float[Array, "N N"]:
        return (
            (
                jax.nn.softmax(self.confidence_metrics.pae_logits)
                * self.pae_bins[None, None, :]
            ).sum(-1)
        ).mean(0)

    @property
    def pae_logits(self) -> Float[Array, "N N 64"]:
        return self.confidence_metrics.pae_logits[0]

    @property
    def pae_bins(self) -> Float[Array, "64"]:
        end = 32.0
        num_bins = 64
        bin_width = end / num_bins
        return np.arange(start=0.5 * bin_width, stop=end, step=bin_width)

    @property
    def backbone_coordinates(self) -> Float[Array, "N 4"]:
        features = self.features
        # In order these are N, C-alpha, C, O
        # assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
        # first step, which is a bit cryptic, is to get the first atom for each token
        n_tokens = features["restype"].shape[0]
        first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
            (features["atom_to_token_idx"][:, None] == jnp.arange(n_tokens)[None, :]).T
        )
        # NOTE: this will completely (and silently) fail if any tokens are non-protein!
        # take first diffusion sample?
        all_atom_coords = self.structure_coordinates[0]
        coords = jnp.stack([all_atom_coords[first_atom_idx + i] for i in range(4)], -2)
        return coords


class ProtenixCoords(eqx.Module):
    coords: Float[Array, "N_samples N 3"]
    plddt: Float[Array, "N_samples N"]


class ProtenixLoss(LossTerm):
    model: Protenij
    features: PyTree
    loss: LossTerm | LinearCombination
    initial_recycling_state: TrunkEmbedding
    recycling_steps: int = 1
    sampling_steps: int = 5
    n_structures: int = 1
    state_index: StateIndex = eqx.field(default_factory=StateIndex)
    name: str = "protenix"
    return_coords: bool = False

    def __call__(self, sequence: Float[Array, "N 20"], key):
        """Compute the loss for a given sequence."""
        # Set the binder sequence in the features
        features = set_binder_sequence(sequence, self.features)
        # features = self.features

        # initialize lazy output object
        output = ProtenixOutput(
            model=self.model,
            features=features,
            key=key,
            recycling_steps=self.recycling_steps,
            sampling_steps=self.sampling_steps,
            initial_recycling_state=self.initial_recycling_state,
            n_structures=self.n_structures,
        )

        v, aux = self.loss(
            sequence=sequence,
            output=output,
            key=key,
        )

        coords = (
            {
                "coords": ProtenixCoords(
                    output.structure_coordinates,
                    (
                        jax.nn.softmax(output.confidence_metrics.plddt_logits)
                        * jnp.linspace(0, 1, 50)[None, None, :]
                    ).sum(-1),
                )
            }
            if self.return_coords
            else {}
        )

        # nested dict to get around jax incomparable keys issue...
        return v, {
            self.name: aux,
            "state_index": (self.state_index, output.trunk_state),
        } | coords

    def update_state(self, update):
        return eqx.tree_at(lambda s: s.initial_recycling_state, self, update)
