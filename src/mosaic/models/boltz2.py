"""Boltz-2 on jfold: features by `jfold.featurize`'s Boltz-2 view (torch-free; the dict
upstream Boltz-2 builds, up to the random pose of each residue's reference conformer),
jfold's converted Boltz-2 (`jfold.pretrained`) for the forward pass, and the view's atom
sites for writing structures.

Features are the view's dict, unbatched, with `msa` one-hot (`[S, T, 33]`) so that a
binder's soft sequence can be set. The view takes protein chains of the 20 amino acids and
X, a template per chain and unpaired MSAs; anything else raises `NotImplementedError`.
`mosaic.models.upstream.boltz2.features` featurizes boltz's own input instead (DNA, RNA,
ligands, paired MSAs, ...), into the same features."""

from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from jfold import pretrained
from jfold.featurize.system import THREE_LETTER
from jfold.featurize.views import boltz2 as view
from jfold.model import StructureModel
from jfold.models import boltz2 as jboltz2
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir, resolve_cache
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.models.jfold_featurize import (
    AtomSitesWriter,
    check_chains,
    check_unpaired_msas,
    jfold_system,
)
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
)

DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
# Trunk passes for `model_output` and the loss builders, and for `predict` (upstream
# `boltz predict --recycling_steps 3`).
DESIGN_RECYCLING_STEPS = 1
INFERENCE_RECYCLING_STEPS = 4


def _atom37_table() -> np.ndarray:
    """Each residue class's atoms (`[33, max atoms]`, in the view's atom order) as atom37
    slots; -1 where there is none."""
    residues = view.residues()
    atoms = {
        token: residues[THREE_LETTER[letter]].atoms
        for letter, token in view.TOKEN.items()
    }
    table = -np.ones((view.NUM_TOKENS, max(map(len, atoms.values()))), np.int32)
    for token, names in atoms.items():
        table[token, : len(names)] = [ATOM37_INDEX.get(name, -1) for name in names]
    return table


_ATOM37_TABLE = _atom37_table()

UPSTREAM = (
    "mosaic.models.upstream.boltz2.features featurizes boltz's own input "
    "(install mosaic[all])"
)


def target_only_features(chains: list[TargetChain], *, key: PRNGKeyArray):
    """The view's features for `chains` (see the module docstring) and a writer. `key`
    poses the reference conformers, as upstream does with torch's global RNG."""
    check_chains(chains, "Boltz-2", templates=True, upstream=UPSTREAM)
    check_unpaired_msas(chains, "Boltz-2", upstream=UPSTREAM)
    system = jfold_system(chains)
    features = view.features(system, key=key)
    features["msa"] = np.eye(view.NUM_TOKENS, dtype=np.float32)[features["msa"]]
    return jax.tree.map(jnp.array, features), AtomSitesWriter(view.atom_sites(system))


def set_binder_sequence(sequence: Float[Array, "N 20"], features: dict) -> dict:
    """`features` with the first `N` tokens (a binder without MSA hits) set to the soft
    `sequence`: residue type, query MSA row and profile."""
    num_residues = sequence.shape[0]
    tokens = jnp.pad(sequence, ((0, 0), (2, view.NUM_TOKENS - 22)))
    num_rows = features["msa"].shape[0]
    profile = (tokens / num_rows).at[:, view.GAP].set((num_rows - 1) / num_rows)
    return features | {
        "res_type": features["res_type"]
        .astype(jnp.float32)
        .at[:num_residues]
        .set(tokens),
        "msa": features["msa"].at[0, :num_residues].set(tokens),
        "profile": features["profile"].at[:num_residues].set(profile),
    }


def boltz2_atom37(features: dict, coords: Float[Array, "A 3"]):
    """Atom37 coordinates and mask of Boltz-2 coordinates (`[A, 3]`), each token's atoms
    named by its residue type's argmax."""
    atom_to_token = features["atom_to_token"].argmax(-1)
    first_atom = features["atom_to_token"].argmax(0)
    position = jnp.arange(atom_to_token.shape[0]) - first_atom[atom_to_token]
    residue = features["res_type"].argmax(-1)[atom_to_token]
    slot = jnp.asarray(_ATOM37_TABLE)[residue, position]
    slot = jnp.where(features["atom_pad_mask"] > 0, slot, -1)
    return scatter_atom37(coords, atom_to_token, slot, features["res_type"].shape[0])


class Boltz2(DiffusionStructurePredictionModel[dict]):
    """Boltz-2 on jfold (see the module docstring)."""

    model: StructureModel
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    distogram_bins = DISTOGRAM_BINS

    def __init__(
        self,
        cache_path: Path | None = None,
        *,
        model: StructureModel | None = None,
        design_recycling_steps: int = DESIGN_RECYCLING_STEPS,
        inference_recycling_steps: int = INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
    ):
        """Loads the converted model unless `model` is passed: converted from the torch
        checkpoint (`cache_path`, or mosaic's Boltz cache) on first use, then read from
        mosaic's `jfold` cache."""
        if model is None:
            checkpoint = resolve_cache(cache_path, "boltz", "boltz2_conf.ckpt")
            model = pretrained.load(
                "boltz2", checkpoint_path=checkpoint, cache=cache_dir("jfold")
            )
        self.model = model
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    def target_only_features(self, chains: list[TargetChain], *, key: PRNGKeyArray):
        return target_only_features(chains, key=key)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        return self.target_only_features(
            [TargetChain(sequence="X" * binder_length, use_msa=False)] + chains, key=key
        )

    def set_sequence(
        self, features: dict, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ) -> dict:
        del key
        return set_binder_sequence(sequence, features)

    def jfold_features(self, features: dict) -> JFeatures:
        return jboltz2.features(features)

    def tokens(self, features: dict) -> TokenInfo:
        return TokenInfo(
            sequence=features["res_type"][:, 2:22],
            asym_id=features["asym_id"],
            residue_index=features["residue_index"],
            mask=features["token_pad_mask"].astype(bool),
        )

    def atom37(self, features: dict, coords: Float[Array, "A 3"]) -> Atom37:
        atom37_coords, atom37_mask = boltz2_atom37(features, coords)
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(
        self, features: dict, writer, output: StructureModelOutput, sequence
    ) -> gemmi.Structure:
        del features, sequence
        return writer(output.structure_coordinates)
