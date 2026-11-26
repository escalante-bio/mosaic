"""Boltz-2 backend running on jfold.

Inputs are featurized with `jfold.featurize`, which builds the same features as upstream
Boltz-2 apart from the random placement of reference conformers. The forward pass uses
jfold's converted Boltz-2 weights from `jfold.pretrained`.

Supported inputs are protein chains of the 20 amino acids and X, with templates and
unpaired MSAs. Other inputs, such as nucleic acids, ligands or paired MSAs, can be
featurized from a boltz input with `mosaic.models.upstream.boltz2.features`."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from jfold import pretrained
from jfold.featurize.views import boltz2 as view
from jfold.model import StructureModel
from jfold.models import boltz2 as jboltz2
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir, resolve_cache
from mosaic.models.jfold_featurize import (
    check_chains,
    check_unpaired_msas,
    jfold_system,
)
from mosaic.structure_prediction import (
    ATOM37_NAMES,
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
    scatter_atom37,
)

DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
# Number of trunk passes for design and for `predict`, which matches upstream
# `boltz predict --recycling_steps 3`.
DESIGN_RECYCLING_STEPS = 1
INFERENCE_RECYCLING_STEPS = 4


_ATOM37_CHARS = np.array(
    [[ord(c) - 32 for c in name.ljust(4)] for name in ATOM37_NAMES]
)

UPSTREAM = (
    "mosaic.models.upstream.boltz2.features featurizes boltz's own input "
    "(install mosaic[all])"
)


def target_only_features(chains: list[TargetChain], *, key: PRNGKeyArray):
    """Featurize `chains`.

    The MSA is one-hot encoded so a binder's soft sequence can be set. `key` randomly
    rotates and translates each residue's reference conformer, as upstream does with
    torch's global RNG."""
    check_chains(chains, "Boltz-2", templates=True, upstream=UPSTREAM)
    check_unpaired_msas(chains, "Boltz-2", upstream=UPSTREAM)
    system = jfold_system(chains)
    features = view.features(system, key=key)
    features["msa"] = np.eye(view.NUM_TOKENS, dtype=np.float32)[features["msa"]]
    return jax.tree.map(jnp.array, features)


def set_binder_sequence(sequence: Float[Array, "N 20"], features: dict) -> dict:
    """Set the residue type, MSA query row and profile of the binder to `sequence`.

    The binder is the first `N` tokens and has no MSA."""
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
    """Scatter Boltz-2 atom coordinates of shape `[A, 3]` to atom37.

    Keep the physical layout, including any placeholder side-chain atoms. For residue-tokenized
    modified proteins, keep recognized names. The upstream writer preserves the CCD identity."""
    atom_to_token = features["atom_to_token"].argmax(-1)
    names = features["ref_atom_name_chars"].argmax(-1)
    matches = (names[:, None, :] == _ATOM37_CHARS).all(-1)
    protein = features["mol_type"] == 0
    valid = (features["atom_pad_mask"] > 0) & protein[atom_to_token] & matches.any(-1)
    slot = jnp.where(valid, matches.argmax(-1), -1)
    return scatter_atom37(coords, atom_to_token, slot, features["mol_type"].shape[0])


class Boltz2(DiffusionStructurePredictionModel[dict]):
    """Boltz-2 running on jfold."""

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
        """Wrap `model`, or load the converted Boltz-2 checkpoint if it is None.

        On first use the torch checkpoint in `cache_path`, or mosaic's Boltz cache, is
        converted and saved to mosaic's `jfold` cache."""
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
            residue_index=features["residue_index"] + 1,
            mask=features["token_pad_mask"].astype(bool),
        )

    def atom37(self, features: dict, coords: Float[Array, "A 3"]) -> Atom37:
        atom37_coords, atom37_mask = boltz2_atom37(features, coords)
        return Atom37(coords=atom37_coords, mask=atom37_mask)
