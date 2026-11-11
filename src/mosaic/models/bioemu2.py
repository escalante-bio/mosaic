"""BioEmu2 on jfold: features by `jfold.featurize` (torch-free and deterministic), jfold's
converted BioEmu2 (`jfold.pretrained`) for the forward pass.

BioEmu2 is a generative model of structure ensembles without confidence heads: every
`StructureModelOutput` / `StructurePrediction` carries NaN `plddt`, `pae` and `iptm`.
`mode` selects upstream's method conditioning and sampler preset: ``"structure"``
(structure prediction) or ``"equilibrium"`` (equilibrium ensembles).

Features are BioEmu2's feature dict (`feature_dict`). jfold's view (`featurize`) takes
protein chains of the 20 amino acids and X, without templates; each chain with `use_msa`
keeps its own MSA rows (its query is its paired MSA). For DNA, RNA or bioemu2's own MSAs,
`mosaic.models.upstream.bioemu2.features` featurizes bioemu2's input (torch) to the same
feature dict.

Sampling an ensemble: run the trunk once with `BioEmu2.denoiser`, then `jax.vmap`
`mosaic.sampling.sample` over keys::

    model = BioEmu2(mode="equilibrium")
    features, writer = model.target_only_features(
        [TargetChain(sequence, use_msa=False)], key=feature_key
    )
    denoise, atom_mask = model.denoiser(features=features, key=key)
    sampler = model.upstream_sampler()
    coords = jax.vmap(lambda k: sample(denoise, atom_mask, sampler, key=k))(keys)
    structures = [model.structure(writer, x) for x in coords]

The first use downloads the converted model into mosaic's `jfold` cache.
"""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from jfold import pretrained
from jfold.featurize.views import bioemu2 as view
from jfold.model import StructureModel
from jfold.models import bioemu2 as jbioemu2
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.models.jfold_featurize import AtomSitesWriter, check_chains, jfold_system
from mosaic.sampling import Denoise
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
)

UPSTREAM = (
    "mosaic.models.upstream.bioemu2.features featurizes bioemu2's own input "
    "(install mosaic[all])"
)
# Upstream's sampler presets (`EDMSampler.structure_prediction` / `.equilibrium`).
STEP_SCALE = {"structure": 1.5, "equilibrium": 1.0}
# BioEmu2 keeps Boltz-1's distogram head (64 bins, 2-22 Å).
DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
# Upstream's inference default (`num_recycles=3`).
INFERENCE_RECYCLING_STEPS = 4
# A featurization's writer: `AtomSitesWriter` (jfold's) or upstream's `CifWriter`.
Writer = Callable[[Array], gemmi.Structure]
# `mol_type` of protein atoms.
PROTEIN = 0
# The token class of each of mosaic's amino acids (`TOKENS` order).
AMINO_ACIDS = np.array([view.TOKEN[aa] for aa in TOKENS])


def feature_dict(features: dict[str, np.ndarray]) -> dict[str, Array]:
    """A featurization's dict as JAX arrays, with `res_type` and `msa` as probabilities over
    the token types (a soft sequence replaces them) and each atom's atom37 slot
    (`atom37_idx`, -1 for none: padding and non-protein atoms). Atom names are decoded from
    `name_encoding` (`ord(c) - 32` per character, 0 = end)."""
    names = [
        "".join(chr(c + 32) for c in enc if c) for enc in features["name_encoding"]
    ]
    protein = features["atom_pad_mask"] & (features["mol_type"] == PROTEIN)
    atom37_idx = [
        ATOM37_INDEX.get(name, -1) if real else -1
        for name, real in zip(names, protein, strict=True)
    ]
    return {k: jnp.asarray(v) for k, v in features.items()} | {
        "res_type": jax.nn.one_hot(features["res_type"], view.NUM_TOKEN_TYPES),
        "msa": jnp.asarray(features["msa"], jnp.float32),
        "atom37_idx": jnp.array(atom37_idx, jnp.int32),
    }


def featurize(
    chains: list[TargetChain], *, mode: str
) -> tuple[dict[str, Array], AtomSitesWriter]:
    """`chains` featurized by jfold's view in `mode` (each chain's query as its paired MSA),
    and their writer."""
    check_chains(chains, "BioEmu2", templates=False, upstream=UPSTREAM)
    system = jfold_system(chains)
    features = view.features(system, mode=mode, paired_queries=True)
    return feature_dict(features), AtomSitesWriter(view.atom_sites(system))


def set_binder_sequence(
    pssm: Float[Array, "L 20"], features: dict[str, Array]
) -> dict[str, Array]:
    """Splice the soft sequence into `res_type`, `profile` and the MSA query row of the
    first `L` tokens.

    `profile` is the mean over all MSA rows, so swapping the query row is a rank-one update
    by 1 / depth that keeps the rest of the MSA's statistics."""
    L = pssm.shape[0]
    soft = jnp.zeros((L, view.NUM_TOKEN_TYPES), pssm.dtype).at[:, AMINO_ACIDS].set(pssm)
    msa = features["msa"]
    profile = features["profile"][:L] + (soft - msa[0, :L]) / msa.shape[0]
    return features | {
        "res_type": features["res_type"].at[:L].set(soft),
        "profile": features["profile"].at[:L].set(profile),
        "msa": msa.at[0, :L].set(soft),
    }


class BioEmu2(DiffusionStructurePredictionModel[dict]):
    """BioEmu2 on jfold, with upstream's features, recycling counts and sampler presets."""

    model: StructureModel
    mode: str = eqx.field(static=True)  # "structure" or "equilibrium"
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    distogram_bins = DISTOGRAM_BINS

    def __init__(
        self,
        mode: str = "structure",
        *,
        model: StructureModel | None = None,
        design_recycling_steps: int = 1,
        inference_recycling_steps: int = INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
    ):
        """Loads the converted model unless `model` is passed. Its sampler settings become
        upstream's preset for `mode` (the presets differ only in `step_scale`)."""
        if mode not in STEP_SCALE:
            raise ValueError(f"mode must be one of {sorted(STEP_SCALE)}, got {mode!r}")
        if model is None:
            model = pretrained.load("bioemu2", cache=cache_dir("jfold"))
        sampling = dataclasses.replace(model.sampling, step_scale=STEP_SCALE[mode])
        self.model = eqx.tree_at(lambda m: m.sampling, model, sampling)
        self.mode = mode
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    def target_only_features(self, chains: list[TargetChain], *, key: PRNGKeyArray):
        """Featurize the chains as given (with `mode`'s method conditioning). Returns
        ``(features, writer)``. Deterministic: `key` is unused."""
        del key
        return featurize(chains, mode=self.mode)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        """Design features: a single-sequence binder of unknown residues (chain 0) plus the
        targets; the PSSM replaces its sequence."""
        binder = TargetChain("X" * binder_length, use_msa=False)
        return self.target_only_features([binder, *chains], key=key)

    def set_sequence(
        self,
        features: dict[str, Array],
        sequence: Float[Array, "L 20"],
        *,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        del key
        return set_binder_sequence(sequence, features)

    def tokens(self, features: dict[str, Array]) -> TokenInfo:
        return TokenInfo(
            sequence=features["res_type"][:, AMINO_ACIDS],
            asym_id=features["asym_id"],
            residue_index=features["residue_index"],
            mask=features["token_pad_mask"].astype(bool),
        )

    def atom37(self, features: dict[str, Array], coords: Float[Array, "N 3"]) -> Atom37:
        atom37_coords, atom37_mask = scatter_atom37(
            atom_coords=coords,
            atom_to_token=features["token_idx_at_atom"],
            atom37_idx=features["atom37_idx"],
            n_token=features["res_type"].shape[0],
        )
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    @staticmethod
    def structure(writer: Writer, coords: Float[Array, "N 3"]) -> gemmi.Structure:
        """One sample as a gemmi structure, all atoms, by the featurization's writer."""
        return writer(coords)

    def to_structure(
        self,
        features: dict[str, Array],
        writer: Writer,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """With a designed sequence, the atom37 view; otherwise all atoms via `writer`."""
        del features
        if sequence is not None:
            return output.to_structure()
        return self.structure(writer, output.structure_coordinates)

    def jfold_features(self, features: dict[str, Array]) -> JFeatures:
        """jfold's features, BioEmu2's conditioning included. Only unconditioned dynamics
        (equilibrium sampling, structure prediction) are supported."""
        return jbioemu2.features(features)

    @eqx.filter_jit
    def denoiser(
        self,
        *,
        features: dict[str, Array],
        PSSM: Float[Array, "N 20"] | None = None,
        recycling_steps: int | None = None,
        key: PRNGKeyArray,
    ) -> tuple[Denoise, Float[Array, " A"]]:
        """Run the trunk once: the denoiser and atom mask for `mosaic.sampling.sample`."""
        if recycling_steps is None:
            recycling_steps = self.inference_recycling_steps
        k_prepare, k_trunk = jax.random.split(key)
        feats, inputs = self.prepare(features, PSSM, key=k_prepare)
        trunk = self.run_trunk(
            feats, inputs, recycling_steps=recycling_steps, key=k_trunk
        )
        return self.structure_denoiser(trunk)
