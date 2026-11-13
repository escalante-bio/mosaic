"""BioEmu2 backend running on jfold.

Inputs are featurized deterministically with `jfold.featurize`, and the forward pass uses
jfold's converted BioEmu2 weights from `jfold.pretrained`.

BioEmu2 generates structure ensembles and has no confidence heads, so `plddt`, `pae` and
`iptm` are always NaN. `mode` selects upstream's conditioning and sampler preset, either
``"structure"`` for structure prediction or ``"equilibrium"`` for equilibrium ensembles.

jfold supports protein chains of the 20 amino acids and X, without templates. For nucleic
acids or bioemu2's own MSAs, use `mosaic.models.upstream.bioemu2.features`, which needs
torch.

To sample an ensemble, run the trunk once with `BioEmu2.denoiser` and then `jax.vmap`
`jfold.sampling.sample` over keys::

    model = BioEmu2(mode="equilibrium")
    features, writer = model.target_only_features(
        [TargetChain(sequence, use_msa=False)], key=feature_key
    )
    denoise, atom_mask = model.denoiser(features=features, key=key)
    sampler = model.upstream_sampler()
    coords = jax.vmap(lambda k: sample(denoise, atom_mask, sampler, key=k))(keys)
    structures = [model.structure(writer, x) for x in coords]
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
from jfold.sampling import Denoise
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.models.jfold_featurize import AtomSitesWriter, check_chains, jfold_system
from mosaic.structure_prediction import (
    ATOM37_INDEX,
    Atom37,
    DiffusionStructurePredictionModel,
    StructureModelOutput,
    TargetChain,
    TokenInfo,
    scatter_atom37,
)

UPSTREAM = (
    "mosaic.models.upstream.bioemu2.features featurizes bioemu2's own input "
    "(install mosaic[all])"
)
# Upstream's sampler presets, from `EDMSampler.structure_prediction` and `.equilibrium`.
STEP_SCALE = {"structure": 1.5, "equilibrium": 1.0}
# BioEmu2 uses Boltz-1's distogram head, with 64 bins from 2 to 22 Å.
DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
# Upstream's inference default, `num_recycles=3`, plus the first pass.
INFERENCE_RECYCLING_STEPS = 4
# Converts predicted coordinates to a `gemmi.Structure`.
Writer = Callable[[Array], gemmi.Structure]
# `mol_type` value of protein atoms.
PROTEIN = 0
# BioEmu2 token class of each amino acid in `TOKENS` order.
AMINO_ACIDS = np.array([view.TOKEN[aa] for aa in TOKENS])


def feature_dict(features: dict[str, np.ndarray]) -> dict[str, Array]:
    """Convert a feature dict to JAX arrays for this backend.

    `res_type` and `msa` become probabilities over token types so that a soft sequence can
    be set. `atom37_idx` is added with each atom's atom37 index, or -1 for padding and
    non-protein atoms."""
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
    """Featurize `chains` with jfold in `mode` and return the features and a writer."""
    check_chains(chains, "BioEmu2", templates=False, upstream=UPSTREAM)
    system = jfold_system(chains)
    features = view.features(system, mode=mode, paired_queries=True)
    return feature_dict(features), AtomSitesWriter(view.atom_sites(system))


def set_binder_sequence(
    pssm: Float[Array, "L 20"], features: dict[str, Array]
) -> dict[str, Array]:
    """Set `res_type`, `profile` and the MSA query row of the first `L` tokens to `pssm`.

    `profile` is the mean over all MSA rows, so replacing the query row changes it by the
    difference divided by the number of rows."""
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
    """BioEmu2 with upstream's features, recycling counts and sampler presets."""

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
        """Wrap `model`, or load the converted BioEmu2 checkpoint if it is None.

        The sampler is set to upstream's preset for `mode`."""
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
        """Featurize `chains` for `mode` and return the features and a writer. `key` is
        unused."""
        del key
        return featurize(chains, mode=self.mode)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        """Featurize a binder of unknown residues, without an MSA, followed by `chains`."""
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
        """Convert one sample's coordinates to a `gemmi.Structure` with all atoms."""
        return writer(coords)

    def to_structure(
        self,
        features: dict[str, Array],
        writer: Writer,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """Build a `gemmi.Structure`, from atom37 if `sequence` is given and otherwise
        from all atoms."""
        del features
        if sequence is not None:
            return output.to_structure()
        return self.structure(writer, output.structure_coordinates)

    def jfold_features(self, features: dict[str, Array]) -> JFeatures:
        """Convert to jfold features. Only equilibrium sampling and structure prediction
        are supported."""
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
        """Run the trunk once and return the denoiser and atom mask for
        `jfold.sampling.sample`."""
        if recycling_steps is None:
            recycling_steps = self.inference_recycling_steps
        k_prepare, k_trunk = jax.random.split(key)
        feats, inputs = self.prepare(features, PSSM, key=k_prepare)
        trunk = self.run_trunk(
            feats, inputs, recycling_steps=recycling_steps, key=k_trunk
        )
        return self.structure_denoiser(trunk)
