#####################
#
#   Uniform interface for structure prediction models: generating features, building losses, and running structure prediction.

import dataclasses
import itertools
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from jfold.featurize import Template
from jfold.model import StructureModel
from jfold.sampling import (
    Denoise,
    SamplerSettings,
    StructureSampler,
    exponential_update,
    schedules,
)
from jfold.sampling.schedules import Schedule
from jfold.slots import TrunkInputs, TrunkState

from mosaic.common import TOKENS, LinearCombination, LossTerm


class PolymerType:
    PROTEIN = "PROTEIN"
    RNA = "RNA"
    DNA = "DNA"

@dataclass(frozen=True, eq=True, slots=True)
class TargetChain:
    sequence: str
    polymer_type: str = PolymerType.PROTEIN
    use_msa: bool = True
    template_chain: gemmi.Chain | Template | None = None
    # An a3m MSA for this chain with the query first. If given, it is used instead
    # of fetching one.
    msa: str | None = None

    def __post_init__(self):
        if self.msa is not None and not self.use_msa:
            raise ValueError("TargetChain: `msa` given with `use_msa=False`")


# Canonical atom37 ordering (matches AlphaFold residue_constants.atom_types).
ATOM37_NAMES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)
assert len(ATOM37_NAMES) == 37

ATOM37_INDEX: dict[str, int] = {name: i for i, name in enumerate(ATOM37_NAMES)}


def scatter_atom37(
    atom_coords: Float[Array, "N_atom 3"],
    atom_to_token: Int[Array, " N_atom"],
    atom37_idx: Int[Array, " N_atom"],
    n_token: int,
) -> tuple[Float[Array, "N_token 37 3"], Float[Array, "N_token 37"]]:
    """Scatter all-atom coordinates into a `[n_token, 37, 3]` array and its mask.

    Atoms with `atom37_idx == -1`, such as padding, are dropped. They are moved to
    slot 37 first, because `mode='drop'` ignores only indices past the end and a
    raw -1 would wrap to slot 36 (OXT).
    """
    slot = jnp.where(atom37_idx < 0, 37, atom37_idx)
    coords = (
        jnp.zeros((n_token, 37, 3), dtype=jnp.float32)
        .at[atom_to_token, slot]
        .set(atom_coords, mode="drop")
    )
    mask = (
        jnp.zeros((n_token, 37), dtype=jnp.float32)
        .at[atom_to_token, slot]
        .set(jnp.ones(atom_coords.shape[:-1], dtype=jnp.float32), mode="drop")
    )
    return coords, mask


# 64 bins, 0.0–32.0 Å, shared across all model backends
PAE_BINS = np.arange(start=0.25, stop=32.0, step=0.5)
# Chain names in `asym_id` order (AlphaFold's `PDB_CHAIN_IDS`).
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _aatype(sequence: Float[Array, "N 20"]) -> np.ndarray:
    """Return each token's most likely amino acid, or 20 if its row is all zero, as it is
    for unknown residues."""
    sequence = np.asarray(sequence)
    return np.where(sequence.max(-1) > 0, sequence.argmax(-1), 20)


def atom37_structure(
    coords: Float[Array, "N 37 3"],
    mask: Float[Array, "N 37"],
    aatype: Int[Array, " N"],
    chain_index: Int[Array, " N"],
    residue_number: Int[Array, " N"],
    b_factor: Float[Array, " N"] | None = None,
    chain_names: str = CHAIN_IDS,
) -> gemmi.Structure:
    """Build a protein gemmi.Structure from atom37 coordinates.

    `aatype` indexes `common.TOKENS`, and 20 is an unknown residue, written as UNK.
    Residues with the same `chain_index` form one
    chain named `chain_names[chain_index]`, and chains are written in order of first
    appearance. Residues without atoms, such as padding, ligands and nucleic acids, are
    left out. `b_factor` is per residue. If it is None, gemmi's default is kept.
    """
    coords = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(mask)
    aatype = np.asarray(aatype).astype(np.int32)
    chain_index = np.asarray(chain_index).astype(np.int32)
    residue_number = np.asarray(residue_number).astype(np.int32)
    names = [gemmi.expand_one_letter(c, gemmi.ResidueKind.AA) for c in TOKENS] + ["UNK"]

    model = gemmi.Model("1")
    present = (mask >= 0.5).any(-1)
    for c in dict.fromkeys(chain_index[present].tolist()):
        chain = gemmi.Chain(chain_names[c])
        for i in np.flatnonzero(present & (chain_index == c)):
            residue = gemmi.Residue()
            residue.name = names[aatype[i]]
            residue.seqid = gemmi.SeqId(int(residue_number[i]), " ")
            residue.entity_type = gemmi.EntityType.Polymer
            for slot in np.flatnonzero(mask[i] >= 0.5):
                atom = gemmi.Atom()
                atom.name = ATOM37_NAMES[slot]
                atom.element = gemmi.Element(ATOM37_NAMES[slot][0])
                atom.pos = gemmi.Position(*coords[i, slot])
                atom.occ = 1.0
                if b_factor is not None:
                    atom.b_iso = float(b_factor[i])
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure = gemmi.Structure()
    structure.add_model(model)
    structure.setup_entities()
    return structure


def predicted_tm_score(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
    pair_mask: jnp.ndarray | None = None,
    token_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    num_res = logits.shape[0]
    if token_mask is not None:
        valid = token_mask[:, None] & token_mask[None, :]
        pair_mask = valid if pair_mask is None else pair_mask & valid
        logits = jnp.where(valid[..., None], logits, 0)
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = jnp.maximum(num_res if token_mask is None else token_mask.sum(), 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Convert logits to probs.
    probs = jax.nn.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    if pair_mask is None:
        pair_mask = jnp.ones(shape=(num_res, num_res), dtype=bool)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment


class StructureModelOutput(eqx.Module):
    distogram_logits: Float[Array, "N N Bins"]
    distogram_bins: Float[Array, " Bins"]
    plddt: Float[Array, " N"]
    pae: Float[Array, "N N"]
    pae_logits: Float[Array, "N N Bins"]
    pae_bins: Float[Array, " Bins"]
    structure_coordinates: Array  # all-atom coords, shape varies by model
    backbone_coordinates: Float[Array, "N 4 3"]
    # Per-token amino-acid distribution in `common.TOKENS` order. Every model must
    # use this order, since `to_structure` and other losses take its argmax.
    full_sequence: Float[Array, "N 20"]
    asym_id: Float[Array, " N"]
    residue_idx: Int[Array, " N"]
    # Predicted heavy atoms in atom37 layout, ordered as `ATOM37_NAMES`.
    atom37_coords: Float[Array, "N 37 3"]
    atom37_mask: Float[Array, "N 37"]
    token_mask: Bool[Array, " N"] | None = None  # None means every token is present.

    def to_structure(self) -> gemmi.Structure:
        """Build a gemmi.Structure from the atom37 coordinates.

        Residue names come from `full_sequence`, chains from `asym_id`, residue
        numbers from `residue_idx` and B-factors from pLDDT on a 0-100 scale. Only
        standard protein residues are written, because atom37 cannot represent
        ligands or non-canonical residues.
        """
        return atom37_structure(
            coords=self.atom37_coords,
            mask=self.atom37_mask,
            aatype=_aatype(self.full_sequence),
            chain_index=self.asym_id,
            residue_number=self.residue_idx,
            b_factor=np.asarray(self.plddt) * 100.0,
        )

    def iptm(self, binder_length: int) -> Float[Array, ""]:
        """Return the interface pTM between the first `binder_length` tokens and the rest.

        Every other token counts as target, whatever its chain."""
        is_binder = jnp.arange(self.full_sequence.shape[0]) < binder_length
        return predicted_tm_score(
            logits=self.pae_logits,
            bin_centers=self.pae_bins,
            pair_mask=is_binder[:, None] != is_binder[None, :],
            token_mask=self.token_mask,
        ).max()

    def chain_pair_iptm(self) -> dict[tuple[int, int], float]:
        """Return the interface pTM for every pair of chains.

        This is the `iptm` formula restricted to one pair of chains at a time.
        For a two-chain complex the single value equals the overall ipTM. It loops
        over chains in Python, so call it outside `jit`. Keys are `(asym_id,
        asym_id)` pairs with the lower id first.
        """
        asym = np.asarray(self.asym_id).astype(np.int32)
        present = asym if self.token_mask is None else asym[np.asarray(self.token_mask)]
        chains = np.unique(present).tolist()
        out: dict[tuple[int, int], float] = {}
        for a, b in itertools.combinations(chains, 2):
            inter = ((asym == a)[:, None] & (asym == b)[None, :]) | (
                (asym == b)[:, None] & (asym == a)[None, :]
            )
            iptm = predicted_tm_score(
                logits=self.pae_logits,
                bin_centers=self.pae_bins,
                pair_mask=jnp.asarray(inter),
                token_mask=self.token_mask,
            ).max()
            out[(a, b)] = float(iptm)
        return out


class StructurePrediction(eqx.Module):
    st: gemmi.Structure
    plddt: Float[Array, " N"]
    pae: Float[Array, "N N"]
    iptm: float
    model_output: StructureModelOutput


class StructurePredictionModel(eqx.Module):
    @abstractmethod
    def target_only_features(self, chains: list[TargetChain], *, key: PRNGKeyArray) -> PyTree:
        """Generate model features for the target chains only.

        `key` is for random featurization, for example placing reference conformers.
        Models with deterministic featurization ignore it."""

    @abstractmethod
    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> PyTree:
        """Generate model features for a binder of `binder_length` followed by the
        target chains. `key` is as in `target_only_features`."""

    @abstractmethod
    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        recycling_steps: int = 1,
        key,
    ) -> StructurePrediction:
        pass

    @abstractmethod
    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        recycling_steps: int = 1,
        key,
    ) -> StructureModelOutput:
        pass

    @abstractmethod
    def build_loss(
        self, *, loss: LossTerm | LinearCombination, features: PyTree, recycling_steps: int = 1
    ) -> LossTerm:
        pass


# Diffusion steps of the default samplers.
DEFAULT_SAMPLING_STEPS = 8


class TokenInfo(eqx.Module):
    """Per-token sequence in `common.TOKENS` order, chain id and residue index.

    `asym_id` counts chains from 0 and `residue_index` numbers each chain's residues
    from 1, as they are written to structures."""

    sequence: Float[Array, "T 20"]
    asym_id: Int[Array, " T"]
    residue_index: Int[Array, " T"]
    mask: Bool[Array, " T"]  # non-padding tokens


class Atom37(eqx.Module):
    """Coordinates in AlphaFold's atom37 layout."""

    coords: Float[Array, "T 37 3"]
    mask: Float[Array, "T 37"]


class Confidence(eqx.Module):
    plddt: Float[Array, " T"]
    pae_logits: Float[Array, "T T B"]
    pae_bins: Float[Array, " B"]

    @classmethod
    def unavailable(cls, num_tokens: int) -> "Confidence":
        """Return NaN pLDDT and PAE for models without a confidence head."""
        return cls(
            plddt=jnp.full((num_tokens,), jnp.nan),
            pae_logits=jnp.full((num_tokens, num_tokens, len(PAE_BINS)), jnp.nan),
            pae_bins=jnp.array(PAE_BINS, dtype=jnp.float32),
        )


class Distogram(eqx.Module):
    logits: Float[Array, "T T B"]
    bins: Float[Array, " B"]


class Inputs(eqx.Module):
    """The output of `embed`: jfold's features and the trunk inputs."""

    features: eqx.Module
    trunk: TrunkInputs


class Trunk[Features](eqx.Module):
    """The features, embedded inputs and last trunk state that the heads read."""

    features: Features
    inputs: Inputs
    state: TrunkState
    # Manual constructions can supply just the raw recycling state.
    finished_state: TrunkState | None = None


def _float32(denoise, x, sigma):
    with jax.default_matmul_precision("float32"):
        return denoise(x, sigma)


def _step(model, features, inputs, state, key):
    return model.trunk(features, inputs, state, key=key)


class DiffusionStructurePredictionModel[Features](StructurePredictionModel):
    """Base class for trunk and diffusion models that run on a jfold `StructureModel`.

    `model` runs the forward pass. Each backend implements featurization, setting the
    binder sequence, and token and atom37 information. A backend can also override
    `token_plddt` if its pLDDT head needs it."""

    model: AbstractVar[StructureModel]
    distogram_bins: ClassVar[np.ndarray]
    # Recompute each trunk pass during the backward pass, to save memory on large trunks.
    remat_steps: ClassVar[bool] = False
    # Default step count for `upstream_sampler`. If None, use `model.sampling.num_steps`.
    upstream_sampling_steps: ClassVar[int | None] = None
    # Number of trunk passes, counting the first, for design and for inference.
    design_recycling_steps: AbstractVar[int]
    inference_recycling_steps: AbstractVar[int]
    # If False, backpropagate through the last trunk pass only.
    backprop_all_passes: AbstractVar[bool]

    def prepare(
        self, features: Features, sequence: Float[Array, "L 20"] | None, *, key: PRNGKeyArray
    ) -> tuple[Features, Inputs]:
        """Set the binder sequence if one is given, then embed the features."""
        k_sequence, k_embed = jax.random.split(key)
        if sequence is not None:
            features = self.set_sequence(features, sequence, key=k_sequence)
        return features, self.embed(features, key=k_embed)

    def run_trunk(
        self,
        features: Features,
        inputs: Inputs,
        *,
        recycling_steps: int,
        key: PRNGKeyArray,
        initial_state: TrunkState | None = None,
    ) -> Trunk[Features]:
        """Run `recycling_steps` trunk passes.

        The first pass starts from `initial_state`, or from `init_state` if it is None.
        Gradients flow through every pass if `backprop_all_passes` is set, and
        otherwise through the last pass only."""
        if recycling_steps < 1:
            raise ValueError(f"recycling_steps must be at least 1, got {recycling_steps}")
        k_init, k_passes = jax.random.split(key)
        keys = jax.random.split(k_passes, recycling_steps)

        def one_pass(state, k):
            return self.step(inputs, state, key=k), None

        state = initial_state
        if state is None:
            state = self.init_state(inputs, key=k_init)
        if not self.backprop_all_passes:
            state = jax.lax.stop_gradient(jax.lax.scan(one_pass, state, keys[:-1])[0])
            keys = keys[-1:]
        state = jax.lax.scan(one_pass, state, keys)[0]
        return Trunk(features, inputs, state, self.model.trunk.finish(inputs.features, state))

    def sample_structure(
        self, trunk: Trunk[Features], *, sampler: SamplerSettings, key: PRNGKeyArray
    ) -> StructureModelOutput:
        """Sample one structure from the trunk and evaluate the heads."""
        coords = self.structure_sampler(trunk, sampler).sample(key)
        tokens = self.tokens(trunk.features)
        distogram = self.distogram(trunk)
        confidence = self.confidence(trunk, coords)
        atoms = self.atom37(trunk.features, coords)
        pae = jnp.sum(jax.nn.softmax(confidence.pae_logits, axis=-1) * confidence.pae_bins, axis=-1)
        return StructureModelOutput(
            distogram_logits=distogram.logits,
            distogram_bins=distogram.bins,
            plddt=confidence.plddt,
            pae=pae,
            pae_logits=confidence.pae_logits,
            pae_bins=confidence.pae_bins,
            structure_coordinates=coords,
            backbone_coordinates=atoms.coords[:, _BACKBONE_ATOM37],
            full_sequence=tokens.sequence,
            asym_id=tokens.asym_id,
            residue_idx=tokens.residue_index,
            atom37_coords=atoms.coords,
            atom37_mask=atoms.mask,
            token_mask=tokens.mask,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: Float[Array, "L 20"] | None = None,
        features: Features,
        recycling_steps: int | None = None,
        sampler: SamplerSettings | None = None,
        initial_state: TrunkState | None = None,
        key: PRNGKeyArray,
    ) -> StructureModelOutput:
        """Predict one structure, by default with `design_recycling_steps` trunk
        passes and `sampler()`."""
        if recycling_steps is None:
            recycling_steps = self.design_recycling_steps
        sampler = self.sampler() if sampler is None else sampler
        k_prepare, k_trunk, k_sample = jax.random.split(key, 3)
        features, inputs = self.prepare(features, PSSM, key=k_prepare)
        trunk = self.run_trunk(
            features, inputs, recycling_steps=recycling_steps, key=k_trunk, initial_state=initial_state
        )
        return self.sample_structure(trunk, sampler=sampler, key=k_sample)

    def predict(
        self,
        *,
        PSSM: Float[Array, "L 20"] | None = None,
        features: Features,
        recycling_steps: int | None = None,
        sampler: SamplerSettings | None = None,
        initial_state: TrunkState | None = None,
        key: PRNGKeyArray,
    ) -> StructurePrediction:
        """Predict one structure, by default with `inference_recycling_steps` trunk
        passes and `sampler()`.

        Pass `sampler=upstream_sampler()` to use the model's original sampler."""
        if recycling_steps is None:
            recycling_steps = self.inference_recycling_steps
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampler=sampler,
            initial_state=initial_state,
            key=key,
        )
        iptm = output.iptm(0 if PSSM is None else PSSM.shape[0])
        return StructurePrediction(
            st=output.to_structure(),
            plddt=output.plddt,
            pae=output.pae,
            iptm=iptm,
            model_output=output,
        )

    def build_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: Features,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        reduction: Callable = jnp.mean,
        sampler: SamplerSettings | None = None,
        initial_state: TrunkState | None = None,
    ) -> "MultiSampleLoss[Features]":
        """Build a design loss that runs the trunk once and reduces `loss` over
        `num_samples` diffusion samples."""
        return MultiSampleLoss(
            model=self,
            features=features,
            loss=loss,
            sampler=self.sampler() if sampler is None else sampler,
            recycling_steps=self.design_recycling_steps if recycling_steps is None else recycling_steps,
            num_samples=num_samples,
            reduction=reduction,
            initial_state=initial_state,
        )

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> SamplerSettings:
        """Return the model's original sampler, by default with its original step count."""
        if num_steps is None:
            num_steps = self.upstream_sampling_steps
        return self.model.sampling.settings(num_steps, schedule=schedule)

    def sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> SamplerSettings:
        """Return the default sampler.

        This is the model's own sampler with the exponential update, a geometric
        schedule and `DEFAULT_SAMPLING_STEPS` steps. Models such as ESMFold2 whose
        sampler limits the schedule may take fewer steps."""
        sampler = self.upstream_sampler(
            DEFAULT_SAMPLING_STEPS if num_steps is None else num_steps,
            schedule=schedules.Geometric() if schedule is None else schedule,
        )
        return dataclasses.replace(sampler, update=exponential_update)

    @abstractmethod
    def target_only_features(
        self, chains: Sequence[TargetChain], *, key: PRNGKeyArray
    ) -> Features:
        """Featurize the chains as given."""

    @abstractmethod
    def binder_features(
        self, binder_length: int, chains: Sequence[TargetChain], *, key: PRNGKeyArray
    ) -> Features:
        """Featurize a placeholder binder as chain 0, followed by `chains`."""

    def structure(self, features: Features, coords: Float[Array, "A 3"]) -> gemmi.Structure:
        """Build a protein structure from all-atom coordinates, for example one
        sampling step's. Residue names come from `features` as they are, so set a
        design's binder with `set_sequence` first."""
        tokens = self.tokens(features)
        atoms = self.atom37(features, coords)
        return atom37_structure(
            coords=atoms.coords,
            mask=atoms.mask,
            aatype=_aatype(tokens.sequence),
            chain_index=tokens.asym_id,
            residue_number=tokens.residue_index,
        )

    @abstractmethod
    def set_sequence(
        self, features: Features, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray
    ) -> Features:
        """Set the first `L` tokens to the soft `sequence`. This runs inside `jit`."""

    @abstractmethod
    def jfold_features(self, features: Features) -> eqx.Module:
        """Build jfold's features from the backend's features. This runs inside `jit`."""

    @abstractmethod
    def tokens(self, features: Features) -> TokenInfo: ...

    @abstractmethod
    def atom37(self, features: Features, coords: Float[Array, "A 3"]) -> Atom37: ...

    def embed(self, features: Features, *, key: PRNGKeyArray) -> Inputs:
        """Compute the inputs shared by every trunk pass and the heads."""
        jfeatures = self.jfold_features(features)
        return Inputs(features=jfeatures, trunk=self.model.embed(jfeatures, key=key))

    def init_state(self, inputs: Inputs, *, key: PRNGKeyArray) -> TrunkState:
        """Return the trunk state before the first pass."""
        return self.model.trunk.init_state(inputs.trunk, key=key)

    def step(self, inputs: Inputs, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        """Run one trunk pass."""
        step = eqx.filter_checkpoint(_step) if self.remat_steps else _step
        return step(self.model, inputs.features, inputs.trunk, state, key)

    def head_state(self, trunk: Trunk[Features]) -> TrunkState:
        """Return the last trunk state after `model.trunk.finish`, which the heads read."""
        if trunk.finished_state is None:
            return self.model.trunk.finish(trunk.inputs.features, trunk.state)
        return trunk.finished_state

    def structure_sampler(self, trunk: Trunk[Features], sampler: SamplerSettings) -> StructureSampler:
        """Build a structure sampler conditioned on `trunk` with `sampler`'s settings.

        The structure module denoises at float32 matmul precision."""
        jfold_sampler = self.model.structure_sampler(
            trunk.inputs.features, trunk.inputs.trunk, self.head_state(trunk)
        )
        return StructureSampler(
            denoise=eqx.Partial(_float32, jfold_sampler.denoise),
            atom_mask=jfold_sampler.atom_mask,
            settings=sampler,
        )

    def structure_denoiser(self, trunk: Trunk[Features]) -> tuple[Denoise, Float[Array, " A"]]:
        """Return the structure module conditioned on the trunk, and the atom mask."""
        structure_sampler = self.structure_sampler(trunk, self.upstream_sampler())
        return structure_sampler.denoise, structure_sampler.atom_mask

    def distogram(self, trunk: Trunk[Features]) -> Distogram:
        return Distogram(
            logits=self.model.distogram(self.head_state(trunk).z),
            bins=jnp.array(self.distogram_bins),
        )

    def confidence(self, trunk: Trunk[Features], coords: Float[Array, "A 3"]) -> Confidence:
        """Return per-token pLDDT and PAE logits.

        Models without a confidence head return `Confidence.unavailable`."""
        if self.model.confidence is None:
            return Confidence.unavailable(self.tokens(trunk.features).mask.shape[0])
        logits = self.model.confidence_logits(
            trunk.inputs.features, trunk.inputs.trunk, self.head_state(trunk), coords
        )
        return Confidence(
            plddt=self.token_plddt(self.plddt(logits.plddt), trunk),
            pae_logits=logits.pae,
            pae_bins=jnp.array(PAE_BINS),
        )

    def plddt(self, logits: Float[Array, "... B"]) -> Float[Array, "..."]:
        """Return pLDDT in [0, 1] as the expected value over evenly spaced bins."""
        num_bins = logits.shape[-1]
        return jax.nn.softmax(logits, axis=-1) @ ((jnp.arange(num_bins) + 0.5) / num_bins)

    def token_plddt(self, plddt: Float[Array, " N"], trunk: Trunk[Features]) -> Float[Array, " T"]:
        """Return per-token pLDDT.

        For a per-atom head, each token takes the value of its representative atom."""
        features = trunk.inputs.features
        if plddt.shape[0] == features.token_mask.shape[0]:
            return plddt
        return plddt[features.rep_atom]


_BACKBONE_ATOM37 = (0, 1, 2, 4)  # N, CA, C, O


def reduce_samples(values, auxiliary, reduction, num_samples):
    """Reduce per-sample loss ``values`` and sort scalar aux metrics by loss.

    Aux leaves with shape ``(num_samples,)`` are returned as lists in ascending-loss
    order, even when ``num_samples`` is 1. Other leaves, such as predicted
    structures, are returned unchanged.
    """
    sortperm = jnp.argsort(values)

    def _sort_if_scalar(value):
        if isinstance(value, jax.Array) and value.shape == (num_samples,):
            return list(value[sortperm])
        return value

    return reduction(values), jax.tree.map(_sort_if_scalar, auxiliary)


class MultiSampleLoss[Features](LossTerm):
    """Run the trunk once and reduce `loss` over `num_samples` diffusion samples."""

    model: DiffusionStructurePredictionModel[Features]
    features: Features
    loss: LossTerm | LinearCombination
    sampler: SamplerSettings
    recycling_steps: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True, default=1)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)
    # Trunk state before the first pass. If None, the model's `init_state` is used.
    initial_state: TrunkState | None = None

    def __call__(self, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray):
        k_prepare, k_trunk, k_samples = jax.random.split(key, 3)
        features, inputs = self.model.prepare(self.features, sequence, key=k_prepare)
        trunk = self.model.run_trunk(
            features,
            inputs,
            recycling_steps=self.recycling_steps,
            key=k_trunk,
            initial_state=self.initial_state,
        )

        def one_sample(k):
            k_sample, k_loss = jax.random.split(k)
            output = self.model.sample_structure(trunk, sampler=self.sampler, key=k_sample)
            return self.loss(sequence, output, key=k_loss)

        values, aux = jax.vmap(one_sample)(jax.random.split(k_samples, self.num_samples))
        return reduce_samples(values, aux, self.reduction, self.num_samples)
