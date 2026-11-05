#####################
#
#   Uniform interface for structure prediction models: generating features, building losses, and running structure prediction.

import dataclasses
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.structure_prediction import (
    PAE_BINS,
    IPTMLoss,
    StructureModelOutput,
    reduce_samples,
)
from mosaic.sampling import Denoise, Sampler, exponential_update, sample, schedules
from mosaic.sampling.schedules import Schedule


class PolymerType:
    PROTEIN = "PROTEIN"
    RNA = "RNA"
    DNA = "DNA"

@dataclass(frozen=True, eq=True, slots=True)
class TargetChain:
    sequence: str
    polymer_type: str = PolymerType.PROTEIN
    use_msa: bool = True
    template_chain: gemmi.Chain | None = None


class StructurePrediction(eqx.Module):
    st: gemmi.Structure
    plddt: Float[Array, " N"]
    pae: Float[Array, "N N"]
    iptm: float
    model_output: StructureModelOutput


class StructurePredictionModel(eqx.Module):
    @abstractmethod
    def target_only_features(self, chains: list[TargetChain]) -> tuple[PyTree, any]:
        """
        Generate model features and postprocessor for the target chains only.

        Args:
            chains: List of TargetChain objects representing the target chains.
            
        Returns:
            tuple of (PyTree, StructureWriter) containing the generated features and an object for turning a prediction into a gemmi.Structure.

        """
        pass

    @abstractmethod
    def binder_features(self, binder_length: int, chains: list[TargetChain]) -> tuple[PyTree, any]:
        """
        Generate model features and postprocessor for a binder of given length and the target chains.

        Args:
            binder_length: Length of the binder chain.
            chains: List of TargetChain objects representing the target chains.

        Returns:
            tuple of (PyTree, StructureWriter) containing the generated features and an object for turning a prediction into a gemmi.Structure.

        """
        pass


    @abstractmethod
    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        writer: any,
        recycling_steps: int = 1,
        key,
    ) -> StructurePrediction:
        pass
       
    @abstractmethod
    def model_output(self, *, PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        recycling_steps: int = 1,
        key) -> StructureModelOutput:
        pass


    @abstractmethod
    def build_loss(self, *, loss: LossTerm | LinearCombination, features: PyTree,  recycling_steps: int = 1) -> LossTerm:
        pass


Writer = Any  # turns coordinates into a structure (model-specific)

# Diffusion steps of the default samplers.
DEFAULT_SAMPLING_STEPS = 8


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


class TrunkState(eqx.Module):
    """State carried between trunk passes. `single` is `None` for pair-only trunks."""

    pair: Float[Array, "... T T Cz"]
    single: Float[Array, "... T Cs"] | None = None


class TokenInfo(eqx.Module):
    """Per-token identity; `sequence` is in `common.TOKENS` order."""

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
        """NaN pLDDT and PAE, for models without a confidence head."""
        return cls(
            plddt=jnp.full((num_tokens,), jnp.nan),
            pae_logits=jnp.full((num_tokens, num_tokens, len(PAE_BINS)), jnp.nan),
            pae_bins=jnp.array(PAE_BINS, dtype=jnp.float32),
        )


class Distogram(eqx.Module):
    logits: Float[Array, "T T B"]
    bins: Float[Array, " B"]


class Trunk[Features, Inputs](eqx.Module):
    """What the heads read: the features, the embedded inputs and the final trunk state."""

    features: Features
    inputs: Inputs
    state: TrunkState


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


class DiffusionStructurePredictionModel[Features, Inputs](StructurePredictionModel):
    """A trunk + diffusion structure model. `Features` is its feature type and
    `Inputs` what `embed` returns."""

    # Number of trunk passes (counting the first) for design and for inference.
    design_recycling_steps: AbstractVar[int]
    inference_recycling_steps: AbstractVar[int]
    # Backpropagate through every trunk pass, or only the last.
    backprop_all_passes: AbstractVar[bool]

    # -- shared logic -------------------------------------------------------

    def prepare(
        self, features: Features, sequence: Float[Array, "L 20"] | None, *, key: PRNGKeyArray
    ) -> tuple[Features, Inputs]:
        """Set the binder sequence (if given) and embed."""
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
    ) -> Trunk[Features, Inputs]:
        """Run `recycling_steps` trunk passes from `initial_state` (default
        `init_state`; e.g. the `state` of an earlier `run_trunk` on the same
        tokens). Gradients flow through every pass if `backprop_all_passes`,
        otherwise through the last only."""
        if recycling_steps < 1:
            raise ValueError(f"recycling_steps must be at least 1, got {recycling_steps}")
        k_init, k_passes = jax.random.split(key)
        keys = jax.random.split(k_passes, recycling_steps)

        def one_pass(state, k):
            return self.step(features, inputs, state, key=k), None

        state = initial_state
        if state is None:
            state = self.init_state(features, inputs, key=k_init)
        if not self.backprop_all_passes:
            state = jax.lax.stop_gradient(jax.lax.scan(one_pass, state, keys[:-1])[0])
            keys = keys[-1:]
        state = jax.lax.scan(one_pass, state, keys)[0]
        return Trunk(features=features, inputs=inputs, state=state)

    def sample_structure(
        self, trunk: Trunk[Features, Inputs], *, sampler: Sampler, key: PRNGKeyArray
    ) -> StructureModelOutput:
        """Sample one structure from the trunk and evaluate the heads."""
        denoise, atom_mask = self.structure_denoiser(trunk)
        coords = sample(denoise, atom_mask, sampler, key=key)
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
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: Float[Array, "L 20"] | None = None,
        features: Features,
        recycling_steps: int | None = None,
        sampler: Sampler | None = None,
        initial_state: TrunkState | None = None,
        key: PRNGKeyArray,
    ) -> StructureModelOutput:
        """Predict one structure: `design_recycling_steps` trunk passes and the
        default `sampler()` unless given."""
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
        writer: Writer = None,
        recycling_steps: int | None = None,
        sampler: Sampler | None = None,
        initial_state: TrunkState | None = None,
        key: PRNGKeyArray,
    ) -> StructurePrediction:
        """Predict one structure: `inference_recycling_steps` trunk passes and the
        default `sampler()` unless given (pass `upstream_sampler()` for the
        model's own sampler)."""
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
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
        return StructurePrediction(
            st=self.to_structure(features, writer, output, PSSM),
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
        sampler: Sampler | None = None,
        initial_state: TrunkState | None = None,
    ) -> "MultiSampleLoss[Features, Inputs]":
        """A design loss: run the trunk once (from `initial_state`, if given) and
        reduce `loss` over `num_samples` diffusion samples."""
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

    @abstractmethod
    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        """The model's own sampler as upstream defines it (upstream's step count
        by default)."""

    def sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        """The default sampler: the model's own, with the exponential update on a
        geometric schedule, `DEFAULT_SAMPLING_STEPS` steps (fewer for models whose
        sampler caps the schedule, e.g. ESMFold2)."""
        sampler = self.upstream_sampler(
            DEFAULT_SAMPLING_STEPS if num_steps is None else num_steps,
            schedule=schedules.Geometric() if schedule is None else schedule,
        )
        return dataclasses.replace(sampler, update=exponential_update)

    # -- host side --------------------------------------------------------

    @abstractmethod
    def target_only_features(self, chains: Sequence[TargetChain]) -> tuple[Features, Writer]:
        """Featurize the chains as given."""

    @abstractmethod
    def binder_features(
        self, binder_length: int, chains: Sequence[TargetChain]
    ) -> tuple[Features, Writer]:
        """Features for a placeholder binder (chain 0) followed by `chains`."""

    @abstractmethod
    def to_structure(
        self,
        features: Features,
        writer: Writer,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """Convert `output` to a structure."""

    # -- traced -----------------------------------------------------------

    @abstractmethod
    def set_sequence(
        self, features: Features, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray
    ) -> Features:
        """Set the first `L` tokens to the (soft) `sequence`."""

    @abstractmethod
    def embed(self, features: Features, *, key: PRNGKeyArray) -> Inputs:
        """Inputs shared by every trunk pass and the heads."""

    @abstractmethod
    def init_state(self, features: Features, inputs: Inputs, *, key: PRNGKeyArray) -> TrunkState:
        """The state before the first pass."""

    @abstractmethod
    def step(
        self, features: Features, inputs: Inputs, state: TrunkState, *, key: PRNGKeyArray
    ) -> TrunkState:
        """One trunk pass."""

    @abstractmethod
    def structure_denoiser(
        self, trunk: Trunk[Features, Inputs]
    ) -> tuple[Denoise, Float[Array, " A"]]:
        """The structure module conditioned on the trunk, and the atom mask."""

    @abstractmethod
    def distogram(self, trunk: Trunk[Features, Inputs]) -> Distogram: ...

    @abstractmethod
    def confidence(
        self, trunk: Trunk[Features, Inputs], coords: Float[Array, "A 3"]
    ) -> Confidence:
        """`Confidence.unavailable` for models without a confidence head."""

    @abstractmethod
    def tokens(self, features: Features) -> TokenInfo: ...

    @abstractmethod
    def atom37(self, features: Features, coords: Float[Array, "A 3"]) -> Atom37: ...


_BACKBONE_ATOM37 = (0, 1, 2, 4)  # N, CA, C, O


class MultiSampleLoss[Features, Inputs](LossTerm):
    """Run the trunk once and reduce `loss` over `num_samples` diffusion samples."""

    model: DiffusionStructurePredictionModel[Features, Inputs]
    features: Features
    loss: LossTerm | LinearCombination
    sampler: Sampler
    recycling_steps: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True, default=1)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)
    # Trunk state before the first pass; `None`: the model's `init_state`.
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
