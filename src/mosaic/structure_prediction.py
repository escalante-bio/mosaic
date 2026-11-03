#####################
#
#   Uniform interface for structure prediction models: generating features, building losses, and running structure prediction.

from abc import abstractmethod
from collections.abc import Callable, Sequence
import dataclasses
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import gemmi
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.sampling import Denoise, Sampler, exponential_update, schedules
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


class Distogram(eqx.Module):
    logits: Float[Array, "T T B"]
    bins: Float[Array, " B"]


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

    # -- public API ---------------------------------------------------------

    def build_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: Features,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        reduction: Callable = jnp.mean,
        sampler: Sampler | None = None,
    ) -> LossTerm:
        """A design loss: run the trunk once and reduce `loss` over `num_samples`
        diffusion samples."""
        from mosaic import structure_model

        return structure_model.build_loss(
            self,
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            reduction=reduction,
            sampler=sampler,
        )

    def model_output(
        self,
        *,
        PSSM: Float[Array, "L 20"] | None = None,
        features: Features,
        recycling_steps: int | None = None,
        sampler: Sampler | None = None,
        key: PRNGKeyArray,
    ) -> StructureModelOutput:
        """Predict one structure, with design settings by default."""
        from mosaic import structure_model

        return structure_model.model_output(
            self, features, sequence=PSSM, recycling_steps=recycling_steps, sampler=sampler, key=key
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "L 20"] | None = None,
        features: Features,
        writer: Writer = None,
        recycling_steps: int | None = None,
        sampler: Sampler | None = None,
        key: PRNGKeyArray,
    ) -> StructurePrediction:
        """Predict one structure, with inference settings by default."""
        from mosaic import structure_model

        return structure_model.predict(
            self,
            features,
            writer,
            sequence=PSSM,
            recycling_steps=recycling_steps,
            sampler=sampler,
            key=key,
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
        geometric schedule, `DEFAULT_SAMPLING_STEPS` steps."""
        sampler = self.upstream_sampler(
            num_steps or DEFAULT_SAMPLING_STEPS, schedule=schedule or schedules.Geometric()
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
        self, features: Features, inputs: Inputs, state: TrunkState
    ) -> tuple[Denoise, Float[Array, " A"]]:
        """The structure module conditioned on `state`, and the atom mask."""

    @abstractmethod
    def distogram(self, features: Features, inputs: Inputs, state: TrunkState) -> Distogram: ...

    @abstractmethod
    def confidence(
        self,
        features: Features,
        inputs: Inputs,
        state: TrunkState,
        coords: Float[Array, "A 3"],
    ) -> Confidence | None:
        """`None` for models without a confidence head."""

    @abstractmethod
    def tokens(self, features: Features) -> TokenInfo: ...

    @abstractmethod
    def atom37(self, features: Features, coords: Float[Array, "A 3"]) -> Atom37: ...
