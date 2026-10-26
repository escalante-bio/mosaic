#####################
#
#   Uniform interface for structure prediction models: generating features, building losses, and running structure prediction.

# TODO: remove lots of redundant code.
# It would be nice if all models `predict` method went through `model_output` ( right now this is only the case for Protenix and AF2).

import gemmi
from dataclasses import dataclass
import equinox as eqx
from jaxtyping import Array, Float, PyTree

from abc import abstractmethod

from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.common import LossTerm, LinearCombination
from mosaic.sampling import Sampler

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


class DiffusionStructurePredictionModel(StructurePredictionModel):
    """A `StructurePredictionModel` whose structure module is an AF3-style diffusion denoiser.

    `predict` defaults to the upstream model's own inference settings: its sampler
    and `recycling_steps`. `model_output` and the loss builders default to cheaper
    settings suited to design.

    Every method takes `sampler: Sampler | None` to replace the default sampler,
    e.g. `model.inference_sampler(50)` for fewer steps, or
    `dataclasses.replace(model.inference_sampler(), gamma_0=0.0)`.
    """

    @abstractmethod
    def design_sampler(self, num_steps: int | None = None) -> Sampler:
        """The sampler `model_output` and the loss builders use by default."""

    @abstractmethod
    def inference_sampler(self, num_steps: int | None = None) -> Sampler:
        """The upstream model's own sampler, which `predict` uses by default."""


