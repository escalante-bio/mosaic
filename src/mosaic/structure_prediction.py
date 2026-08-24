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
    def supports_template_chains(self) -> bool:
        """Whether :class:`TargetChain.template_chain` may be supplied.

        Most backends can consume a target template. Models that cannot must
        override this so binder design can still use them for de novo
        hallucination and sequence-only folding without passing unsupported
        input.
        """
        return True

    def prediction_features_depend_on_sequence(self) -> bool:
        """Whether finished-sequence scoring must be featurized per sequence.

        AF2, Boltz2 and Protenix can build a reusable placeholder-binder feature
        pack and splice a PSSM into it. ESMFold2 native validation must instead
        featurize the real sequence, including its atom packing.
        """
        return False

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
        sampling_steps: int | None = None,
        key,
    ) -> StructurePrediction:
        pass
       
    @abstractmethod
    def model_output(self, *, PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        recycling_steps: int = 1,
        sampling_steps: int | None = None,
        key) -> StructureModelOutput:
        pass


    @abstractmethod
    def build_loss(self, *, loss: LossTerm | LinearCombination, features: PyTree,  recycling_steps: int = 1, sampling_steps: int | None = None,) -> LossTerm:
        pass

    def ensemble_members(self) -> tuple[int, ...]:
        """Identifiers for members that can be scored independently.

        AF2 overrides this for separately trained networks. Diffusion backends
        can request several members without an override because callers split
        the RNG key for each member.
        """
        return (0,)

    def member_kwargs(self, member: int) -> dict:
        """Extra :meth:`predict` arguments selecting one ensemble member."""
        return {}

    def design_member_kwargs(self, member: int, *, use_dropout: bool) -> dict:
        """Arguments selecting one member during differentiable design."""
        del use_dropout
        return self.member_kwargs(member)

