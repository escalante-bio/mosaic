# TODO: figure out how to NOT produce MSA for a target chain
from mosaic.structure_prediction import (
    StructurePredictionModel,
    TargetChain,
    PolymerType,
)

import numpy as np


from mosaic.losses.protenix import (
    load_protenix_mini,
    load_protenix_tiny,
    load_features_from_json,
    ProtenixLoss,
    set_binder_sequence,
    ProtenixOutput,
    biotite_array_to_gemmi_struct,
)

from mosaic.losses.structure_prediction import ConcreteStructureOutput

from jaxtyping import Array, Float, PyTree
import equinox as eqx



class Protenix(eqx.Module, StructurePredictionModel):
    protenix: eqx.Module

    def __init__(self, protenix_model: eqx.Module):
        self.protenix = protenix_model

    def target_only_features(self, chains: list[TargetChain]):
        for c in chains:
            assert c.use_msa, "Protenix interface must use MSA for all chains"

        def _polymer_type_to_str(pt: str) -> str:
            match pt:
                case PolymerType.PROTEIN:
                    return "proteinChain"
                case PolymerType.RNA:
                    return "rnaSequence"
                case PolymerType.DNA:
                    return "dnaSequence"

        json = {
            "sequences": [
                {
                    _polymer_type_to_str(c.polymer_type): {
                        "sequence": c.sequence,
                        "count": 1,
                    }
                }
            ],
            "name": "protenix",
        }
        return load_features_from_json(json)

    def binder_features(self, binder_length, chains: list[TargetChain]):
        binder = TargetChain(sequence="X" * binder_length, use_msa=True)
        return self.target_only_features([binder] + chains)

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        sampling_steps=2,
        initial_recycling_state=None,
        return_coords=True,
    ):
        return ProtenixLoss(
            self.protenix,
            features,
            loss,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            n_structures=1,
            initial_recycling_state=initial_recycling_state,
            return_coords=return_coords,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=2,
        initial_recycling_state=None,
        key,
    ):
        features = set_binder_sequence(features, PSSM) if PSSM is not None else features
        o = ProtenixOutput(
            model=self.protenix,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            n_structures=1,
            initial_recycling_state=initial_recycling_state,
            key=key,
        )
        return ConcreteStructureOutput.from_source(o)

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer,
        recycling_steps=1,
        sampling_steps=2,
        initial_recycling_state=None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            initial_recycling_state=initial_recycling_state,
            key=key,
        )

        return biotite_array_to_gemmi_struct(
            writer, np.array(output.structure_coordinates[0])
        )


def ProtenixMini():
    return Protenix(load_protenix_mini())


def ProtenixTiny():
    return Protenix(load_protenix_tiny())
