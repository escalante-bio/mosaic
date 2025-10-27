# TODO: figure out how to NOT produce MSA for a target chain
from mosaic.structure_prediction import (
    StructurePredictionModel,
    TargetChain,
    PolymerType,
    StructurePrediction,
)

from mosaic.losses.structure_prediction import IPTMLoss

import numpy as np
import jax
import jax.numpy as jnp


from mosaic.losses.protenix import (
    load_protenix_mini,
    load_protenix_tiny,
    load_features_from_json,
    ProtenixLoss,
    set_binder_sequence,
    ProtenixOutput,
    biotite_array_to_gemmi_struct,
)


from jaxtyping import Array, Float, PyTree
import equinox as eqx


class Protenix(StructurePredictionModel):
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
                for c in chains
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
        return_state=True,
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
            return_state=return_state,
        )

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
        features = set_binder_sequence(PSSM, features) if PSSM is not None else features
        o = ProtenixOutput(
            model=self.protenix,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            n_structures=1,
            initial_recycling_state=initial_recycling_state,
            key=key,
        )
        return o

    @eqx.filter_jit
    def _coords_and_confidences(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
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
        if PSSM is None:
            PSSM = jnp.zeros((0, 20))
        iptm = -IPTMLoss()(PSSM, output, key=jax.random.key(0))[0]
        return (output.structure_coordinates[0], output.pae, output.plddt, iptm)

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
        (coords, pae, plddt, iptm) = self._coords_and_confidences(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            initial_recycling_state=initial_recycling_state,
            key=key,
        )
        return StructurePrediction(
            st=biotite_array_to_gemmi_struct(writer, np.array(coords)),
            plddt=plddt,
            pae=pae,
            iptm=iptm,
        )


def ProtenixMini():
    return Protenix(load_protenix_mini())


def ProtenixTiny():
    return Protenix(load_protenix_tiny())
