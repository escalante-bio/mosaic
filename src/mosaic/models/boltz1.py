from mosaic.structure_prediction import StructurePredictionModel, TargetChain

from mosaic.losses.boltz import (
    load_boltz as lb,
    load_features_and_structure_writer,
    set_binder_sequence,
    Boltz1Loss,
    Boltz1Output,
)

from pathlib import Path
from jaxtyping import Array, Float, PyTree
import equinox as eqx
import gemmi


class Boltz1(eqx.Module, StructurePredictionModel):
    model: eqx.Module

    def __init__(self, cache_path: Path | None = None):
        self.model = lb(cache_path) if cache_path is not None else lb()

    @staticmethod
    def _prefix():
        return """version: 1
sequences:"""

    @staticmethod
    def chain_yaml(chain_name: str, chain: TargetChain) -> str:
        assert chain.template_chain is None, "Boltz1 does not support templates."
        raw = f"""  - {chain.polymer_type.lower()}:
        id: [{chain_name}]
        sequence: {chain.sequence}"""
        if not chain.use_msa:
            raw += """
        msa: empty"""

        return raw

    def target_only_features(self, chains: list[TargetChain]):
        yaml = "\n".join(
            [self._prefix()]
            + [
                self.chain_yaml(chain_id, c)
                for chain_id, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
            ]
        )
        features, writer = load_features_and_structure_writer(yaml)
        return (features, writer)

    def binder_features(self, binder_length, chains: list[TargetChain]):
        binder_yaml = f"""  - protein:
      id: [A]
      sequence: {"X" * binder_length}
      msa: empty"""
        yaml = "\n".join(
            [
                self._prefix(),
                binder_yaml,
            ]
            + [
                self.chain_yaml(chain_id, c)
                for chain_id, c in zip("BCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
            ]
        )
        features, writer = load_features_and_structure_writer(yaml)
        return (features, writer)

    def build_loss(self, *, loss, features, recycling_steps=1, sampling_steps=None):
        return Boltz1Loss(
            joltz1=self.model,
            features=features,
            recycling_steps=recycling_steps
            - 1,  # Really awkward off-by-one issue in Joltz1 :/
            sampling_steps=sampling_steps if sampling_steps is not None else 25,
            loss=loss,
            deterministic=True,
        )

    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        if PSSM is not None:
            features = set_binder_sequence(features, PSSM)

        return Boltz1Output(
            joltz=self.model,
            features=features,
            recycling_steps=recycling_steps
            - 1,  # Really awkward off-by-one issue in Joltz1 :/
            num_sampling_steps=sampling_steps if sampling_steps is not None else 25,
            key=key,
            deterministic=True,
        )

    @eqx.filter_jit
    def _pred(self, features, key, recycling_steps, sampling_steps):
        print("JIT compiling Boltz1...")
        return self.model(
            features,
            key=key,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            deterministic=True,
            sample_structure=True,
            confidence_prediction=True,
        )

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer: any,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        if PSSM is not None:
            features = set_binder_sequence(PSSM, features)

        dict_out = self._pred(
            features,
            key=key,
            recycling_steps=recycling_steps - 1,
            sampling_steps=sampling_steps if sampling_steps is not None else 25,
        )

        return gemmi.read_structure(str(writer(dict_out["sample_atom_coords"])))