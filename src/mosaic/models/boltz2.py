from mosaic.structure_prediction import (
    StructurePredictionModel,
    TargetChain,
    StructurePrediction,
)

from mosaic.losses.structure_prediction import IPTMLoss

from mosaic.losses.boltz2 import (
    load_boltz2 as lb,
    load_features_and_structure_writer,
    set_binder_sequence,
    Boltz2Loss,
    Boltz2Output,
)

from pathlib import Path
from jaxtyping import Array, Float, PyTree
import equinox as eqx
import jax
import jax.numpy as jnp


class Boltz2(eqx.Module, StructurePredictionModel):
    model: eqx.Module

    def __init__(self, cache_path: Path | None = None):
        self.model = lb(cache_path) if cache_path is not None else lb()

    @staticmethod
    def _prefix():
        return """version: 1
sequences:"""

    @staticmethod
    def chain_yaml(chain_name: str, chain: TargetChain) -> str:
        assert chain.template_chain is None, (
            "Templates not supported for Boltz2 interface yet (construct a loss manually using mosaic.losses.boltz2)"
        )
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
        return Boltz2Loss(
            joltz2=self.model,
            features=features,
            recycling_steps=recycling_steps
            - 1,  # Really awkward off-by-one issue in Joltz2 :/
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
            features = set_binder_sequence(PSSM, features)

        return Boltz2Output(
            joltz2=self.model,
            features=features,
            recycling_steps=recycling_steps
            - 1,  # Really awkward off-by-one issue in Joltz2 :/
            num_sampling_steps=sampling_steps if sampling_steps is not None else 25,
            key=key,
            deterministic=True,
        )

    @eqx.filter_jit
    def _coords_and_confidences(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=key,
        )

        coords = output.structure_coordinates
        pae = output.pae
        plddt = output.plddt
        if PSSM is None:
            PSSM = jnp.zeros((0, 20))
        iptm = -IPTMLoss()(PSSM, output, key=jax.random.key(0))[0]
        return coords, pae, plddt, iptm

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
        coords, pae, plddt, iptm = self._coords_and_confidences(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=key,
        )

        return StructurePrediction(st=writer(coords), plddt=plddt, pae=pae, iptm=iptm)
