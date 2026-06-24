from __future__ import annotations

import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)

from jpromera.config import DIFFUSION
from jpromera.features import build_msas, copy_coords, featurize
from jpromera.model import JPromera as _JPromera
from jpromera.serialize import load_pretrained
from mosaic.losses.promera import (
    _DEFAULT_SAMPLING_STEPS,
    MosaicFeatures,
    apply_binder_sequence,
    jpromera_output,
)

_POLY = {
    PolymerType.PROTEIN: "protein",
    PolymerType.RNA: "rna",
    PolymerType.DNA: "dna",
}


class _StructureWriter:
    def __init__(self, struct):
        self.struct = struct

    def __call__(self, coords) -> "gemmi.Structure":
        import gemmi

        s = copy_coords(self.struct, np.asarray(coords))
        with tempfile.NamedTemporaryFile(suffix=".cif") as f:
            s.to_mmcif(f.name, metadata=False)
            return gemmi.read_structure(f.name)


def _schema(chains: list[TargetChain]) -> dict:
    return {
        cid: {"type": _POLY[c.polymer_type], "sequence": c.sequence}
        for cid, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    }


class JPromeraModel(StructurePredictionModel):
    model: _JPromera
    subsample: int = 1024

    def __init__(self, model: _JPromera | None = None, subsample: int = 1024):
        self.model = model if model is not None else load_pretrained()
        self.subsample = subsample

    def _features(self, chains: list[TargetChain], *, binder_len=None):
        from tinyprot.structure import Structure

        schema = _schema(chains)
        struct0 = Structure.from_schema(schema)
        want = {
            cid: struct0.chains[cid]
            for cid, c in zip(struct0.chains, chains)
            if c.use_msa
        }
        if want:
            build_msas(want)
        feats, struct = featurize(schema, build_msa=False)
        mf = MosaicFeatures.of(feats, binder_len=binder_len)
        return mf, _StructureWriter(struct)

    def target_only_features(self, chains: list[TargetChain]):
        return self._features(list(chains))

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        binder = TargetChain(sequence="X" * binder_length, use_msa=False)
        return self._features([binder] + list(chains), binder_len=binder_length)

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=3,
        sampling_steps=None,
        mask_fraction=0.0,
        num_samples=1,
    ):
        from mosaic.losses.promera import JPromeraLoss

        return JPromeraLoss(
            model=self.model,
            features=features,
            loss=loss,
            recycling_steps=recycling_steps,
            sampling_steps=(
                sampling_steps
                if sampling_steps is not None
                else _DEFAULT_SAMPLING_STEPS
            ),
            mask_fraction=mask_fraction,
            subsample=self.subsample,
            num_samples=num_samples,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: MosaicFeatures,
        recycling_steps: int = 3,
        sampling_steps: int | None = None,
        step_scale: float | None = None,
        key,
    ):
        if PSSM is not None:
            feats, atom37_idx = apply_binder_sequence(PSSM, features, key=key)
        else:
            feats, atom37_idx = features.feats, features.atom37_idx
        diffusion_cfg = (
            DIFFUSION if step_scale is None else {**DIFFUSION, "step_scale": step_scale}
        )
        return jpromera_output(
            self.model,
            feats,
            atom37_idx,
            recycling_steps=recycling_steps,
            num_steps=(
                sampling_steps
                if sampling_steps is not None
                else _DEFAULT_SAMPLING_STEPS
            ),
            diffusion_cfg=diffusion_cfg,
            subsample=self.subsample,
            key=key,
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        writer,
        recycling_steps: int = 3,
        sampling_steps: int | None = None,
        step_scale: float | None = None,
        key,
    ) -> StructurePrediction:
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            step_scale=step_scale,
            key=key,
        )
        if PSSM is not None:
            iptm = -IPTMLoss()(PSSM, output, key=jax.random.key(0))[0]
        else:
            iptm = jnp.nan
        return StructurePrediction(
            st=writer(output.structure_coordinates),
            plddt=output.plddt,
            pae=output.pae,
            iptm=iptm,
            model_output=output,
        )
