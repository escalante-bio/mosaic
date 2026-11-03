from __future__ import annotations

import dataclasses
import tempfile
from copy import deepcopy

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from jpromera.config import DIFFUSION, NTOKS
from jpromera.features import build_msas, copy_coords, featurize
from jpromera.model import InitialEmbedding
from jpromera.model import JPromera as _JPromera
from jpromera.model import TrunkState as PromeraTrunkState
from jpromera.serialize import load_model as load_promera_model
from jpromera.trunk import TrunkOutput

import mosaic.structure_model as sm
from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import scatter_atom37
from mosaic.losses.promera import (
    _AA0,
    DISTOGRAM_BINS,
    MosaicFeatures,
    apply_binder_sequence,
)
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput
from mosaic.sampling import Denoise, Sampler
from mosaic.sampling.promera import promera_denoiser, promera_sampler
from mosaic.sampling.schedules import Schedule
from mosaic.structure_model import (
    AbstractStructureModel,
    Atom37,
    Confidence,
    Distogram,
    TokenInfo,
    TrunkState,
)
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    TargetChain,
)

# `model_output` and the loss builders.
_DESIGN_SAMPLING_STEPS = 100
_DESIGN_RECYCLING_STEPS = 3
# `predict`: upstream's `inference/cofolding.yaml`.
_INFERENCE_SAMPLING_STEPS = 200
_INFERENCE_RECYCLING_STEPS = 4

_POLY = {
    PolymerType.PROTEIN: "protein",
    PolymerType.RNA: "rna",
    PolymerType.DNA: "dna",
}


class _StructureWriter:
    def __init__(self, struct, schema):
        self.struct = struct
        self.schema = schema

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        struct = self.struct
        if binder_pssm is not None:
            schema = deepcopy(self.schema)
            binder_id = next(iter(schema))
            binder_sequence = "".join(
                TOKENS[index] for index in np.asarray(binder_pssm).argmax(-1)
            )
            schema[binder_id]["sequence"] = binder_sequence
            _features, struct = featurize(schema, build_msa=False)
        s = copy_coords(struct, np.asarray(coords))
        with tempfile.NamedTemporaryFile(suffix=".cif") as f:
            s.to_mmcif(f.name, metadata=False)
            return gemmi.read_structure(f.name)


def _schema(chains: list[TargetChain]) -> dict:
    return {
        cid: {"type": _POLY[c.polymer_type], "sequence": c.sequence}
        for cid, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    }


def load_pretrained():
    from huggingface_hub import hf_hub_download

    cache = cache_dir() / "promera"
    stem = hf_hub_download(
        repo_id="escalante-bio/jpromera",
        filename="promera_2606.eqx",
        cache_dir=cache,
    )[: -len(".eqx")]
    hf_hub_download(
        repo_id="escalante-bio/jpromera",
        filename="promera_2606.skeleton.pkl",
        cache_dir=cache,
    )
    return load_promera_model(stem)


class JPromeraModel(AbstractStructureModel[MosaicFeatures, InitialEmbedding]):
    """ProMERA. `recycling_steps` counts trunk passes, as upstream's does."""

    model: _JPromera
    subsample: int = 1024
    # Fraction of binder tokens masked to UNK each step (design losses only).
    mask_fraction: float | Array = 0.0

    def __init__(self, model: _JPromera | None = None, subsample: int = 1024, mask_fraction=0.0):
        self.model = model if model is not None else load_pretrained()
        self.subsample = subsample
        self.mask_fraction = mask_fraction

    @property
    def design_passes(self) -> int:
        return _DESIGN_RECYCLING_STEPS

    @property
    def inference_passes(self) -> int:
        return _INFERENCE_RECYCLING_STEPS

    @property
    def backprop_all_passes(self) -> bool:
        return False

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
        return mf, _StructureWriter(struct, schema)

    def target_only_features(self, chains: list[TargetChain]):
        return self._features(list(chains))

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        binder = TargetChain(sequence="X" * binder_length, use_msa=False)
        return self._features([binder] + list(chains), binder_len=binder_length)

    def design_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return promera_sampler(num_steps or _DESIGN_SAMPLING_STEPS, DIFFUSION, schedule=schedule)

    def inference_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return promera_sampler(num_steps or _INFERENCE_SAMPLING_STEPS, DIFFUSION, schedule=schedule)

    # -- the structure-model contract -----------------------------------------

    def set_sequence(
        self, features: MosaicFeatures, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ) -> MosaicFeatures:
        """Set the binder's sequence channels and repack its atoms from
        `argmax(sequence)`, masking `mask_fraction` of it to UNK."""
        feats, atom37_idx = apply_binder_sequence(
            sequence, features, mask_fraction=self.mask_fraction, key=key
        )
        return dataclasses.replace(features, feats=feats, atom37_idx=atom37_idx)

    def embed(self, features: MosaicFeatures, *, key: PRNGKeyArray) -> InitialEmbedding:
        del key
        return self.model.embed_inputs(features.feats)

    def init_state(self, features: MosaicFeatures, inputs: InitialEmbedding, *, key: PRNGKeyArray) -> TrunkState:
        del features, key
        state = self.model.init_state(inputs)
        return TrunkState(pair=state.z[0], single=state.s[0])

    def step(self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        """One pass on a random subset of `subsample` MSA rows."""
        out = self.model.step(
            features.feats, inputs, _promera_state(state), key=key, subsample=self.subsample
        )
        return TrunkState(pair=out.z[0], single=out.s[0])

    def _trunk_output(self, inputs: InitialEmbedding, state: TrunkState) -> TrunkOutput:
        z = state.pair[None]
        return TrunkOutput(
            s=state.single[None], z=z, s_inputs=inputs.s_inputs, s_init=inputs.s_init,
            z_init=inputs.z_init, relpos=inputs.relpos, pdistogram=self.model.distogram_module(z),
        )

    def structure_denoiser(
        self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState
    ) -> tuple[Denoise, Float[Array, " A"]]:
        return promera_denoiser(self.model, features.feats, self._trunk_output(inputs, state))

    def distogram(self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState) -> Distogram:
        del features, inputs
        return Distogram(
            logits=self.model.distogram_module(state.pair[None])[0],
            bins=jnp.asarray(DISTOGRAM_BINS),
        )

    def confidence(
        self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState, coords: Float[Array, "A 3"]
    ) -> Confidence:
        out = self.model.sm_confidence_module(
            features.feats, self._trunk_output(inputs, state), coords[None], multiplicity=1
        )
        return Confidence(plddt=out.plddt[0], pae_logits=out.pae_logits[0], pae_bins=jnp.asarray(PAE_BINS))

    def tokens(self, features: MosaicFeatures) -> TokenInfo:
        feats = features.feats
        restype = feats.restype if feats.restype.ndim == 3 else jax.nn.one_hot(feats.restype, NTOKS)
        return TokenInfo(
            sequence=restype[0][:, _AA0 : _AA0 + 20],
            asym_id=feats.asym_id[0],
            residue_index=feats.residue_index[0],
            mask=feats.token_pad_mask[0].astype(bool),
        )

    def atom37(self, features: MosaicFeatures, coords: Float[Array, "A 3"]) -> Atom37:
        feats = features.feats
        atom37_coords, atom37_mask = scatter_atom37(
            coords, feats.atom_to_token[0], features.atom37_idx, feats.restype.shape[1]
        )
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(self, features, writer, output: StructureModelOutput, sequence) -> gemmi.Structure:
        del features
        return writer(output.structure_coordinates, binder_pssm=sequence)

    # -- public API (forwards to `mosaic.structure_model`) ----------------------

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=_DESIGN_RECYCLING_STEPS,
        mask_fraction=0.0,
        num_samples=1,
        sampler: Sampler | None = None,
        backprop_all_passes: bool | None = None,
    ) -> sm.MultiSampleLoss:
        return sm.build_loss(
            dataclasses.replace(self, mask_fraction=jnp.asarray(mask_fraction, jnp.float32)),
            loss=loss,
            features=features,
            num_passes=recycling_steps,
            num_samples=num_samples,
            sampler=sampler,
            backprop_all_passes=backprop_all_passes,
        )

    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: MosaicFeatures,
        recycling_steps: int = _DESIGN_RECYCLING_STEPS,
        step_scale: float | None = None,
        sampler: Sampler | None = None,
        key,
    ) -> StructureModelOutput:
        """`step_scale` overrides the sampler's."""
        sampler = sampler if sampler is not None else self.design_sampler()
        if step_scale is not None:
            sampler = dataclasses.replace(sampler, step_scale=step_scale)
        return sm.model_output(
            self, features, sequence=PSSM, num_passes=recycling_steps, sampler=sampler, key=key
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        writer,
        recycling_steps: int = _INFERENCE_RECYCLING_STEPS,
        step_scale: float | None = None,
        sampler: Sampler | None = None,
        key,
    ) -> StructurePrediction:
        sampler = sampler if sampler is not None else self.inference_sampler()
        if step_scale is not None:
            sampler = dataclasses.replace(sampler, step_scale=step_scale)
        return sm.predict(
            self, features, writer, sequence=PSSM, num_passes=recycling_steps, sampler=sampler, key=key
        )


def _promera_state(state: TrunkState) -> PromeraTrunkState:
    return PromeraTrunkState(s=state.single[None], z=state.pair[None])
