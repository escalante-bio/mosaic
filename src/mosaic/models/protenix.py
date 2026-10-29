# Note `model_output` and the losses use a deterministic ODE sampler by default;
# `predict` uses upstream's.
import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import protenix.backend as protenix_backend
from jaxtyping import Array, Float, PyTree

from protenix.data.template import ChainInput, featurize

from mosaic.common import TOKENS
from mosaic.losses.protenix import (
    MultiSampleProtenixLoss,
    ProtenixDesignFeatures,
    binder_atoms_key,
    binder_design_features,
    biotite_array_to_gemmi_struct,
    build_atom_templates,
    get_trunk_state,
    protenix_forward_from_trunk,
    set_binder_sequence,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.cache import cache_dir
from mosaic.sampling import Sampler
from mosaic.sampling.protenix import protenix_sampler
from mosaic.structure_prediction import (
    DiffusionStructurePredictionModel,
    PolymerType,
    StructurePrediction,
    TargetChain,
)

_MSA_AUTH_FIX_INSTALLED = False


def install_protenix_msa_auth_fix():
    global _MSA_AUTH_FIX_INSTALLED
    if _MSA_AUTH_FIX_INSTALLED:
        return
    try:
        import protenix.web_service.colab_request_utils as _cru
        _cru.HTTPBasicAuth = lambda *a, **kw: None
    except Exception:
        return
    _MSA_AUTH_FIX_INSTALLED = True

install_protenix_msa_auth_fix()

@functools.cache
def atom_templates():
    """Per-residue atom templates read off one featurization: residues 1-20 of
    `G + TOKENS + G` (internal) and the last residue of each `G + aa` (C-terminal)."""
    sequences = ["G" + TOKENS + "G", *("G" + aa for aa in TOKENS)]
    features, atom_array, _ = featurize([ChainInput(s, compute_msa=False) for s in sequences])
    tokens = [range(1, 21), [len(sequences[0]) + 2 * i + 1 for i in range(20)]]
    return build_atom_templates(features, atom_array.ref_pos, tokens)


def _chain_input(chain: TargetChain | ChainInput) -> ChainInput:
    """Protenix chains pass through (e.g. with `precomputed_msa_dir`)."""
    if isinstance(chain, ChainInput):
        return chain
    assert chain.polymer_type == PolymerType.PROTEIN, (
        "Protenix interface only supports Protein chains. Manually build features for more complex targets. "
    )
    return ChainInput(sequence=chain.sequence, compute_msa=chain.use_msa, template=chain.template_chain)


def load_model(name="protenix_mini_default_v0.5.0"):
    protenix_backend._CACHE_DIR = str(cache_dir() / "protenix")
    return protenix_backend.load_model(name)


class Protenix(DiffusionStructurePredictionModel):
    """`recycling_steps` counts trunk passes (upstream's `N_cycle`)."""

    protenix: eqx.Module
    design_sampling_steps: int
    # Upstream's per-model inference settings (`configs_model_type.py`).
    inference_sampling_steps: int
    inference_recycling_steps: int

    def target_only_features(self, chains: list[TargetChain | ChainInput]):
        features_dict, atom_array, _ = featurize([_chain_input(c) for c in chains])
        return features_dict, atom_array

    def binder_features(self, binder_length, chains: list[TargetChain | ChainInput]):
        """A poly-Trp binder followed by `chains`, as a `ProtenixDesignFeatures`
        pack: a PSSM passed with it rebuilds the binder's atoms (Protenix's own
        featurization of `argmax(PSSM)`, padded to the poly-Trp atom budget).
        The writer is the placeholder's; `predict` writes the atom37 view instead."""
        binder = TargetChain(sequence="W" * binder_length, use_msa=False)
        features, atom_array = self.target_only_features([binder, *chains])
        return binder_design_features(features, binder_length, atom_templates()), atom_array

    def design_sampler(self, num_steps: int | None = None) -> Sampler:
        """Upstream's sampler as a deterministic ODE, without random augmentation."""
        return dataclasses.replace(
            self.inference_sampler(num_steps or self.design_sampling_steps),
            gamma_0=0.0,
            step_scale=1.0,
            augment=False,
        )

    def inference_sampler(self, num_steps: int | None = None) -> Sampler:
        return protenix_sampler(self.protenix, num_steps or self.inference_sampling_steps)

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        initial_recycling_state=None,
        sampler: Sampler | None = None,
    ):
        return self.build_multisample_loss(
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            num_samples=1,
            initial_recycling_state=initial_recycling_state,
            sampler=sampler,
        )

    def build_multisample_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        num_samples: int = 4,
        reduction=jnp.mean,
        initial_recycling_state=None,
        sampler: Sampler | None = None,
    ):
        return MultiSampleProtenixLoss(
            model=self.protenix,
            features=features,
            loss=loss,
            sampler=sampler if sampler is not None else self.design_sampler(),
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            reduction=reduction,
            initial_recycling_state=initial_recycling_state,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        initial_recycling_state=None,
        sampler: Sampler | None = None,
        key,
    ):
        if PSSM is not None:
            features = set_binder_sequence(PSSM, features, binder_atoms_key(key))
        elif isinstance(features, ProtenixDesignFeatures):
            features = features.features  # the poly-Trp placeholder

        initial_embedding, trunk_state = get_trunk_state(
            model=self.protenix,
            features=features,
            initial_recycling_state=initial_recycling_state,
            recycling_steps=recycling_steps,
            key=key,
        )

        return protenix_forward_from_trunk(
            model=self.protenix,
            features=features,
            initial_embedding=initial_embedding,
            trunk_state=trunk_state,
            sampler=sampler if sampler is not None else self.design_sampler(),
            key=key,
        )

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer,
        recycling_steps=None,
        initial_recycling_state=None,
        sampler: Sampler | None = None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=(
                recycling_steps if recycling_steps is not None else self.inference_recycling_steps
            ),
            initial_recycling_state=initial_recycling_state,
            sampler=sampler if sampler is not None else self.inference_sampler(),
            key=key,
        )
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
        return StructurePrediction(
            st=(
                output.to_structure()
                if isinstance(features, ProtenixDesignFeatures)
                else biotite_array_to_gemmi_struct(writer, np.array(output.structure_coordinates[0]))
            ),
            plddt=output.plddt,
            pae=output.pae,
            iptm=iptm,
            model_output=output,
        )


def _protenix(name: str, *, design_sampling_steps: int, mini: bool) -> Protenix:
    return Protenix(
        load_model(name=name),
        design_sampling_steps=design_sampling_steps,
        inference_sampling_steps=5 if mini else 200,
        inference_recycling_steps=4 if mini else 10,
    )


def ProtenixMini():
    return _protenix("protenix_mini_default_v0.5.0", design_sampling_steps=2, mini=True)


def ProtenixTiny():
    return _protenix("protenix_tiny_default_v0.5.0", design_sampling_steps=2, mini=True)


def ProtenixBase():
    return _protenix("protenix_base_default_v1.0.0", design_sampling_steps=20, mini=False)


def Protenix2025():
    return _protenix("protenix_base_20250630_v1.0.0", design_sampling_steps=20, mini=False)


def ProtenixV2():
    return _protenix("protenix-v2", design_sampling_steps=20, mini=False)
