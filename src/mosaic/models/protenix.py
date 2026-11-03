import dataclasses

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import protenix.backend as protenix_backend
from jaxtyping import Array, Float, PRNGKeyArray
from protenix.data.template import ChainInput, featurize
from protenix.protenij import InitialEmbedding, TrunkEmbedding

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import scatter_atom37
from mosaic.losses.protenix import (
    _TOKATOM_TO_ATOM37,
    BOLTZ_TO_PROTENIX,
    PROTENIX_DISTOGRAM_BINS,
    ProtenixDesignFeatures,
    binder_design_features,
    biotite_array_to_gemmi_struct,
    build_atom_templates,
    set_binder_sequence,
)
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput
from mosaic.sampling import Denoise, Sampler
from mosaic.sampling.protenix import protenix_denoiser, protenix_sampler
from mosaic.sampling.schedules import Schedule
from mosaic.structure_prediction import (
    Atom37,
    Confidence,
    DiffusionStructurePredictionModel,
    Distogram,
    PolymerType,
    TargetChain,
    TokenInfo,
    TrunkState,
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


class Protenix(DiffusionStructurePredictionModel[dict | ProtenixDesignFeatures, InitialEmbedding]):
    """`recycling_steps` counts trunk passes (upstream's `N_cycle`)."""

    protenix: eqx.Module
    # Upstream's per-model inference settings (`configs_model_type.py`).
    upstream_sampling_steps: int
    inference_recycling_steps: int
    # Initial trunk state; zeros if None.
    initial_recycling_state: TrunkEmbedding | None = None
    design_recycling_steps: int = eqx.field(static=True, default=1)
    backprop_all_passes: bool = eqx.field(static=True, default=False)

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

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return protenix_sampler(
            self.protenix, num_steps or self.upstream_sampling_steps, schedule=schedule
        )

    # -- the structure-model contract -----------------------------------------

    def set_sequence(self, features, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray):
        """Set the binder's sequence channels; a design pack also rebuilds the
        binder's atoms from `argmax(sequence)`."""
        refreshed = set_binder_sequence(sequence, features, key)
        if isinstance(features, ProtenixDesignFeatures):
            return dataclasses.replace(features, features=refreshed)
        return refreshed

    def embed(self, features, *, key: PRNGKeyArray) -> InitialEmbedding:
        del key
        return self.protenix.embed_inputs(input_feature_dict=_feat(features))

    def init_state(self, features, inputs: InitialEmbedding, *, key: PRNGKeyArray) -> TrunkState:
        """Zeros, or `initial_recycling_state` when set."""
        del features, key
        state = self.initial_recycling_state or self.protenix.init_state(initial_embedding=inputs)
        return TrunkState(pair=state.z, single=state.s)

    def step(self, features, inputs: InitialEmbedding, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        out = self.protenix.step(
            initial_embedding=inputs,
            input_feature_dict=_feat(features),
            state=_protenij_state(state),
            key=key,
        )
        return TrunkState(pair=out.z, single=out.s)

    def structure_denoiser(
        self, features, inputs: InitialEmbedding, state: TrunkState
    ) -> tuple[Denoise, Float[Array, " A"]]:
        return protenix_denoiser(self.protenix, _feat(features), inputs, _protenij_state(state))

    def distogram(self, features, inputs: InitialEmbedding, state: TrunkState) -> Distogram:
        del features, inputs
        return Distogram(
            logits=self.protenix.distogram_head(state.pair),
            bins=jnp.array(PROTENIX_DISTOGRAM_BINS),
        )

    def confidence(
        self, features, inputs: InitialEmbedding, state: TrunkState, coords: Float[Array, "A 3"]
    ) -> Confidence:
        feats = _feat(features)
        out = self.protenix.confidence_metrics(
            initial_embedding=inputs,
            trunk_embedding=_protenij_state(state),
            input_feature_dict=feats,
            coordinates=coords[None],
            key=jax.random.key(0),  # inference: no dropout
        )
        # pLDDT normalized to [0, 1]
        plddt = (
            jax.nn.softmax(out.plddt_logits[0][feats["atom_rep_atom_idx"]])
            * jnp.linspace(0, 1, 50)[None, :]
        ).sum(-1)
        return Confidence(plddt=plddt, pae_logits=out.pae_logits[0], pae_bins=jnp.array(PAE_BINS))

    def tokens(self, features) -> TokenInfo:
        feats = _feat(features)
        n = feats["restype"].shape[0]
        return TokenInfo(
            sequence=feats["restype"] @ BOLTZ_TO_PROTENIX.T,
            asym_id=feats["asym_id"],
            residue_index=feats["residue_index"],
            mask=feats.get("token_pad_mask", jnp.ones(n, bool)),
        )

    def atom37(self, features, coords: Float[Array, "A 3"]) -> Atom37:
        feats = _feat(features)
        a2t = feats["atom_to_token_idx"]
        atom37_idx = jnp.asarray(_TOKATOM_TO_ATOM37)[feats["restype"].argmax(-1)[a2t], feats["atom_to_tokatom_idx"]]
        atom37_idx = jnp.where(feats["ref_mask"] > 0.5, atom37_idx, jnp.int32(-1))
        atom37_coords, atom37_mask = scatter_atom37(coords, a2t, atom37_idx, feats["restype"].shape[0])
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(self, features, writer, output: StructureModelOutput, sequence) -> gemmi.Structure:
        """A design pack is written from the atom37 view; otherwise with `writer`."""
        del sequence
        if isinstance(features, ProtenixDesignFeatures):
            return output.to_structure()
        return biotite_array_to_gemmi_struct(writer, np.array(output.structure_coordinates))


def _feat(features) -> dict:
    return features.features if isinstance(features, ProtenixDesignFeatures) else features


def _protenij_state(state: TrunkState) -> TrunkEmbedding:
    return TrunkEmbedding(s=state.single, z=state.pair)


def _protenix(name: str, *, mini: bool) -> Protenix:
    return Protenix(
        load_model(name=name),
        upstream_sampling_steps=5 if mini else 200,
        inference_recycling_steps=4 if mini else 10,
    )


def ProtenixMini():
    return _protenix("protenix_mini_default_v0.5.0", mini=True)


def ProtenixTiny():
    return _protenix("protenix_tiny_default_v0.5.0", mini=True)


def ProtenixBase():
    return _protenix("protenix_base_default_v1.0.0", mini=False)


def Protenix2025():
    return _protenix("protenix_base_20250630_v1.0.0", mini=False)


def ProtenixV2():
    return _protenix("protenix-v2", mini=False)
