"""ESMFold2 loss + soft-sequence plumbing."""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from esmjfold2 import Features
from esmjfold2.lm_features import _prepare_lm_inputs, BOS_ID, EOS_ID
from esmjfold2.model import ESMFold2 as _JaxESMFold2Release
from esmjfold2.experimental import ESMFold2Experimental as _JaxESMFold2Experimental
from esmjfold2.esmc import ESMC, ESMCForMaskedLM

from mosaic.common import LossTerm, LinearCombination, TOKENS
from mosaic.losses.structure_prediction import StructureModelOutput, PAE_BINS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.transformations import norm_gradient

# Either flavor of the JAX trunk class. Both expose the same
# `_prepare_embeddings` / `_run_trunk` / `_sample_structure` /
# `_compute_distogram` / `_compute_confidence` / `language_model` interface,
# so every helper here is variant-agnostic.
ESMFold2 = _JaxESMFold2Release | _JaxESMFold2Experimental


NUM_RES_TYPES = 33  # esmjfold2.inputs.NUM_RES_TYPES


# ---------------------------------------------------------------------------
# Feature pack
# ---------------------------------------------------------------------------


class EsmFold2FeaturePack(eqx.Module):
    """Bundle of everything the model forward needs at design time.

    Convention: the binder, when present, lives at token positions
    `[0, pssm.shape[0])` and asym_id 0. The loss reads `pssm.shape[0]` to know
    the binder length.
    """

    features: Features
    # [B, L_total, n_layers+1, D] — ESMC hidden states computed once at
    # feature-build time, with input_ids rewritten to UNK at binder slots.
    # Frozen throughout design.
    target_lm_hidden: Float[Array, "B L NL D"]
    # Per-atom mapping to one of 37 atom37 slots (-1 = no slot).
    atom37_idx: Int[Array, "A"]
    # Backbone atom index in the per-atom layout, stacked N/CA/C/O.
    backbone_atom_idx: Int[Array, "4 L"]
    # Whether the forward refreshes the binder in-loop from argmax(PSSM) —
    # geometry (atom14 repack) AND ESMC LM features, the two tied together. True
    # for a `binder_features` design pack (W / 14-atom allocation); False for a
    # native `target_only_features` pack, which refreshes neither. Static, so it
    # selects the forward path at trace time and is part of the JIT signature.
    refresh: bool = eqx.field(static=True, default=False)


# ---------------------------------------------------------------------------
# Soft sequence injection into Features.res_type
# ---------------------------------------------------------------------------


def set_binder_sequence(
    pssm: Float[Array, "N 20"],
    features: Features,
    res_type_perm: Float[Array, "20 33"],
) -> Features:
    """Splice the soft PSSM into the first `pssm.shape[0]` positions of
    `res_type` (and the MSA query row, if present).

    `res_type`/`msa` are promoted to float so `_prepare_embeddings` treats
    them as already-one-hot.

    The homolog rows' binder columns are also masked out in
    `msa_attention_mask` so the binder presents as a single-sequence region:
    its profile becomes the soft PSSM itself rather than a gap-diluted average
    over depth, and the profile-path gradient isn't divided by the MSA depth.
    Inert when the MSA is depth-1 (our usual no-MSA setup) — there are no
    homolog rows to mask.
    """
    N = pssm.shape[0]
    res_type = features.res_type
    if res_type.ndim == 2:
        res_type_oh = jax.nn.one_hot(
            res_type.astype(jnp.int32), NUM_RES_TYPES
        ).astype(jnp.float32)
    else:
        res_type_oh = res_type.astype(jnp.float32)
    soft = pssm @ res_type_perm  # [N, 33]
    new_res_type = res_type_oh.at[0, :N].set(soft)
    features = eqx.tree_at(lambda f: f.res_type, features, new_res_type)

    # MSA query row: the input builder populates `features.msa` even in
    # single-sequence mode (depth-1 MSA whose row 0 is the query). Without
    # patching this, `_prepare_embeddings` aggregates an X-frozen profile at
    # binder positions and the gradient through the profile path is dead.
    if features.msa is not None:
        msa = features.msa
        if msa.ndim == 3:
            msa_oh = jax.nn.one_hot(
                msa.astype(jnp.int32), NUM_RES_TYPES
            ).astype(jnp.float32)
        else:
            msa_oh = msa.astype(jnp.float32)
        new_msa = msa_oh.at[0, 0, :N].set(soft)
        features = eqx.tree_at(lambda f: f.msa, features, new_msa)

        # Make the binder single-sequence: the builder gap-fills (and counts)
        # the binder columns of every homolog row in a paired MSA, which
        # otherwise dilutes the binder profile toward gap and divides its
        # gradient by depth. Masking those columns leaves only the query row
        # valid there, so profile[binder] == soft PSSM. Empty (no-op) slice
        # when depth==1.
        if features.msa_attention_mask is not None:
            new_attn = features.msa_attention_mask.at[0, 1:, :N].set(False)
            features = eqx.tree_at(
                lambda f: f.msa_attention_mask, features, new_attn
            )

    return features


# ---------------------------------------------------------------------------
# Trunk / heads split (mirrors boltz1_trunk / boltz1_forward_from_trunk)
# ---------------------------------------------------------------------------


def esmfold2_trunk(
    esmf: ESMFold2,
    features: Features,
    lm_hidden_states: Float[Array, "B L NL D"] | None,
    key,
    *,
    num_loops: int,
    msa_max_depth: int | None = 1024,
):
    """Returns the per-token pair representation `z` + the embedding context.

    Heads (diffusion sampler, confidence, distogram) consume both. Recycling
    backprops through every iteration.
    """
    ctx = esmf._prepare_embeddings(features)
    # The LM is a frozen feature extractor — stop-gradient at the consumption
    # site so no gradient reaches the design PSSM through the LM, regardless of
    # how `lm_hidden_states` was produced. (The cached `target_lm_hidden` is
    # already detached; this re-asserts it.)
    lm_z = (
        esmf.language_model(jax.lax.stop_gradient(lm_hidden_states))
        if lm_hidden_states is not None
        else None
    )
    ktrunk, _ = jax.random.split(key)
    z = esmf._run_trunk(
        ctx, lm_z, ktrunk, num_loops=num_loops, msa_max_depth=msa_max_depth
    )
    return ctx, z


def esmfold2_forward_from_trunk(
    esmf: ESMFold2,
    ctx,
    z,
    key,
    *,
    num_sampling_steps: int,
    noise_scale: float | None = None,
    step_scale: float | None = None,
):
    """Run the diffusion sampler + confidence + distogram heads.

    If the downstream loss only reads trunk-derived fields (distogram,
    asym_id, residue_idx) and ignores `structure_coordinates` /
    `atom37_*` / `plddt` / `pae`, XLA's DCE prunes both `_sample_structure`
    and `_compute_confidence` from the compiled graph — no flag needed.
    """
    _, ksample = jax.random.split(key)
    sample_atom_coords = esmf._sample_structure(
        ctx, z, ksample,
        num_sampling_steps=num_sampling_steps,
        noise_scale=noise_scale,
        step_scale=step_scale,
    )
    distogram_logits = esmf._compute_distogram(z)
    confidence = esmf._compute_confidence(ctx, z, sample_atom_coords)
    return (
        sample_atom_coords,
        distogram_logits,
        confidence,
    )


# ---------------------------------------------------------------------------
# StructureModelOutput conversion
# ---------------------------------------------------------------------------


def distogram_bin_centers(
    n_bins: int, min_dist: float, max_dist: float,
) -> Float[Array, "Bins"]:
    """Midpoints of `linspace(min_dist, max_dist, n_bins + 1)`. The release
    `ESMFold2-Fast` checkpoint uses (2.0, 22.0, 64); `ESMFold2-Experimental-*`
    uses (2.0, 52.0, 128).
    """
    if n_bins == 128:
        boundaries = np.linspace(2.0, 52.0, 127)
        edges = np.concatenate([np.array([1.0]), boundaries, np.array([52.0 + 5.0])])
        return 0.5 * (edges[:-1] + edges[1:])
    else:
        boundaries = np.linspace(min_dist, max_dist, n_bins + 1, dtype=np.float32)
        return 0.5 * (boundaries[:-1] + boundaries[1:])


def _to_structure_model_output(
    features: Features,
    full_sequence: Float[Array, "L 20"],
    sample_atom_coords: Float[Array, "1 A 3"],
    distogram_logits: Float[Array, "1 L L Bins"],
    distogram_bins: Float[Array, "Bins"],
    confidence: dict,
    atom37_idx: Int[Array, "A"],
    backbone_atom_idx: Int[Array, "4 L"],
) -> StructureModelOutput:
    """Pack esmjfold2 outputs into mosaic's `StructureModelOutput`.

    Drops the batch axis (assumes B=1, which holds for design — diffusion
    samples are produced by re-running with different keys, not batching).
    """
    n_tokens = features.res_type.shape[1]
    coords = sample_atom_coords[0]  # [A, 3]
    atom37_coords, atom37_mask = scatter_atom37(
        atom_coords=coords,
        atom_to_token=features.atom_to_token[0].astype(jnp.int32),
        atom37_idx=atom37_idx,
        n_token=n_tokens,
    )
    # Backbone gather. backbone_atom_idx is stacked [4, L]: N, CA, C, O.
    bb = jnp.transpose(coords[backbone_atom_idx], (1, 0, 2))  # [L, 4, 3]

    return StructureModelOutput(
        distogram_logits=distogram_logits[0],
        distogram_bins=distogram_bins,
        plddt=confidence["plddt"][0],
        pae=confidence["pae"][0],
        pae_logits=confidence["pae_logits"][0],
        pae_bins=jnp.asarray(PAE_BINS, dtype=jnp.float32),
        structure_coordinates=coords,
        backbone_coordinates=bb,
        full_sequence=full_sequence,
        asym_id=features.asym_id[0].astype(jnp.int32),
        residue_idx=features.residue_index[0].astype(jnp.int32),
        atom37_coords=atom37_coords,
        atom37_mask=atom37_mask,
    )


# ---------------------------------------------------------------------------
# Loss term
# ---------------------------------------------------------------------------


def forward_with_pssm(
    esmf: ESMFold2,
    pack: EsmFold2FeaturePack,
    pssm: Float[Array, "N 20"] | None,
    res_type_perm: Float[Array, "20 33"],
    distogram_bins: Float[Array, "Bins"],
    *,
    key,
    num_loops: int,
    num_sampling_steps: int,
    msa_max_depth: int | None,
    geom_templates: ResidueAtomTemplates | None = None,
    esmc: ESMC | None = None,
    esmc_token_of_aa: Int[Array, "20"] | None = None,
    refresh_lm: bool = False,
) -> StructureModelOutput:
    """One end-to-end forward. `pssm=None` → target-only prediction; otherwise
    the PSSM is spliced into the binder slots. The cached UNK LM features in
    `pack.target_lm_hidden` are reused unchanged unless `refresh_lm` is set
    (see `refresh_lm` below and the module docstring).

    `geom_templates`: when set, the binder atom geometry is refreshed in-loop
    from `argmax(pssm)` (see `refresh_binder_geometry`). Requires the 14-atom/
    residue binder allocation that `binder_features` builds (a design pack).

    `refresh_lm`: recompute the binder's ESMC features from `argmax(pssm)` each
    step (detached) instead of reusing the cached UNK features. Re-encodes only
    the binder chain and splices it into `target_lm_hidden` (see
    `recompute_binder_lm_hidden`); requires `esmc` and `esmc_token_of_aa`.
    """
    features = pack.features
    L_total = features.res_type.shape[1]

    # LM features: cached UNK by default; `refresh_lm` recomputes them from the
    # argmax each step (sequence-tracking).
    lm_hidden = pack.target_lm_hidden
    # atom37 / backbone scaffolding tracks the atom layout; the geom refresh
    # repacks it, so those are rebuilt alongside (the pack's are stale then).
    atom37_idx = pack.atom37_idx
    backbone_atom_idx = pack.backbone_atom_idx
    if pssm is not None:
        features = set_binder_sequence(pssm, features, res_type_perm)
        if geom_templates is not None:
            features, atom37_idx, backbone_atom_idx = refresh_binder_geometry(
                features, pssm, geom_templates, atom37_idx, backbone_atom_idx
            )
        if refresh_lm:
            lm_hidden = recompute_binder_lm_hidden(
                esmc, esmc_token_of_aa, lm_hidden, pssm
            )

    k_trunk, k_heads = jax.random.split(key)
    ctx, z = esmfold2_trunk(
        esmf, features, lm_hidden, k_trunk,
        num_loops=num_loops, msa_max_depth=msa_max_depth,
    )
    sample_atom_coords, distogram_logits, confidence = (
        esmfold2_forward_from_trunk(
            esmf, ctx, z, k_heads,
            num_sampling_steps=num_sampling_steps,
        )
    )

    # `full_sequence` carries every token's identity (like every other model
    # wrapper), not just the binder — `res_type @ res_type_perm.T` inverts the
    # one-hot `res_type` back to the 20-dim alphabet (target identities included;
    # consumers such as proteina / protein_mpnn read the target slice). The
    # binder slots are then set to the exact soft PSSM (the design region).
    full_seq = features.res_type[0] @ res_type_perm.T
    if pssm is not None:
        N = pssm.shape[0]
        full_seq = full_seq.at[:N].set(pssm)

    return _to_structure_model_output(
        features=features,
        full_sequence=full_seq,
        sample_atom_coords=sample_atom_coords,
        distogram_logits=distogram_logits,
        distogram_bins=distogram_bins,
        confidence=confidence,
        atom37_idx=atom37_idx,
        backbone_atom_idx=backbone_atom_idx,
    )


class ESMFold2Loss(LossTerm):
    """Mosaic-style loss wrapping an ESMFold2 forward.

    Three optional knobs control how the binder is represented during
    optimization:

      * `geom_templates`: repack the binder atoms from `argmax(PSSM)` each step.
        Requires the 14-atom binder allocation that `binder_features` builds (a
        design pack); a tight `target_only_features` pack produces a broken
        prediction.
      * `refresh_lm`: re-encode the binder's ESMC features from the argmax
        (detached) instead of the frozen UNK features.
      * `lm_dropout`: dropout on the LM→pair contribution.

    With all three off (the default), the binder keeps a fixed placeholder
    backbone and frozen UNK LM features.
    """

    esmf: ESMFold2
    pack: EsmFold2FeaturePack
    loss: LossTerm | LinearCombination
    res_type_perm: Float[Array, "20 33"]
    distogram_bins: Float[Array, "Bins"]
    num_loops: int = eqx.field(static=True)
    num_sampling_steps: int = eqx.field(static=True)
    msa_max_depth: int | None = eqx.field(static=True)
    # In-loop binder geometry refresh. None = fixed placeholder backbone
    # (default). A pytree of arrays, so an ordinary (non-static) leaf.
    geom_templates: ResidueAtomTemplates | None = None
    # ESMC model + per-residue ESMC token map + flag to recompute binder LM
    # features from argmax each step (vs the cached UNK default).
    # `esmc` / `esmc_token_of_aa` are ordinary leaves.
    esmc: ESMC | None = None
    esmc_token_of_aa: Int[Array, "20"] | None = None
    refresh_lm: bool = eqx.field(static=True, default=False)

    def __call__(self, sequence: Float[Array, "N 20"], *, key):
        smo = forward_with_pssm(
            self.esmf, self.pack, sequence,
            self.res_type_perm,
            self.distogram_bins,
            key=key,
            num_loops=self.num_loops,
            num_sampling_steps=self.num_sampling_steps,
            msa_max_depth=self.msa_max_depth,
            geom_templates=self.geom_templates,
            esmc=self.esmc,
            esmc_token_of_aa=self.esmc_token_of_aa,
            refresh_lm=self.refresh_lm,
        )
        return self.loss(sequence, smo, key=key)

class StackedEsmFold2(LossTerm):
    """ Share the same ESMC model across multiple ESMFold models to save VRAM."""
    stacked: ESMFold2Loss
    static: ESMFold2Loss
    n: int = eqx.field(static=True)
    esmc: ESMCForMaskedLM  # only for its backbone (esmc.esmc); no PLL/lm_head use

    def __init__(self, structure_losses: list, esmc):
        if not structure_losses:
            raise ValueError("StackedEsmFold2 needs at least one structure loss.")
        for sl in structure_losses:
            if sl.esmc is not None:
                raise ValueError("loss.esmc must be None for StackedEsmFold2")

        dyn, stat = zip(*(eqx.partition(l, eqx.is_inexact_array) for l in structure_losses))
        self.stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *dyn)
        self.static = stat[0]
        self.n = len(structure_losses)
        self.esmc = esmc

    def __call__(self, sequence, *, key):
        k_pick, k_struct = jax.random.split(key, 2)
        i = jax.random.randint(k_pick, (), 0, self.n)
        model = eqx.combine(
            jax.tree_util.tree_map(lambda x: x[i], self.stacked), self.static
        )
        model = eqx.tree_at(
            lambda m: m.esmc, model, self.esmc.esmc, is_leaf=lambda x: x is None
        )
        v_s, aux_s = model(norm_gradient(1.0, sequence), key=k_struct)
        return v_s, {"model_index": i, "": aux_s}


class StackedEsmFold2PLL(LossTerm):
    """One ESMC shared across a stack of (esmc-less) ESMFold2 structure losses
    plus an ESMC pseudo-perplexity term — a single ESMC buffer for the whole
    loss, to save VRAM.
    """

    # Stacked esmc-less structure losses (each leaf gains a leading axis n) and
    # the shared non-array part — same split as `RandomChoice`.
    stacked: ESMFold2Loss
    static: ESMFold2Loss
    n: int = eqx.field(static=True)
    # The single shared ESMC (backbone + MLM head): the only ESMC in the tree.
    esmc: ESMCForMaskedLM
    # PLL term built with `esm=None`; the shared ESMC is spliced in at call time.
    pll: LossTerm
    pll_weight: float

    def __init__(self, structure_losses, esmc, pll, pll_weight=0.15):
        if not structure_losses:
            raise ValueError("StackedEsmFold2PLL needs at least one structure loss.")
        struct = jax.tree_util.tree_structure(structure_losses[0])
        for i, l in enumerate(structure_losses[1:], 1):
            if jax.tree_util.tree_structure(l) != struct:
                raise ValueError(
                    f"structure_losses[{i}] differs in pytree structure from [0]."
                )
        dyn, stat = zip(
            *(eqx.partition(l, eqx.is_inexact_array) for l in structure_losses)
        )
        self.stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *dyn)
        self.static = stat[0]
        self.n = len(structure_losses)
        self.esmc = esmc
        self.pll = pll
        self.pll_weight = pll_weight

    def __call__(self, sequence: Float[Array, "N 20"], *, key):
        k_pick, k_struct, k_pll = jax.random.split(key, 3)
        i = jax.random.randint(k_pick, (), 0, self.n)
        model = eqx.combine(
            jax.tree_util.tree_map(lambda x: x[i], self.stacked), self.static
        )
        # Splice the shared backbone into the chosen (esmc=None) structure loss
        # and the full shared ESMC into the (esm=None) PLL. Same object both
        # places → one buffer. `norm_gradient` norms each sub-term's gradient.
        model = eqx.tree_at(
            lambda m: m.esmc, model, self.esmc.esmc, is_leaf=lambda x: x is None
        )
        v_s, aux_s = model(norm_gradient(1.0, sequence), key=k_struct)
        pll = eqx.tree_at(
            lambda p: p.esm, self.pll, self.esmc, is_leaf=lambda x: x is None
        )
        v_p, aux_p = pll(norm_gradient(self.pll_weight, sequence), key=k_pll)
        return v_s + self.pll_weight * v_p, {
            "model_index": i,
            "structure": aux_s,
            "pll": aux_p,
        }


# ---------------------------------------------------------------------------
# Atom37 / backbone scaffolding (numpy-side, run once per feature build)
# ---------------------------------------------------------------------------


def _decode_atom_name(chars_row: np.ndarray) -> str:
    """ESMFold2 stores atom names as 4 small ints (`ord(c) - 32`, 0 = pad)."""
    return "".join(chr(int(c) + 32) for c in chars_row if int(c) != 0).strip()


def build_atom37_scaffolding(
    ref_atom_name_chars: np.ndarray,  # [B, A, 4]
    atom_to_token: np.ndarray,         # [B, A]
    atom_attention_mask: np.ndarray,   # [B, A] bool
    n_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (`atom37_idx`, `backbone_atom_idx`) for the per-token scatter.

    Returns:
      * `atom37_idx: [A] int32` — slot 0..36 for each atom, or -1 if no
        atom37 slot. Padding atoms get -1 too.
      * `backbone_atom_idx: [4, L] int32` — atom-index in the global atom
        layout for N, CA, C, O (in that order) per token. 0 if missing.
    """
    A = ref_atom_name_chars.shape[1]
    atom37_idx = np.full((A,), -1, dtype=np.int32)
    bb = np.zeros((4, n_tokens), dtype=np.int32)
    names = ("N", "CA", "C", "O")
    chars = ref_atom_name_chars[0]
    a2t = atom_to_token[0]
    mask = atom_attention_mask[0]
    for a in range(A):
        if not bool(mask[a]):
            continue
        name = _decode_atom_name(chars[a])
        atom37_idx[a] = ATOM37_INDEX.get(name, -1)
        if name in names:
            tok = int(a2t[a])
            if 0 <= tok < n_tokens:
                bb[names.index(name), tok] = a
    return atom37_idx, bb


# ---------------------------------------------------------------------------
# In-loop binder geometry refresh
# ---------------------------------------------------------------------------
#
# Without this, the design path would fold against a fixed placeholder backbone
# for the whole run, with only ``res_type`` / the MSA query row patched by the
# soft PSSM. This refreshes the binder atom geometry from ``argmax(soft_design)``
# each step so the geometry the trunk / atom-encoder sees tracks the evolving
# discrete sequence, *under JIT* (no per-step host re-featurization or recompiles):
#
#   * **fixed buffer via atom14 allocation.** ``binder_features`` allocates the
#     binder at a 14-atom/residue placeholder (Trp has the max, 14) so the atom
#     buffer is big enough to hold any sequence.
#   * **per-residue templates.** Read each of the 20 residue types' reference
#     atoms off the input builder once (``_residue_atom_templates``).
#     ``ref_pos`` is context-independent (a CCD conformer in a local frame), so
#     a single isolated featurization per residue is exact.
#   * **tight-pack by argmax.** Each step, gather the chosen residues' atoms and
#     scatter them tightly at the front (native order, via an exclusive
#     prefix-sum of per-residue atom counts), shift the target atoms to sit
#     immediately after, and mask the tail. Pure ``jnp.take`` + scatter → static
#     shapes, no recompiles.
#
# Every atom field is rebuilt for the tight layout, including the positional
# ``atom_to_token`` / ``ref_space_uid`` and the token-level
# ``distogram_atom_idx``. Tight packing (no masked padding interleaved between
# residues) keeps the atom encoder's windowed attention on-distribution.
#
# The argmax is non-differentiable and the reference conformer is a conditioning
# input, so the gather runs under ``stop_gradient``: the PSSM gradient still
# flows only through the ``res_type`` / MSA channels.

MAX_ATOMS_PER_RES = 14  # Trp; the atom14 budget.


class ResidueAtomTemplates(eqx.Module):
    """Per-residue (mosaic-20 / ``TOKENS`` order) atom14 reference templates.

    Each residue's real atoms occupy the first ``n_atoms`` of the 14 slots,
    backbone-first (N, CA, C, O, [CB, ...]); the rest are masked padding.
    """

    ref_pos: Float[Array, "20 A 3"]
    ref_element: Int[Array, "20 A"]
    ref_charge: Int[Array, "20 A"]
    ref_atom_name_chars: Int[Array, "20 A 4"]
    atom_mask: Bool[Array, "20 A"]
    # Local atom offset (within the residue's 14-slot block) of the distogram
    # representative atom (CB for most, CA for Gly). Token-level field.
    disto_offset: Int[Array, "20"]
    n_atoms: Int[Array, "20"]
    # atom37 slot (0..36, or -1) of each of the 14 local atoms — so the in-loop
    # refresh can rebuild a layout-correct `atom37_idx` (see `refresh_binder_geometry`).
    atom37_slot: Int[Array, "20 A"]
    max_atoms: int = eqx.field(static=True)


def _residue_atom_templates(
    max_atoms: int = MAX_ATOMS_PER_RES,
) -> ResidueAtomTemplates:
    """Per-residue atom14 reference templates for the in-loop geometry refresh.

    Built once on the host (CPU, no model weights) and cached process-wide:
    featurizes each amino acid as an isolated single-residue chain and reads its
    atom block. ``ref_pos`` is context- and checkpoint-independent, so the result
    is a singleton — `ESMFold2.build_loss` pulls it internally for a refresh
    (design) pack, so callers never build it themselves.
    """
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    builder = ESMFold2InputBuilder()
    ref_pos = np.zeros((20, max_atoms, 3), np.float32)
    ref_element = np.zeros((20, max_atoms), np.int32)
    ref_charge = np.zeros((20, max_atoms), np.int32)
    ref_name = np.zeros((20, max_atoms, 4), np.int32)
    atom_mask = np.zeros((20, max_atoms), bool)
    disto_off = np.zeros((20,), np.int32)
    n_atoms = np.zeros((20,), np.int32)
    atom37_slot = np.full((20, max_atoms), -1, np.int32)

    for i, aa in enumerate(TOKENS):
        spi = StructurePredictionInput(
            sequences=[ProteinInput(id="A", sequence=aa)]
        )
        raw, _ = builder.prepare_input(spi)
        m = np.asarray(raw["atom_attention_mask"])[0].astype(bool)
        k = int(m.sum())
        assert k <= max_atoms, f"{aa} has {k} atoms > max_atoms={max_atoms}"
        ref_pos[i, :k] = np.asarray(raw["ref_pos"])[0][m]
        ref_element[i, :k] = np.asarray(raw["ref_element"])[0][m]
        ref_charge[i, :k] = np.asarray(raw["ref_charge"])[0][m]
        names = np.asarray(raw["ref_atom_name_chars"])[0][m]
        ref_name[i, :k] = names
        atom_mask[i, :k] = True
        n_atoms[i] = k
        # `refresh_binder_geometry` rebuilds the binder backbone as local atoms
        # 0..3 (no name lookup), so the builder MUST emit N,CA,C,O first or the
        # refreshed `backbone_atom_idx` is silently wrong. Verified true for the
        # current CCD; assert so a future atom-order change fails loudly here.
        bb_decoded = tuple(_decode_atom_name(names[j]) for j in range(min(4, k)))
        assert bb_decoded == ("N", "CA", "C", "O"), (
            f"{aa}: expected N,CA,C,O as the first 4 atoms, got {bb_decoded}"
        )
        # atom37 slot per local atom, from the same name decode the native pack
        # uses (`build_atom37_scaffolding`), so the refreshed `atom37_idx` matches.
        for j in range(k):
            atom37_slot[i, j] = ATOM37_INDEX.get(_decode_atom_name(names[j]), -1)
        # token 0's representative atom; for an isolated residue the atom index
        # equals the local offset within the residue's block.
        disto_off[i] = int(np.asarray(raw["distogram_atom_idx"])[0][0])

    return ResidueAtomTemplates(
        ref_pos=jnp.asarray(ref_pos),
        ref_element=jnp.asarray(ref_element),
        ref_charge=jnp.asarray(ref_charge),
        ref_atom_name_chars=jnp.asarray(ref_name),
        atom_mask=jnp.asarray(atom_mask),
        disto_offset=jnp.asarray(disto_off),
        n_atoms=jnp.asarray(n_atoms),
        atom37_slot=jnp.asarray(atom37_slot),
        max_atoms=max_atoms,
    )


def refresh_binder_geometry(
    features: Features,
    pssm: Float[Array, "N 20"],
    tmpl: ResidueAtomTemplates,
    atom37_idx: Int[Array, "A"],
    backbone_atom_idx: Int[Array, "4 L"],
) -> tuple[Features, Int[Array, "A"], Int[Array, "4 L"]]:
    """Rebuild the atom axis with the argmax(pssm) binder residues' atoms
    **tightly** packed at the front (native order), the target atoms shifted to
    sit immediately after, and zero-padding at the end.

    Returns the refreshed ``features`` plus the layout-corrected ``atom37_idx``
    (per-atom 0..36 slot) and ``backbone_atom_idx`` (``[4, L]``, N/CA/C/O flat
    atom indices per token). Those two are pack-level scaffolding consumed by
    ``_to_structure_model_output``; because the repack changes the flat atom
    order, the pack's originals are stale and **must** be rebuilt here, else the
    atom37 / backbone views of a refreshed fold are garbage (the flat
    ``structure_coordinates`` stays correct, so this only affects those views).

    Tight packing (no masked padding interleaved between residues) keeps the
    atom encoder's windowed attention on-distribution.

    JIT-safe (static shapes): per-residue atom counts come from the templates,
    tight destinations from an exclusive prefix-sum, and every field is a scatter
    into a fixed-size buffer (``mode='drop'`` discards the out-of-range padding
    slots). Detached — geometry is a conditioning input, so the PSSM gradient
    still flows only through ``res_type`` / MSA.

    Requires the binder allocated at ``max_atoms`` atoms/residue (the 14-atom
    "W" allocation `binder_features` builds) so the fixed buffer can hold any
    sequence.
    """
    ma = tmpl.max_atoms
    L_b = pssm.shape[0]
    A_alloc = L_b * ma                       # binder atom buffer in the W-pack
    A_total = features.ref_pos.shape[1]       # total atoms (static)
    n_tgt = A_total - A_alloc
    ids = jax.lax.stop_gradient(jnp.argmax(pssm, axis=-1))  # [L_b], mosaic-20

    k = tmpl.n_atoms[ids].astype(jnp.int32)               # real atoms / residue
    offset = (jnp.cumsum(k) - k).astype(jnp.int32)        # exclusive prefix sum
    sum_k = jnp.sum(k)

    # Tight binder destination per (residue, local atom); invalid slots (a>=k)
    # map out of range and are dropped by the scatter.
    a_local = jnp.arange(ma, dtype=jnp.int32)[None, :]
    valid = a_local < k[:, None]
    dest = jnp.where(valid, offset[:, None] + a_local, A_total).reshape(-1)  # [L_b*ma]
    b_tok = jnp.repeat(jnp.arange(L_b, dtype=jnp.int32), ma)                 # owning token

    # Target atoms move from [A_alloc, A_total) to sit right after the binder.
    tgt_src = jnp.arange(A_alloc, A_total)
    tgt_dest = sum_k + jnp.arange(n_tgt)

    f = features
    rp, re, rc, rn = f.ref_pos[0], f.ref_element[0], f.ref_charge[0], f.ref_atom_name_chars[0]
    a2t, uid = f.atom_to_token[0], f.ref_space_uid[0]

    def _scatter(buf, binder_vals, target_full):
        buf = buf.at[dest].set(binder_vals.astype(buf.dtype), mode="drop")
        return buf.at[tgt_dest].set(target_full[tgt_src], mode="drop")

    new_pos = _scatter(jnp.zeros_like(rp), tmpl.ref_pos[ids].reshape(-1, 3), rp)
    new_el = _scatter(jnp.zeros_like(re), tmpl.ref_element[ids].reshape(-1), re)
    new_ch = _scatter(jnp.zeros_like(rc), tmpl.ref_charge[ids].reshape(-1), rc)
    new_nm = _scatter(jnp.zeros_like(rn), tmpl.ref_atom_name_chars[ids].reshape(-1, 4), rn)
    new_a2t = _scatter(jnp.zeros_like(a2t), b_tok, a2t)
    new_uid = _scatter(jnp.zeros_like(uid), b_tok, uid)
    # Real atoms occupy [0, sum_k) (binder) + [sum_k, sum_k + n_tgt_real)
    # (target); the rest — including the W-pack's own trailing padding that got
    # shifted along — is masked off at the end.
    n_tgt_real = jnp.sum(f.atom_attention_mask[0, A_alloc:].astype(jnp.int32))
    new_mask = jnp.arange(A_total) < (sum_k + n_tgt_real)

    # Distogram representative atom per token: binder -> tight CB/CA;
    # target -> its old atom index shifted by (sum_k - A_alloc).
    dai = f.distogram_atom_idx[0]
    new_dai = jnp.concatenate(
        [offset + tmpl.disto_offset[ids], dai[L_b:] - A_alloc + sum_k]
    ).astype(dai.dtype)

    f = eqx.tree_at(lambda x: x.ref_pos, f, new_pos[None].astype(rp.dtype))
    f = eqx.tree_at(lambda x: x.ref_element, f, new_el[None].astype(re.dtype))
    f = eqx.tree_at(lambda x: x.ref_charge, f, new_ch[None].astype(rc.dtype))
    f = eqx.tree_at(lambda x: x.ref_atom_name_chars, f, new_nm[None].astype(rn.dtype))
    f = eqx.tree_at(lambda x: x.atom_to_token, f, new_a2t[None].astype(a2t.dtype))
    f = eqx.tree_at(lambda x: x.ref_space_uid, f, new_uid[None].astype(uid.dtype))
    f = eqx.tree_at(lambda x: x.atom_attention_mask, f, new_mask[None].astype(f.atom_attention_mask.dtype))
    f = eqx.tree_at(lambda x: x.distogram_atom_idx, f, new_dai[None])

    # atom37 slot per atom (binder from templates, target shifted) and the
    # backbone (N,CA,C,O = local 0..3) flat indices — both rebuilt for the tight
    # layout so the atom37 / backbone views track the refreshed geometry.
    new_a37 = jnp.full((A_total,), -1, dtype=atom37_idx.dtype)
    new_a37 = new_a37.at[dest].set(
        tmpl.atom37_slot[ids].reshape(-1).astype(atom37_idx.dtype), mode="drop"
    )
    new_a37 = new_a37.at[tgt_dest].set(atom37_idx[tgt_src], mode="drop")

    bb_binder = offset[None, :] + jnp.arange(4, dtype=backbone_atom_idx.dtype)[:, None]
    bb_target = backbone_atom_idx[:, L_b:] + (sum_k - A_alloc)
    new_bb = jnp.concatenate([bb_binder.astype(backbone_atom_idx.dtype), bb_target], axis=1)
    return f, new_a37, new_bb


# ---------------------------------------------------------------------------
# Target LM pre-encoding (used by ESMFold2.target_only_features /
# ESMFold2.binder_features)
# ---------------------------------------------------------------------------


def precompute_target_lm_hidden(
    esmc: ESMC,
    input_ids: np.ndarray,       # [1, L_total]
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    mol_type: np.ndarray,
    token_attention_mask: np.ndarray,
    *,
    design_positions: np.ndarray,
    unk_input_id: int,
) -> Float[Array, "1 L NL D"]:
    """Run ESMC on the pack and scatter its hiddens into per-token layout.

    ``design_positions`` are the token indices the frozen embedder should read
    as UNK — the *designed* positions. Their `input_ids` are rewritten to
    ``unk_input_id`` before ESMC runs, so ESMC sees "unknown" there while every
    other token keeps its real residue id. An empty ``design_positions`` (the
    real-chain / reprediction case) rewrites nothing.

    The LM is a frozen feature extractor: it never sees the *soft* design
    sequence (it consumes integer `input_ids`), and the returned hiddens are
    stop-gradded, so no gradient reaches the optimized PSSM through the LM.
    """
    design_positions = np.asarray(design_positions, dtype=np.int64).reshape(-1)
    if design_positions.size:
        input_ids = np.array(input_ids, copy=True)
        input_ids[0, design_positions] = unk_input_id

    # Pack scaffolding (includes binder placeholder slots if present).
    lm_input_ids, sequence_id, expand_map = _prepare_lm_inputs(
        input_ids, asym_id, residue_index, mol_type, token_attention_mask,
    )

    lm_in = jnp.asarray(lm_input_ids)
    seqid = jnp.asarray(sequence_id)
    _, hidden = esmc(lm_in, seqid, collect_hidden_states=True)
    # hidden: [NL, B, max_len, D]
    hs_perm = jnp.transpose(hidden, (1, 2, 0, 3))         # [B, max_len, NL, D]
    em = jnp.asarray(expand_map)
    safe_em = jnp.where(em < 0, 0, em)
    gathered = jnp.take_along_axis(
        hs_perm, safe_em[:, :, None, None].astype(jnp.int32), axis=1
    )                                                      # [B, L, NL, D]
    valid = (em >= 0)[..., None, None]
    full_lm = jnp.where(valid, gathered, 0.0)

    return jax.lax.stop_gradient(full_lm)


def recompute_binder_lm_hidden(
    esmc: ESMC,
    esmc_token_of_aa: Int[Array, "20"],
    target_lm_hidden: Float[Array, "1 L NL D"],
    pssm: Float[Array, "N 20"],
) -> Float[Array, "1 L NL D"]:
    """Recompute the binder's ESMC hidden states from ``argmax(pssm)`` and
    splice them into the cached target features.

    Re-encodes **only the binder chain** — ``[BOS, esmc_token_of_aa[argmax], EOS]``
    — rather than the whole pack. This is exact (bit-identical to a full-pack
    re-encode of the binder block) because ESMC's attention is block-diagonal by
    chain and its RoPE is absolute: with the binder contiguous at the front
    (``asym_id == 0``, tokens ``[0, N)``), the binder-alone tokens occupy the
    same absolute positions and attend over the same set as in the full pack, so
    the target chains' hiddens are unchanged and need not be recomputed. The
    binder result is spliced into ``target_lm_hidden[:, :N]``.

    Detached — the LM consumes integer ids and its output is stop-gradded, so no
    gradient reaches the PSSM through the LM (the trunk also stop-gradients it).
    Assumes a fully-designed, contiguous, first binder.
    """
    N = pssm.shape[0]
    ids = jax.lax.stop_gradient(jnp.argmax(pssm, axis=-1))  # [N], mosaic-20
    esmc_ids = esmc_token_of_aa[ids]
    lm_in = jnp.concatenate([
        jnp.array([BOS_ID], dtype=esmc_ids.dtype),
        esmc_ids,
        jnp.array([EOS_ID], dtype=esmc_ids.dtype),
    ])[None]                                                 # [1, N+2]
    # Single chain → no chain-aware masking needed (sequence_id=None).
    _, hs = esmc(lm_in, None, collect_hidden_states=True)    # [NL, 1, N+2, D]
    binder = jnp.transpose(hs[:, :, 1:N + 1], (1, 2, 0, 3))  # [1, N, NL, D] (drop BOS/EOS)
    return target_lm_hidden.at[:, :N].set(binder)
