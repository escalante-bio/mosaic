"""ESMFold2 loss + soft-sequence plumbing.

ESMC is a frozen feature extractor (upstream's design API detaches its hidden
states before the trunk): we run it once at feature-build time on the
placeholder+target pack with binder slots rewritten to UNK, and cache the
per-token hiddens on the pack. The soft binder PSSM never reaches ESMC; its
gradients flow only through the trunk's `res_type` / MSA-query-row inputs
(patched by `set_binder_sequence`). The binder lives at `asym_id == 0`,
token positions `[0, binder_length)`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from esmjfold2 import Features
from esmjfold2.lm_features import _prepare_lm_inputs
from esmjfold2.model import ESMFold2 as _JaxESMFold2Release
from esmjfold2.experimental import ESMFold2Experimental as _JaxESMFold2Experimental
from esmjfold2.esmc import ESMC

from mosaic.common import LossTerm, LinearCombination
from mosaic.losses.structure_prediction import StructureModelOutput, PAE_BINS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37

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

    Convention (shared with every other mosaic backend): the binder, when
    present, lives at token positions `[0, pssm.shape[0])` and asym_id 0.
    The loss reads `pssm.shape[0]` to know the binder length.

    Note: this carries only what the (jitted) forward consumes. Output-writer
    metadata (`chain_infos`) is deliberately *not* here — it's returned
    separately by `binder_features` / `target_only_features` and passed to
    `predict(writer=...)`. Keeping it off the pack avoids it becoming a static
    JIT cache key (it differs per design → would force a recompile every call).
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
    stop_recycling_grad: bool = False,
):
    """Returns the per-token pair representation `z` + the embedding context.

    Heads (diffusion sampler, confidence, distogram) consume both.

    `stop_recycling_grad` (default False) selects the recycling gradient
    policy. False backprops through every recycling iteration (the upstream
    `esmf._run_trunk` behaviour). True runs the first `num_loops` iterations as
    a detached burn-in and differentiates only the final trunk pass — the
    AF-style policy used by the protenix / boltz2 / af2 backends. See
    `_run_trunk_stopgrad` (single-sequence experimental path only).
    """
    ctx = esmf._prepare_embeddings(features)
    # The LM is a frozen feature extractor — stop-gradient at the consumption
    # site so no gradient reaches the design PSSM through the LM, regardless of
    # how `lm_hidden_states` was produced. Mirrors the reference ESMFold2's
    # `.detach()` on its LM hidden states. (The cached `target_lm_hidden` is
    # already stop-gradded upstream; this guarantees the contract.)
    lm_z = (
        esmf.language_model(jax.lax.stop_gradient(lm_hidden_states))
        if lm_hidden_states is not None
        else None
    )
    ktrunk, _ = jax.random.split(key)
    if stop_recycling_grad:
        z = _run_trunk_stopgrad(esmf, ctx, lm_z, ktrunk, num_loops=num_loops)
    else:
        z = esmf._run_trunk(
            ctx, lm_z, ktrunk, num_loops=num_loops, msa_max_depth=msa_max_depth
        )
    return ctx, z


def _run_trunk_stopgrad(esmf: ESMFold2, ctx, lm_z, key, *, num_loops: int):
    """Experimental-trunk recycling with AF-style gradient truncation.

    Reuses the upstream `esmf._run_trunk` for the detached burn-in (the first
    `num_loops` iterations, wrapped in `stop_gradient`) and re-derives only the
    single final trunk step so it can be differentiated. The design inputs (via
    `ctx.z_init`) therefore receive gradient through the last pass only —
    matching the protenix / boltz2 / af2 backends, versus the default which
    backprops through every iteration.

    Why not just call `_run_trunk` twice (once stop-gradded, once not)? It
    always re-initialises the recycling state to zeros and returns only the
    final `z`, so a second call can't continue from the burn-in state — it
    would recompute iteration 1 from scratch. And the last-step-only gradient
    can't be reconstructed from a full-unroll gradient plus a stop-gradded
    call. So the final single step must be re-derived here. That step is
    exactly `folding_trunk(z_init + pair_loop_proj(z))` for the single-sequence
    config used in design (`msa_encoder is None`, `lm_dropout == 0`), which is
    bit-identical to `_run_trunk`'s body; we raise otherwise rather than
    silently diverge.
    """
    if esmf.msa_encoder is not None:
        raise NotImplementedError(
            "stop_recycling_grad is only implemented for the single-sequence "
            "(no-MSA) experimental trunk used in design."
        )
    if float(getattr(esmf, "lm_dropout", 0.0)) > 0.0:
        raise NotImplementedError(
            "stop_recycling_grad assumes lm_dropout == 0 (the design setting)."
        )
    if num_loops <= 0:
        # Single iteration, no burn-in to detach — differentiate it directly.
        return esmf._run_trunk(ctx, lm_z, key, num_loops=num_loops)
    # Detached burn-in: the real recycling loop for the first total_steps-1
    # iterations (num_loops-1 → num_loops steps), cut from autodiff.
    z_burn = jax.lax.stop_gradient(
        esmf._run_trunk(ctx, lm_z, key, num_loops=num_loops - 1)
    )
    # Final iteration — differentiated. One trunk step, identical to the
    # upstream body for this config.
    z_init = ctx.z_init + lm_z if lm_z is not None else ctx.z_init
    z = z_init + esmf.pair_loop_proj(z_burn)
    return esmf.folding_trunk(z, pair_attention_mask=ctx.pair_mask)


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
    uses (2.0, 52.0, 128) — verified empirically on 1UBQ.
    """
    boundaries = jnp.linspace(min_dist, max_dist, n_bins + 1, dtype=jnp.float32)
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
    stop_recycling_grad: bool = False,
) -> StructureModelOutput:
    """One end-to-end forward. `pssm=None` → target-only prediction; otherwise
    the PSSM is spliced into the binder slots. The cached UNK LM features in
    `pack.target_lm_hidden` are reused unchanged (see module docstring).

    `stop_recycling_grad` is forwarded to `esmfold2_trunk` (recycling gradient
    policy; see there). Irrelevant for forward-only prediction.
    """
    features = pack.features
    L_total = features.res_type.shape[1]

    if pssm is not None:
        features = set_binder_sequence(pssm, features, res_type_perm)

    k_trunk, k_heads = jax.random.split(key)
    ctx, z = esmfold2_trunk(
        esmf, features, pack.target_lm_hidden, k_trunk,
        num_loops=num_loops, msa_max_depth=msa_max_depth,
        stop_recycling_grad=stop_recycling_grad,
    )
    sample_atom_coords, distogram_logits, confidence = (
        esmfold2_forward_from_trunk(
            esmf, ctx, z, k_heads,
            num_sampling_steps=num_sampling_steps,
        )
    )

    full_seq = jnp.zeros((L_total, 20), dtype=jnp.float32)
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
        atom37_idx=pack.atom37_idx,
        backbone_atom_idx=pack.backbone_atom_idx,
    )


class ESMFold2Loss(LossTerm):
    """Mosaic-style loss wrapping an ESMFold2 forward.

    Binder LM slots stay frozen at the cached UNK features. We tried
    re-encoding the LM during design (argmax / soft, ± stop-gradient) and
    every variant hurt iPTM monotonically (0.92 → ~0.48 worst case): the
    model expects UNK at unknown positions, and APGM does best when the
    binder LM features are constant across steps.
    """

    esmf: ESMFold2
    pack: EsmFold2FeaturePack
    loss: LossTerm | LinearCombination
    res_type_perm: Float[Array, "20 33"]
    distogram_bins: Float[Array, "Bins"]
    num_loops: int = eqx.field(static=True)
    num_sampling_steps: int = eqx.field(static=True)
    msa_max_depth: int | None = eqx.field(static=True)
    stop_recycling_grad: bool = eqx.field(static=True, default=False)

    def __call__(self, sequence: Float[Array, "N 20"], *, key):
        smo = forward_with_pssm(
            self.esmf, self.pack, sequence,
            self.res_type_perm,
            self.distogram_bins,
            key=key,
            num_loops=self.num_loops,
            num_sampling_steps=self.num_sampling_steps,
            msa_max_depth=self.msa_max_depth,
            stop_recycling_grad=self.stop_recycling_grad,
        )
        return self.loss(sequence, smo, key=key)


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
    sequence, and the returned hiddens are stop-gradded, so no gradient reaches
    the optimized PSSM through the LM. This matches the reference ESMFold2,
    which feeds the LM integer `input_ids` (not the soft sequence) and detaches
    its output.
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
