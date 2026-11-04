"""ESMFold2 backend for mosaic structure prediction.

Factories come in two families, each in a `Fast` (no MSA encoder, default for
design) and `Full` (MSA-conditioned, for target prediction) variant:

- Release: `ESMFold2Fast()`, `ESMFold2Full()`.
- Experimental (AF3-style: LM added once to `z_init` outside the trunk loop,
  slimmer confidence head): `ESMFold2ExperimentalFast()`,
  `ESMFold2ExperimentalFull()`, plus `...2025()` variants on the 2025-cutoff
  checkpoints and the `base{300M,600M,6B}` training-step sweep via
  `ESMFold2ExperimentalFastBase(size, steps)` / `...Fast300M()` / `...Fast600M()`
  / `...Fast6B()` — lighter design models with non-functional confidence heads
  (validate with a full model).

`recycling_steps` counts trunk passes (upstream's `num_loops + 1`); `predict` uses
the torch `processor.fold()` count. Design gradients flow through every trunk pass.
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import hashlib

import equinox as eqx
import esmjfold2
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from esmjfold2 import Features, TrunkInputs
from esmjfold2.diffusion import DiffusionCache, DiffusionModule, DiffusionStructureHead
from esmjfold2.esmc import ESMC
from esmjfold2.experimental import ESMFold2Experimental as JaxESMFold2Experimental
from esmjfold2.lm_features import BOS_ID, EOS_ID, _prepare_lm_inputs
from esmjfold2.model import ESMFold2 as JaxESMFold2Release
from esmjfold2.structure_output import output_to_mmcif
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput
from mosaic.sampling import Denoise, Sampler, schedules
from mosaic.sampling.core import masked_centering, rigid_align
from mosaic.sampling.schedules import Schedule
from mosaic.structure_prediction import (
    Atom37,
    Confidence,
    DiffusionStructurePredictionModel,
    Distogram,
    PolymerType,
    TargetChain,
    TokenInfo,
    Trunk,
    TrunkState,
)


def _denoise(diffusion_module: DiffusionModule, cache: DiffusionCache, x, sigma):
    return diffusion_module.denoise(
        cache, x[None], jnp.full((1,), sigma, dtype=jnp.float32)
    )[0]


def esmfold2_denoiser(
    structure_head: DiffusionStructureHead, ctx: eqx.Module, z
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk output (`ESMFold2.embed` and `ESMFold2.step`).

    Returns the denoiser and the atom mask.
    """
    module = structure_head.diffusion_module
    # Step-invariant conditioning, computed once.
    cache = module.prepare(
        ref_pos=ctx.ref_pos,
        ref_charge=ctx.ref_charge,
        ref_mask=ctx.atm_mask,
        ref_element_oh=ctx.ref_element_oh,
        ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh,
        ref_space_uid=ctx.ref_space_uid,
        tok_idx=ctx.atom_to_token,
        s_inputs=ctx.x_inputs,
        z_trunk=z,
        relative_position_encoding=ctx.relative_position_encoding,
        n_tokens=ctx.n_tokens,
        token_attention_mask=ctx.tok_mask,
    )
    denoise = eqx.Partial(_denoise, module, cache)
    return denoise, ctx.atm_mask[0].astype(jnp.float32)


def esmfold2_sampler(
    structure_head: DiffusionStructureHead,
    num_steps: int,
    *,
    max_sigma: float | None = 256.0,
    schedule: schedules.Schedule | None = None,
) -> Sampler:
    """The `Sampler` reproducing `DiffusionStructureHead.sample`.

    The default `max_sigma` cap drops the high-noise steps, so fewer than
    `num_steps` denoiser calls run (e.g. 34 for 50); how many survive depends on
    `schedule`, which defaults to the model's own `Karras`.
    """
    h = structure_head
    if schedule is None:
        schedule = schedules.Karras(rho=h.inference_p)
    sigmas = schedules.append_zero(
        schedule(
            np.linspace(0.0, 1.0, num_steps),
            sigma_data=h.sigma_data,
            s_max=h.inference_s_max,
            s_min=h.inference_s_min,
        )
    )
    if max_sigma is not None:
        sigmas = schedules.cap(sigmas, max_sigma)
    return Sampler(
        sigmas=sigmas,
        gamma_0=h.gamma_0,
        gamma_min=h.gamma_min,
        noise_scale=h.noise_scale,
        step_scale=h.step_scale,
        center=masked_centering,
        align=rigid_align,
    )

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
    atom37_idx: Int[Array, " A"]
    # Residue templates for rebuilding the binder's atoms and ESMC features from
    # argmax(PSSM) each step (design packs); None for a native pack.
    geom_templates: ResidueAtomTemplates | None = None

    @property
    def refresh(self) -> bool:
        return self.geom_templates is not None


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


def distogram_bin_centers(
    n_bins: int, min_dist: float, max_dist: float,
) -> Float[Array, " Bins"]:
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


# ---------------------------------------------------------------------------
# Atom37 / backbone scaffolding (numpy-side, run once per feature build)
# ---------------------------------------------------------------------------


def _decode_atom_name(chars_row: np.ndarray) -> str:
    """ESMFold2 stores atom names as 4 small ints (`ord(c) - 32`, 0 = pad)."""
    return "".join(chr(int(c) + 32) for c in chars_row if int(c) != 0).strip()


def build_atom37_index(
    ref_atom_name_chars: np.ndarray,  # [B, A, 4]
    atom_attention_mask: np.ndarray,   # [B, A] bool
) -> np.ndarray:
    """`[A] int32` atom37 slot (0..36) of each atom, or -1 (no slot / padding)."""
    A = ref_atom_name_chars.shape[1]
    atom37_idx = np.full((A,), -1, dtype=np.int32)
    chars = ref_atom_name_chars[0]
    mask = atom_attention_mask[0]
    for a in range(A):
        if bool(mask[a]):
            atom37_idx[a] = ATOM37_INDEX.get(_decode_atom_name(chars[a]), -1)
    return atom37_idx


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


@functools.cache
def _residue_atom_templates(
    max_atoms: int = MAX_ATOMS_PER_RES,
) -> ResidueAtomTemplates:
    """Per-residue atom14 reference templates for the in-loop geometry refresh.

    Built once on the host (CPU, no model weights) and cached process-wide:
    featurizes each amino acid as an isolated single-residue chain and reads its
    atom block. ``ref_pos`` is context- and checkpoint-independent, so the result
    is a singleton — `ESMFold2.binder_features` attaches it to a design pack, so
    callers never build it themselves.
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
        # The repack assumes each residue starts N, CA, C, O.
        bb_decoded = tuple(_decode_atom_name(names[j]) for j in range(min(4, k)))
        assert bb_decoded == ("N", "CA", "C", "O"), (
            f"{aa}: expected N,CA,C,O as the first 4 atoms, got {bb_decoded}"
        )
        # atom37 slot per local atom, decoded as in `build_atom37_index`.
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
    atom37_idx: Int[Array, " A"],
) -> tuple[Features, Int[Array, " A"]]:
    """Rebuild the atom axis with the argmax(pssm) binder residues' atoms
    **tightly** packed at the front (native order), the target atoms shifted to
    sit immediately after, and zero-padding at the end.

    Returns the refreshed ``features`` and the ``atom37_idx`` matching the new
    atom order.

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

    # atom37 slot per atom: the binder's from the templates, the target's shifted.
    new_a37 = jnp.full((A_total,), -1, dtype=atom37_idx.dtype)
    new_a37 = new_a37.at[dest].set(
        tmpl.atom37_slot[ids].reshape(-1).astype(atom37_idx.dtype), mode="drop"
    )
    new_a37 = new_a37.at[tgt_dest].set(atom37_idx[tgt_src], mode="drop")
    return f, new_a37


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

JaxESMFold2 = JaxESMFold2Release | JaxESMFold2Experimental


_DESIGN_RECYCLING_STEPS = 4
# `predict`: the torch `processor.fold()` defaults (20 recycles, 0.3 LM dropout).
_INFERENCE_RECYCLING_STEPS = 21
_UPSTREAM_SAMPLING_STEPS = 200
_LM_DROPOUT = 0.3
_DEFAULT_MSA_SERVER = "https://api.colabfold.com"


def _fetch_msa(sequence: str, max_sequences: int = 8192):
    """Query the ColabFold MSA server (via boltz's MMseqs2 client) for an a3m
    and return an `esm.utils.msa.MSA`. Cached by sequence hash under
    `MOSAIC_CACHE_DIR/msa` so repeated featurizations skip the network.
    """
    from boltz.data.msa.mmseqs2 import run_mmseqs2
    from esm.utils.msa import MSA

    msa_cache = cache_dir() / "msa"
    msa_cache.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    a3m_path = msa_cache / f"{seq_hash}.a3m"

    if not a3m_path.exists():
        tmp_prefix = msa_cache / seq_hash
        result = run_mmseqs2(
            [sequence],
            prefix=str(tmp_prefix),
            use_env=True,
            use_filter=True,
            use_pairing=False,
            host_url=_DEFAULT_MSA_SERVER,
        )
        # run_mmseqs2 returns either list[str] or tuple of lists depending on
        # version; unwrap defensively.
        a3m_strings = result[0] if isinstance(result, tuple) else result
        a3m_path.write_text(a3m_strings[0])

    return MSA.from_a3m(
        path=str(a3m_path), remove_insertions=True, max_sequences=max_sequences
    )


def _load_pretrained(
    checkpoint: str,
    *,
    experimental: bool = False,
    dtype=None,
) -> tuple[JaxESMFold2, ESMC]:
    """Load ESMFold2 + ESMC from a HuggingFace checkpoint and convert to JAX.

    Loads the native Biohub Torch model on CPU before converting to JAX.
    ESMC loads directly in fp32 for both released and experimental checkpoints.
    """
    import torch
    from esm.models.esmfold2 import EsmFold2ExperimentalModel, EsmFold2Model

    model_class = EsmFold2ExperimentalModel if experimental else EsmFold2Model
    torch_model = model_class.from_pretrained(
        checkpoint,
        cache_dir=cache_dir() / "huggingface",
        device="cpu",
        load_esmc=True,
        dtype=dtype or torch.float32,
        esmc_precision="fp32",
    ).eval()
    eqx_esmc = esmjfold2.from_torch(torch_model.esmc)
    eqx_trunk = esmjfold2.from_torch(torch_model)

    del torch_model
    gc.collect()
    return eqx_trunk, eqx_esmc


def _featurize(
    chains: list[TargetChain],
    *,
    design_char: str | None = None,
    design_geometry: str = "G",
):
    """Build esmjfold2 features + chain_infos + design positions via the builder.

    Each chain is a real protein chain — except any residue equal to
    ``design_char`` marks a *designed* position. Designed residues are handed to
    the builder as ``design_geometry`` ("A" → N/CA/C/O/CB; "G" → N/CA/C/O) so the
    trunk gets a clean placeholder backbone there, rather than the degenerate
    atom-tokenized geometry of a literal "X". Every *non*-designed residue keeps
    its real identity, so it gets correct full-atom geometry and real res_type.

    Returns ``(raw_features, chain_infos, design_positions)`` where
    ``design_positions`` is the token-index array of the designed residues (used
    by the caller to blank them to UNK for the LM and to inject/score the design
    PSSM). ``design_char=None`` is the pure real-chain case → no design positions.
    Token indices assume one token per (standard/placeholder) residue, chains
    concatenated in order — the codebase-wide convention.
    """
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    proteins: list = []
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    design_positions: list[int] = []
    offset = 0
    for i, c in enumerate(chains):
        if c.polymer_type != PolymerType.PROTEIN:
            raise NotImplementedError(
                "ESMFold2 wrapper currently supports protein chains only; got "
                f"{c.polymer_type}"
            )
        if c.template_chain is not None:
            raise NotImplementedError(
                "ESMFold2 wrapper does not support template chains."
            )
        seq = c.sequence
        if design_char is not None and design_char in seq:
            design_positions.extend(
                offset + j for j, ch in enumerate(seq) if ch == design_char
            )
            # Hand the builder a clean placeholder backbone at designed
            # positions (the trunk reads geometry here; res_type is overwritten
            # by the PSSM and input_ids is blanked to UNK downstream).
            seq = seq.replace(design_char, design_geometry)
        # The Fast checkpoint has no MSA encoder but still uses query row 0 as
        # the `profile` aggregate, so passing the MSA through is correct in
        # both modes.
        msa = _fetch_msa(c.sequence) if c.use_msa else None
        proteins.append(
            ProteinInput(id=chain_letters[i], sequence=seq, msa=msa)
        )
        offset += len(c.sequence)

    spi = StructurePredictionInput(sequences=proteins)
    builder = ESMFold2InputBuilder()
    raw_features, chain_infos = builder.prepare_input(spi)

    # Note: we intentionally do NOT re-stamp the binder `token_bonds` block.
    # A standard-residue placeholder ("A"/"G") leaves it empty (chain
    # connectivity is implicit via residue_index, exactly as for a real chain
    # and as the target_only reprediction sees it).

    return raw_features, chain_infos, np.array(design_positions, dtype=np.int64)


def _build_pack(
    esmc: ESMC,
    raw: dict,
    *,
    design_positions: np.ndarray,
    unk_input_id: int,
) -> EsmFold2FeaturePack:
    features = Features.from_input_builder(raw)
    target_lm = precompute_target_lm_hidden(
        esmc,
        input_ids=np.asarray(raw["input_ids"]),
        asym_id=np.asarray(raw["asym_id"]),
        residue_index=np.asarray(raw["residue_index"]),
        mol_type=np.asarray(raw["mol_type"]),
        token_attention_mask=np.asarray(raw["token_attention_mask"]),
        design_positions=design_positions,
        unk_input_id=unk_input_id,
    )
    atom37_idx = build_atom37_index(
        ref_atom_name_chars=np.asarray(raw["ref_atom_name_chars"]),
        atom_attention_mask=np.asarray(raw["atom_attention_mask"]),
    )
    return EsmFold2FeaturePack(
        features=features,
        target_lm_hidden=target_lm,
        atom37_idx=jnp.asarray(atom37_idx, dtype=jnp.int32),
    )


class ESMFold2(DiffusionStructurePredictionModel[EsmFold2FeaturePack, TrunkInputs]):
    """ESMFold2 wrapped behind mosaic's structure-prediction interface.

    Public construction: use the `ESMFold2Fast()` / `ESMFold2Full()` factories
    rather than `__init__` directly — those handle checkpoint download +
    alphabet probing.

    The heads read `esmf.readout` of the trunk pair. To share one ESMC across
    several losses, set `esmc=None` and splice it in at call time (see
    `StackedEsmFold2PLL`).
    """

    esmf: JaxESMFold2
    esmc: ESMC | None
    res_type_perm: Float[Array, "20 33"]
    distogram_bins: Float[Array, " Bins"]
    # ESMC input-id for each mosaic-20 residue (LM re-encoding).
    esmc_token_of_aa: Int[Array, "20"]
    unk_input_id: int = eqx.field(static=True)
    # MSA rows sampled per trunk pass (the query is always kept).
    msa_max_depth: int | None = eqx.field(static=True, default=1024)
    # Dropout on the language-model pair features; 0 keeps the checkpoint's own.
    lm_dropout: float = eqx.field(static=True, default=_LM_DROPOUT)
    design_recycling_steps: int = eqx.field(static=True, default=_DESIGN_RECYCLING_STEPS)
    inference_recycling_steps: int = eqx.field(static=True, default=_INFERENCE_RECYCLING_STEPS)
    backprop_all_passes: bool = eqx.field(static=True, default=True)

    def _check_msa_compat(self, chains: list[TargetChain]) -> None:
        # The Fast checkpoint has no MSA encoder; an MSA would put its
        # `profile` aggregate off-distribution, so refuse rather than
        # silently produce bad predictions.
        if self.esmf.msa_encoder is None and any(c.use_msa for c in chains):
            raise ValueError(
                "ESMFold2-Fast does not support MSAs (model.esmf.msa_encoder is "
                "None). Set use_msa=False on all chains, or use ESMFold2Full()."
            )

    def target_only_features(
        self,
        chains: list[TargetChain],
        *,
        design_char: str | None = None,
        design_geometry: str = "G",
    ):
        """Featurize a list of real protein chains.

        With ``design_char`` set, any residue equal to it (in any chain) is a
        *designed* position: the trunk gets a ``design_geometry`` ("G"/"A")
        backbone there and the frozen ESMC sees UNK; everything else is the real
        residue on every channel (full-atom geometry, real res_type, real LM
        token). This is the whole design contract — a binder is just a chain
        with designed positions in it.

        Returns ``(pack, chain_infos)``. The designed token positions (where the
        frozen ESMC sees UNK) are wherever ``design_char`` appears in the input
        sequences — the caller put them there, so it can recompute them.
        """
        self._check_msa_compat(chains)
        raw, chain_infos, design_positions = _featurize(
            chains, design_char=design_char, design_geometry=design_geometry,
        )
        pack = _build_pack(
            self.esmc, raw,
            design_positions=design_positions,
            unk_input_id=self.unk_input_id,
        )
        return pack, chain_infos

    def binder_features(
        self,
        binder_length: int,
        chains: list[TargetChain],
    ):
        """Design-time pack for a fully-designed binder (chain 0).

        Thin wrapper over ``target_only_features``: prepends an all-``X`` binder
        chain (UNK to the frozen ESMC — the design contract) at token positions
        ``[0, binder_length)``.

        The returned pack has ``refresh`` set: the forward rebuilds the binder
        geometry + ESMC features in-loop from ``argmax(PSSM)``. That geometry
        refresh tight-packs each residue's atoms, so the binder is allocated at
        the 14-atom maximum (``"W"``); the placeholder geometry is overwritten
        before the trunk ever sees it, so its identity is immaterial and not a
        caller choice.

        For a partially-fixed binder (e.g. a VHH with a fixed framework) still
        use this function: make the whole binder a designed (all-``X``) chain and
        pin the framework with ``SetPositions`` during optimization. The in-loop
        geometry refresh means there's no need to bake the real framework into a
        ``target_only_features`` template, unlike the other backends.
        """
        pack, chain_infos = self.target_only_features(
            [TargetChain("X" * binder_length, use_msa=False), *chains],
            # "W" = the 14-atom (max) allocation the in-loop geom refresh needs.
            design_char="X", design_geometry="W",
        )
        # Tie geom + LM refresh on for the design pack.
        return dataclasses.replace(pack, geom_templates=_residue_atom_templates()), chain_infos

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        if num_steps is None:
            num_steps = _UPSTREAM_SAMPLING_STEPS
        return esmfold2_sampler(self.esmf.structure_head, num_steps, schedule=schedule)

    # -- the structure-model contract -----------------------------------------

    def set_sequence(
        self,
        features: EsmFold2FeaturePack,
        sequence: Float[Array, "L 20"],
        *,
        key: PRNGKeyArray,
    ) -> EsmFold2FeaturePack:
        """Set the binder's `res_type` and MSA query row; a design pack also
        rebuilds the binder's atoms and ESMC features from `argmax(sequence)`."""
        del key
        pack = features
        feats = set_binder_sequence(sequence, pack.features, self.res_type_perm)
        atom37_idx, lm_hidden = pack.atom37_idx, pack.target_lm_hidden
        if pack.geom_templates is not None:
            feats, atom37_idx = refresh_binder_geometry(
                feats, sequence, pack.geom_templates, atom37_idx
            )
            lm_hidden = recompute_binder_lm_hidden(
                self.esmc, self.esmc_token_of_aa, lm_hidden, sequence
            )
        return eqx.tree_at(
            lambda p: (p.features, p.target_lm_hidden, p.atom37_idx),
            pack,
            (feats, lm_hidden, atom37_idx),
        )

    def embed(self, features: EsmFold2FeaturePack, *, key: PRNGKeyArray) -> TrunkInputs:
        # The language model is frozen: no gradient flows through it.
        lm_hidden = jax.lax.stop_gradient(features.target_lm_hidden).astype(jnp.float32)
        esmf = _with_lm_dropout(self.esmf, self.lm_dropout)
        return esmf.embed(features.features, esmf.language_model(lm_hidden), key=key)

    def init_state(
        self, features: EsmFold2FeaturePack, inputs: TrunkInputs, *, key: PRNGKeyArray
    ) -> TrunkState:
        del features
        return TrunkState(pair=self.esmf.init_state(inputs, key=key)[0])

    def step(
        self,
        features: EsmFold2FeaturePack,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray,
    ) -> TrunkState:
        del features
        esmf = _with_lm_dropout(self.esmf, self.lm_dropout)
        pair = esmf.step(inputs, state.pair[None], key=key, msa_max_depth=self.msa_max_depth)
        return TrunkState(pair=pair[0])

    def structure_denoiser(self, trunk: Trunk) -> tuple[Denoise, Float[Array, " A"]]:
        z = self.esmf.readout(trunk.inputs, trunk.state.pair[None])
        return esmfold2_denoiser(self.esmf.structure_head, trunk.inputs.context, z)

    def distogram(self, trunk: Trunk) -> Distogram:
        z = self.esmf.readout(trunk.inputs, trunk.state.pair[None])
        return Distogram(logits=self.esmf._compute_distogram(z)[0], bins=self.distogram_bins)

    def confidence(self, trunk: Trunk, coords: Float[Array, "A 3"]) -> Confidence:
        z = self.esmf.readout(trunk.inputs, trunk.state.pair[None])
        out = self.esmf._compute_confidence(trunk.inputs.context, z, coords[None])
        return Confidence(
            plddt=out["plddt"][0],
            pae_logits=out["pae_logits"][0],
            pae_bins=jnp.array(PAE_BINS, dtype=jnp.float32),
        )

    def tokens(self, features: EsmFold2FeaturePack) -> TokenInfo:
        feats = features.features
        res_type = feats.res_type[0]
        if res_type.ndim == 1:
            res_type = jax.nn.one_hot(res_type, NUM_RES_TYPES)
        return TokenInfo(
            sequence=res_type @ self.res_type_perm.T,
            asym_id=feats.asym_id[0].astype(jnp.int32),
            residue_index=feats.residue_index[0].astype(jnp.int32),
            mask=feats.token_attention_mask[0].astype(bool),
        )

    def atom37(self, features: EsmFold2FeaturePack, coords: Float[Array, "A 3"]) -> Atom37:
        feats = features.features
        atom37_coords, atom37_mask = scatter_atom37(
            atom_coords=coords,
            atom_to_token=feats.atom_to_token[0].astype(jnp.int32),
            atom37_idx=features.atom37_idx,
            n_token=feats.res_type.shape[1],
        )
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(
        self,
        features: EsmFold2FeaturePack,
        writer,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """A design pack is written from the atom37 view; a native pack from its
        own atoms with `writer`."""
        if features.refresh:
            if sequence is None:
                raise ValueError(
                    "predict() on a refresh (design) pack needs a PSSM; "
                    "PSSM=None leaves the placeholder binder geometry unrefreshed. "
                    "Use a native target_only_features pack for target-only folds."
                )
            return output.to_structure()
        if writer is None:
            raise ValueError(
                "predict() needs `writer` (the chain_infos from "
                "target_only_features / binder_features) to write the native "
                "structure; it's only optional for a refresh (design) pack."
            )
        feats = features.features
        mmcif_str = output_to_mmcif(
            coords=np.asarray(output.structure_coordinates),
            plddt=np.asarray(output.plddt),
            atom_mask=np.asarray(feats.atom_attention_mask[0]),
            ref_element=np.asarray(feats.ref_element[0]),
            ref_atom_name_chars=np.asarray(feats.ref_atom_name_chars[0]),
            chain_infos=writer,
        )
        st = gemmi.make_structure_from_block(gemmi.cif.read_string(mmcif_str).sole_block())
        if sequence is not None:
            # `chain_infos` names the placeholder residues: relabel the binder.
            binder_names = [
                gemmi.expand_one_letter(TOKENS[int(k)], gemmi.ResidueKind.AA)
                for k in np.argmax(np.asarray(sequence), axis=-1)
            ]
            for res, name in zip(st[0][0], binder_names):
                res.name = name
        return st


def _probe_alphabets() -> tuple[np.ndarray, int, np.ndarray]:
    """Build the mosaic-20 → res_type-33 permutation matrix, UNK input id, and
    the ESMC input-id per mosaic-20 residue.

    The perm is read off `res_type` for a chain of the 20 canonical AAs in
    `TOKENS` order; the UNK id from a one-residue `X` chain; the ESMC ids from
    `input_ids` of that same 20-AA chain. The char→int mapping is
    checkpoint-specific, so probe it via the input builder.
    """
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    builder = ESMFold2InputBuilder()

    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=TOKENS)]
    )
    feats, _ = builder.prepare_input(spi)
    res_type = np.asarray(feats["res_type"])[0]
    assert res_type.shape == (len(TOKENS),), (
        f"expected res_type shape ({len(TOKENS)},), got {res_type.shape} "
        "— the input builder may have inserted extra tokens that break the "
        "1:1 mapping assumption"
    )
    res_type_perm = np.zeros((len(TOKENS), NUM_RES_TYPES), dtype=np.float32)
    for i in range(len(TOKENS)):
        res_type_perm[i, int(res_type[i])] = 1.0

    # ESMC input-id per residue (same 20-AA chain, in TOKENS order).
    esmc_token_of_aa = np.asarray(feats["input_ids"])[0].astype(np.int32)
    assert esmc_token_of_aa.shape == (len(TOKENS),)

    spi_unk = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="X")]
    )
    feats_unk, _ = builder.prepare_input(spi_unk)
    unk_input_id = int(np.asarray(feats_unk["input_ids"])[0, 0])

    return res_type_perm, unk_input_id, esmc_token_of_aa


def _staticize_scalar_config(tree):
    """Coerce 0-d array leaves (jax or numpy) to python scalars.

    esmjfold2 / from_torch store configuration as 0-d arrays: TriangleMultiplication
    `flow` ("incoming"/"outgoing" — a string array), ESMC `n_heads` / `d_head`, the
    diffusion noise-schedule sigmas, layernorm eps, etc. `eqx.is_array` reports these
    as arrays, so `eqx.filter_jit` (the optimizer, `predict`) traces them as dynamic.
    That breaks string leaves outright and any scalar used as a *static* value —
    reshape dims (`n_heads`) and the numpy-built diffusion schedule both require
    concrete Python ints/floats. The model is frozen during design (gradients flow to
    the sequence, never the model), so making these scalars static is safe and is what
    the library's own code already assumes. Multi-element arrays (weights) are
    untouched.
    """
    def conv(x):
        if isinstance(x, (np.ndarray, jax.Array)) and getattr(x, "ndim", 1) == 0:
            return x.item()
        return x

    return jax.tree_util.tree_map(conv, tree)


def _with_lm_dropout(esmf: JaxESMFold2, lm_dropout: float) -> JaxESMFold2:
    """Set LM dropout as torch's `_lm_dropout_context` does; 0 is a no-op."""
    if not lm_dropout:
        return esmf
    if isinstance(esmf, JaxESMFold2Release):
        return dataclasses.replace(esmf, lm_dropout=lm_dropout, per_loop_lm_dropout=True)
    return dataclasses.replace(esmf, lm_dropout=lm_dropout)


def _make(checkpoint: str, *, experimental: bool = False) -> ESMFold2:
    esmf, esmc = _load_pretrained(checkpoint, experimental=experimental)
    esmf = _staticize_scalar_config(esmf)
    esmc = _staticize_scalar_config(esmc)
    res_type_perm, unk_input_id, esmc_token_of_aa = _probe_alphabets()
    # Distogram range (release 2.0–22.0 Å; experimental 2.0–52.0 Å, matching
    # its confidence head).
    min_dist, max_dist = (2.0, 52.0) if experimental else (2.0, 22.0)
    n_bins = int(esmf.distogram_head.weight.shape[0])
    return ESMFold2(
        esmf=esmf,
        esmc=esmc,
        res_type_perm=jnp.asarray(res_type_perm),
        distogram_bins=distogram_bin_centers(n_bins, min_dist, max_dist),
        esmc_token_of_aa=jnp.asarray(esmc_token_of_aa),
        unk_input_id=unk_input_id,
    )


def ESMFold2Fast() -> ESMFold2:
    """Single-sequence ESMFold2 (no MSA encoder). Default for binder design.

    Recycling (`recycling_steps`) and the diffusion `sampler` are per-call
    knobs; see the module docstring for their defaults.
    """
    return _make("biohub/ESMFold2-Fast")


def ESMFold2Full() -> ESMFold2:
    """Full ESMFold2 with MSA encoder. Heavier; intended for target structure
    prediction with real MSAs.
    """
    return _make("biohub/ESMFold2")


def ESMFold2ExperimentalFast() -> ESMFold2:
    """Single-sequence ESMFold2-Experimental-Fast.

    Closer to a standard AF3-style model than the release ``Fast``: no parcae
    refinement state, LM contribution added once to ``z_init`` outside the
    loop, slimmer confidence head (no PDE / no resolved). No MSA encoder.
    """
    return _make("biohub/ESMFold2-Experimental-Fast", experimental=True)

def ESMFold2ExperimentalFast2025() -> ESMFold2:
    """Single-sequence ESMFold2-Experimental-Fast.

    Closer to a standard AF3-style model than the release ``Fast``: no parcae
    refinement state, LM contribution added once to ``z_init`` outside the
    loop, slimmer confidence head (no PDE / no resolved). No MSA encoder.
    """
    return _make("biohub/ESMFold2-Experimental-Fast-Cutoff2025", experimental=True)


def ESMFold2ExperimentalFastBase(
    size: str = "600M", steps: str = "1500k"
) -> ESMFold2:
    """ESMFold2-Experimental-Fast ``base{size}`` training-step sweep.

    ``size`` ∈ {``"300M"``, ``"600M"``, ``"6B"``}; ``steps`` selects the
    training-step checkpoint tag ∈ {``"250k"``, ``"500k"``, ``"750k"``,
    ``"1000k"``, ``"1500k"``}. The 300M / 600M variants are lighter single-
    sequence design models — small enough to sit alongside a second structure
    model (e.g. ``ProtenixV2``) in one kernel without OOM.

    Note: these checkpoints' confidence heads are non-functional
    (``model_output`` returns pLDDT / PAE as 0), so re-fold designs with a full
    model such as ``ESMFold2ExperimentalFast2025`` for held-out iPTM validation.
    """
    return _make(
        f"biohub/ESMFold2-Experimental-Fast-base{size}-step{steps}",
        experimental=True,
    )


def ESMFold2ExperimentalFast300M(steps: str = "1500k") -> ESMFold2:
    """300M ESMFold2-Experimental-Fast design model. See ``ESMFold2ExperimentalFastBase``."""
    return ESMFold2ExperimentalFastBase("300M", steps)


def ESMFold2ExperimentalFast600M(steps: str = "1500k") -> ESMFold2:
    """600M ESMFold2-Experimental-Fast design model. See ``ESMFold2ExperimentalFastBase``."""
    return ESMFold2ExperimentalFastBase("600M", steps)


def ESMFold2ExperimentalFast6B(steps: str = "1500k") -> ESMFold2:
    """6B ESMFold2-Experimental-Fast design model (large). See ``ESMFold2ExperimentalFastBase``."""
    return ESMFold2ExperimentalFastBase("6B", steps)


def ESMFold2ExperimentalFull() -> ESMFold2:
    """ESMFold2-Experimental with MSA encoder. AF3-style architecture; intended
    for target structure prediction with real MSAs.
    """
    return _make("biohub/ESMFold2-Experimental", experimental=True)

def ESMFold2ExperimentalFull2025() -> ESMFold2:
    """ESMFold2-Experimental with MSA encoder. AF3-style architecture; intended
    for target structure prediction with real MSAs.
    """
    return _make("biohub/ESMFold2-Experimental-Cutoff2025", experimental=True)
