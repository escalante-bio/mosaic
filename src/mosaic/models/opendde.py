"""Mosaic `StructurePredictionModel` wrapper for OpenDDE (the `jopendde` JAX
port of Aureka's OpenDDE — an AF3-style all-atom co-folding model).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import tempfile
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jopendde.diffusion import DiffusionModule
from jopendde.features import Features
from jopendde.model import OpenDDE, TrunkInputs
from jopendde.model import OpenDDE as JaxOpenDDE
from jopendde.model import TrunkState as JaxTrunkState
from jopendde.transformer import rearrange_qk_to_dense_trunk

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.sampling import Denoise, Sampler, schedules
from mosaic.sampling.core import mean_centering, no_alignment
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


def _denoise(diffusion_module: DiffusionModule, conditioning: dict, x, sigma):
    return diffusion_module(
        x_noisy=x[None], t_hat_noise_level=sigma[None].astype(x.dtype), **conditioning
    )[0]


def opendde_denoiser(
    model: OpenDDE, feat: Features, s_inputs, s, z
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the residue-branch trunk output (`OpenDDE.get_pairformer_output`).

    Returns the denoiser and the structural-token atom mask.
    """
    struct_feat, s_inputs, s, z, attn_bias = model.expand_to_structural_tokens(
        feat, s_inputs, s, z
    )
    module = model.diffusion_module
    pair_z = module.diffusion_conditioning.prepare_cache(struct_feat.relp, z)
    token_pair_bias = module.diffusion_transformer.precompute_pair_bias(
        pair_z, attn_bias
    )
    p_lm, c_l = module.atom_attention_encoder.prepare_cache(
        ref_pos=struct_feat.ref_pos,
        ref_charge=struct_feat.ref_charge,
        ref_mask=struct_feat.ref_mask,
        ref_atom_name_chars=struct_feat.ref_atom_name_chars,
        ref_element=struct_feat.ref_element,
        atom_to_token_idx=struct_feat.atom_to_token_idx,
        d_lm=struct_feat.d_lm,
        v_lm=struct_feat.v_lm,
        pad_info=struct_feat.pad_info,
        r_l=True,  # non-None gate for the token-pair-context term; value unused
        z=pair_z,
    )
    conditioning = {
        "feat": struct_feat,
        "s_inputs": s_inputs,
        "s_trunk": s,
        "z_trunk": z,
        "extra_attn_bias": attn_bias,
        # Step-invariant caches hoisted out of the loop, as in `sample_diffusion`.
        "pair_z": pair_z,
        "p_lm": p_lm,
        "c_l": c_l,
        "token_pair_bias": token_pair_bias,
    }
    denoise = eqx.Partial(_denoise, module, conditioning)
    return denoise, struct_feat.ref_mask.astype(jnp.float32)


def opendde_sampler(
    model: OpenDDE, num_steps: int, *, schedule: schedules.Schedule | None = None
) -> Sampler:
    """The `Sampler` reproducing `OpenDDE.sample_coordinates`.

    `schedule` defaults to the model's own `Karras`."""
    if schedule is None:
        schedule = schedules.Karras(rho=model.sched_rho)
    sigmas = schedules.zero_last(
        schedule(
            np.linspace(0.0, 1.0, num_steps + 1),
            sigma_data=model.sched_sigma_data,
            s_max=model.sched_s_max,
            s_min=model.sched_s_min,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=model.gamma0,
        gamma_min=model.gamma_min,
        noise_scale=model.noise_scale_lambda,
        step_scale=model.step_scale_eta,
        center=mean_centering,
        align=no_alignment,
    )

# OpenDDE `restype` indices 0-19 are the standard AAs in mosaic's TOKENS order
# ("ARNDCQEGHILKMFPSTWYV"); 20=UNK, 21-30=nucleic, 31=GAP. The mosaic-20 ->
# OpenDDE-32 map is thus an identity embedding into the first 20 columns.
_MOSAIC_TO_OPENDDE = np.zeros((20, 32), dtype=np.float32)
_MOSAIC_TO_OPENDDE[:20, :20] = np.eye(20)
GAP_IDX = 31


def _bin_centers(min_bin: float, max_bin: float, no_bins: int) -> jnp.ndarray:
    """Bin centres matching OpenDDE's `confidence_summary.get_bin_centers`."""
    width = (max_bin - min_bin) / no_bins
    boundaries = jnp.linspace(min_bin, max_bin - width, no_bins)
    return boundaries + 0.5 * width


def _inject_soft_pssm(
    new_sequence: Float[Array, "N 20"], feat: Features
) -> Features:
    """Inject a soft PSSM into the binder positions of a jopendde `Features`.

    The binder occupies the first `new_sequence.shape[0]` tokens. `restype` and
    `profile` are the two per-token channels the input embedder concatenates into
    `s_inputs`; the MSA query row (`msa.query_soft`) is the third. Injecting the
    same soft distribution into all three keeps them consistent and opens a
    gradient path through the MSA stack. Homolog rows (`msa.msa`) are constant
    target data and left untouched.

    This only rewrites the *soft-identity* channels (which carry the design
    gradient); the binder's atom + structural-token geometry is still the
    poly-Trp placeholder's until `refresh_binder_geometry` runs. `set_binder_sequence`
    chains the two so a caller never sees the inconsistent intermediate.
    """
    N = new_sequence.shape[0]
    soft = new_sequence @ _MOSAIC_TO_OPENDDE  # [N, 32]
    restype = feat.restype.at[:N].set(soft)
    profile = feat.profile.at[:N].set(soft)
    if feat.msa is not None:
        query_soft = feat.msa.query_soft.at[:N].set(soft.astype(feat.msa.query_soft.dtype))
        return eqx.tree_at(
            lambda f: (f.restype, f.profile, f.msa.query_soft),
            feat,
            (restype, profile, query_soft),
        )
    return eqx.tree_at(lambda f: (f.restype, f.profile), feat, (restype, profile))


# ---------------------------------------------------------------------------
# In-loop binder geometry refresh
# ---------------------------------------------------------------------------
# `binder_features` allocates the binder at a fixed max-atom budget (a poly-Trp
# buffer; Trp is the largest residue) so any designed sequence fits without a
# recompile. But the poly-Trp buffer's atom-level features are Trp's, not the
# designed residues'. `refresh_binder_geometry` rebuilds every atom- and
# structural-token-indexed field from `argmax(pssm)` so the OpenDDE trunk /
# diffusion see the designed side chains, while the PSSM gradient still flows
# only through `restype` / `profile` / MSA (the refresh runs under
# `stop_gradient`).
#
# Atom fields are tight-packed at the front of their fixed poly-Trp buffer;
# target atoms shift after the packed binder and the remaining tail is masked.
# Structural-token shape remains fixed at two tokens per binder residue. Gly's
# absent sidechain token is retained as a phantom token with no mapped atom.
#
# Binder `ref_pos` conformers receive a fresh random rigid augmentation on each
# loss call. Target conformers retain their featurized orientation.

MAX_ATOMS_PER_RES = 15  # C-terminal Trp; the max atom budget across the 20 AAs.
MAX_STRUCT_PER_RES = 2  # protein backbone + sidechain sub-token.


class OpenDDEAtomTemplates(eqx.Module):
    """Per-residue (mosaic-20 / ``TOKENS`` order) OpenDDE atom + structural-token
    templates, in two contexts: ``int`` (internal / N-terminal) and ``cterm``
    (C-terminal, which carries an extra OXT atom). Built once on the host from
    the torch featurizer; ``ref_pos`` is centred per residue (over its real
    atoms) so the in-loop refresh can apply a fresh random rigid augmentation.

    Leading axis 2 is the context: index 0 = ``int``, 1 = ``cterm``.
    """

    # atom-axis (context, restype, atom-slot, ...)
    ref_pos: Float[Array, "2 20 A 3"]
    ref_element: Float[Array, "2 20 A 128"]
    ref_charge: Float[Array, "2 20 A"]
    ref_atom_name_chars: Float[Array, "2 20 A 4 64"]
    n_atoms: Int[Array, "2 20"]
    disto_off: Int[Array, "2 20"]  # residue-branch distogram rep local atom offset
    pae_off: Int[Array, "2 20"]  # residue-branch pae rep local atom offset
    # structural-token axis: per local atom -> local sub-token (0=bb, 1=sc) + tokatom
    a_struct_tok: Int[Array, "2 20 A"]
    a_struct_tokatom: Int[Array, "2 20 A"]
    # per local sub-token s in {0,1}: rep local atom offsets + frame (3 local offsets)
    s_disto_off: Int[Array, "2 20 2"]
    s_pae_off: Int[Array, "2 20 2"]
    s_frame_off: Int[Array, "2 20 2 3"]
    s_valid: Int[Array, "2 20 2"]  # 1 if the sub-token exists (sc absent for Gly)
    max_atoms: int = eqx.field(static=True, default=MAX_ATOMS_PER_RES)
    max_struct: int = eqx.field(static=True, default=MAX_STRUCT_PER_RES)


def build_opendde_atom_templates(featurize_one) -> OpenDDEAtomTemplates:
    """Build the per-residue templates by featurizing each amino acid in a
    tripeptide context and reading its atom block. ``featurize_one(seq)`` returns
    a jopendde ``Features`` for a single protein chain and is called for 20
    amino acids in two contexts.

    Internal templates come from the middle residue of ``G{aa}G``; C-terminal
    templates (with OXT) from the last residue of ``GG{aa}``.
    """
    print(
        "build_opendde_atom_templates: featurizing 40 tripeptides on the host "
        "(slow); this runs once and is cached to disk."
    )

    ma, ms = MAX_ATOMS_PER_RES, MAX_STRUCT_PER_RES

    def shp(*s):
        return np.zeros((2, 20, *s), np.float32)

    def ishp(*s):
        return np.zeros((2, 20, *s), np.int32)

    ref_pos, ref_el, ref_ch = shp(ma, 3), shp(ma, 128), shp(ma)
    ref_nm = shp(ma, 4, 64)
    n_at, d_off, p_off = ishp(), ishp(), ishp()
    a_stok, a_stka = ishp(ma), ishp(ma)
    s_doff, s_poff = ishp(ms), ishp(ms)
    s_froff, s_val = ishp(ms, 3), ishp(ms)

    for ci, ctx in enumerate(("int", "cterm")):
        tok = 1 if ctx == "int" else 2
        for i, aa in enumerate(TOKENS):
            seq = "G" + aa + "G" if ctx == "int" else "GG" + aa
            f = featurize_one(seq)

            def g(n, _f=f):
                return np.asarray(getattr(_f, n))

            a2t = g("atom_to_token_idx")
            idx = np.nonzero(a2t == tok)[0]
            base, k = int(idx[0]), len(idx)
            assert k <= ma, f"{aa} {ctx}: {k} atoms > MAX_ATOMS_PER_RES={ma}"
            m = a2t == tok
            block = g("ref_pos")[m]
            ref_pos[ci, i, :k] = block - block.mean(0)  # centre over real atoms
            ref_el[ci, i, :k] = g("ref_element")[m]
            ref_ch[ci, i, :k] = g("ref_charge")[m]
            ref_nm[ci, i, :k] = g("ref_atom_name_chars")[m]
            n_at[ci, i] = k
            d_off[ci, i] = [x - base for x in np.nonzero(g("distogram_rep_atom_mask"))[0] if a2t[x] == tok][0]
            p_off[ci, i] = [x - base for x in np.nonzero(g("pae_rep_atom_mask"))[0] if a2t[x] == tok][0]

            parent = g("parent_residue_idx")
            s_idx = np.nonzero(parent == tok)[0]
            s_base, ns = int(s_idx[0]), len(s_idx)
            ast, asta = g("atom_to_structural_token_idx"), g("atom_to_structural_tokatom_idx")
            sdrep, sprep = g("structural_distogram_rep_atom_mask"), g("structural_pae_rep_atom_mask")
            sframe = g("structural_frame_atom_index")
            for s in range(ns):
                gs = int(s_idx[s])
                s_val[ci, i, s] = 1
                drep = [x for x in np.nonzero(sdrep)[0] if ast[x] == gs]
                prep = [x for x in np.nonzero(sprep)[0] if ast[x] == gs]
                s_doff[ci, i, s] = (drep[0] - base) if drep else 0
                s_poff[ci, i, s] = (prep[0] - base) if prep else 0
                s_froff[ci, i, s] = sframe[gs] - base
            for a in range(k):
                ga = base + a
                a_stok[ci, i, a] = int(ast[ga]) - s_base
                a_stka[ci, i, a] = int(asta[ga])
            if ns < ms:  # Gly: the phantom sidechain token gets a benign frame
                s_froff[ci, i, 1] = s_froff[ci, i, 0]

    return OpenDDEAtomTemplates(
        ref_pos=jnp.asarray(ref_pos), ref_element=jnp.asarray(ref_el),
        ref_charge=jnp.asarray(ref_ch), ref_atom_name_chars=jnp.asarray(ref_nm),
        n_atoms=jnp.asarray(n_at), disto_off=jnp.asarray(d_off), pae_off=jnp.asarray(p_off),
        a_struct_tok=jnp.asarray(a_stok), a_struct_tokatom=jnp.asarray(a_stka),
        s_disto_off=jnp.asarray(s_doff), s_pae_off=jnp.asarray(s_poff),
        s_frame_off=jnp.asarray(s_froff), s_valid=jnp.asarray(s_val),
    )


def _random_rotations(key: jax.Array, n: int) -> Float[Array, "n 3 3"]:
    """`n` independent uniform SO(3) rotation matrices from random quaternions
    (matches the featurizer's `Rotation.random()` distribution)."""
    q = jax.random.normal(key, (n, 4))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return jnp.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        axis=-1,
    ).reshape(n, 3, 3)


def _rebuild_dense_trunk(ref_pos, atom_to_token_idx):
    """Rebuild OpenDDE's windowed atom-pair features (`d_lm`, `v_lm`, `pad_info`)
    from the repacked atoms: windowed pairwise `ref_pos` differences and a
    same-residue validity mask. `atom_to_token_idx` (one token per residue)
    stands in for `ref_space_uid` -- `v_lm` only tests equality."""
    a2t = atom_to_token_idx.astype(jnp.float32)
    qs, ks, pad = rearrange_qk_to_dense_trunk(
        [ref_pos, a2t], [ref_pos, a2t], [-2, -1], [-2, -1],
        n_queries=32, n_keys=128, compute_mask=True,
    )
    d_lm = qs[0][..., None, :] - ks[0][..., None, :, :]
    v_lm = (qs[1][..., None] == ks[1][..., None, :])[..., None].astype(ref_pos.dtype)
    return d_lm, v_lm, pad


def refresh_binder_geometry(
    feat: Features,
    pssm: Float[Array, "L 20"],
    tmpl: OpenDDEAtomTemplates,
    key: jax.Array,
    *,
    binder_length: int,
    binder_atom_alloc: int,
) -> Features:
    """Rebuild every atom- and structural-token-indexed field of a poly-Trp
    binder buffer so the trunk / diffusion see the ``argmax(pssm)`` residues'
    geometry, tight-packed with a fresh per-residue random rigid augmentation.

    ``binder_atom_alloc`` is the (static) atom budget the poly-Trp binder
    occupies at token positions ``[0, binder_length)``; target atoms/structural
    tokens follow and are shifted to sit immediately after the packed binder.
    Detached: the argmax + gather + augmentation run under ``stop_gradient``, so
    the PSSM gradient flows only through ``restype`` / ``profile`` / MSA.
    """
    L = binder_length
    A_alloc = binder_atom_alloc
    A_total = feat.ref_pos.shape[0]
    n_token = int(feat.restype.shape[0])
    n_tgt = A_total - A_alloc
    ma = tmpl.max_atoms

    ids = jax.lax.stop_gradient(jnp.argmax(pssm, axis=-1))  # [L] mosaic-20
    is_cterm = jnp.arange(L) == L - 1  # last binder residue carries OXT

    def pick(field):
        arr = getattr(tmpl, field)  # [2, 20, ...]
        vi, vc = arr[0][ids], arr[1][ids]  # internal / cterm gathers, [L, ...]
        w = is_cterm.reshape((L,) + (1,) * (vi.ndim - 1))
        return jnp.where(w, vc, vi)

    k = jnp.where(is_cterm, tmpl.n_atoms[1][ids], tmpl.n_atoms[0][ids]).astype(jnp.int32)
    offset = (jnp.cumsum(k) - k).astype(jnp.int32)  # exclusive prefix sum
    sum_k = jnp.sum(k)

    # per-residue random rigid augmentation (featurizer's ref_pos convention)
    kr, kt = jax.random.split(jax.lax.stop_gradient(key))
    R = _random_rotations(kr, L)
    t = jax.random.uniform(kt, (L, 3), minval=-1.0, maxval=1.0)
    pos = jnp.einsum("lac,lcd->lad", pick("ref_pos"), jnp.swapaxes(R, -1, -2)) + t[:, None, :]

    a_local = jnp.arange(ma, dtype=jnp.int32)[None, :]
    valid = a_local < k[:, None]
    dest = jnp.where(valid, offset[:, None] + a_local, A_total).reshape(-1)  # drop a>=k
    b_tok = jnp.repeat(jnp.arange(L, dtype=jnp.int32), ma)
    b_tka = jnp.broadcast_to(a_local, (L, ma)).reshape(-1)
    tgt_src = jnp.arange(A_alloc, A_total)
    tgt_dst = sum_k + jnp.arange(n_tgt)

    def scatter(buf, binder_vals, target_full):
        buf = buf.at[dest].set(binder_vals.astype(buf.dtype), mode="drop")
        return buf.at[tgt_dst].set(target_full[tgt_src].astype(buf.dtype), mode="drop")

    new_pos = scatter(jnp.zeros_like(feat.ref_pos), pos.reshape(-1, 3), feat.ref_pos)
    new_el = scatter(jnp.zeros_like(feat.ref_element), pick("ref_element").reshape(-1, 128), feat.ref_element)
    new_ch = scatter(jnp.zeros_like(feat.ref_charge), pick("ref_charge").reshape(-1), feat.ref_charge)
    new_nm = scatter(jnp.zeros_like(feat.ref_atom_name_chars), pick("ref_atom_name_chars").reshape(-1, 4, 64), feat.ref_atom_name_chars)
    # real atoms occupy [0, sum_k) (binder) + [sum_k, sum_k + n_tgt); mask the tail.
    new_mask = (jnp.arange(A_total) < (sum_k + n_tgt)).astype(feat.ref_mask.dtype)

    # atom -> token: tail padding maps to n_token (OOB -> dropped by every
    # aggregate_atom_to_token scatter-add, so it never dilutes a real token).
    def scatter_idx(buf_fill, binder_vals, target_full, dtype):
        buf = jnp.full((A_total,), buf_fill, dtype=dtype)
        buf = buf.at[dest].set(binder_vals.astype(dtype), mode="drop")
        return buf.at[tgt_dst].set(target_full[tgt_src].astype(dtype), mode="drop")

    new_a2t = scatter_idx(n_token, b_tok, feat.atom_to_token_idx, feat.atom_to_token_idx.dtype)
    new_tka = scatter_idx(0, b_tka, feat.atom_to_tokatom_idx, feat.atom_to_tokatom_idx.dtype)
    d_lm, v_lm, pad = _rebuild_dense_trunk(new_pos, new_a2t)

    # structural-token axis (fixed 2 sub-tokens per binder residue; Gly's sc is a
    # phantom no atom maps to): binder atom -> 2*residue + local(0=bb,1=sc).
    b_stok = 2 * b_tok + pick("a_struct_tok").reshape(-1)
    new_astok = scatter_idx(n_token, b_stok, feat.atom_to_structural_token_idx, feat.atom_to_structural_token_idx.dtype)
    new_astka = scatter_idx(0, pick("a_struct_tokatom").reshape(-1), feat.atom_to_structural_tokatom_idx, feat.atom_to_structural_tokatom_idx.dtype)

    # representative-atom masks (per-atom booleans), residue + structural branch.
    def rep_mask(binder_at, template_field):
        buf = jnp.zeros((A_total,), dtype=getattr(feat, template_field).dtype)
        buf = buf.at[binder_at].set(1, mode="drop")
        return buf.at[tgt_dst].set(getattr(feat, template_field)[tgt_src], mode="drop")

    new_drep = rep_mask(offset + pick("disto_off"), "distogram_rep_atom_mask")
    new_prep = rep_mask(offset + pick("pae_off"), "pae_rep_atom_mask")
    new_sdrep = rep_mask((offset[:, None] + pick("s_disto_off")).reshape(-1), "structural_distogram_rep_atom_mask")
    new_sprep = rep_mask((offset[:, None] + pick("s_pae_off")).reshape(-1), "structural_pae_rep_atom_mask")

    # frame atom indices (global): residue branch [n_token, 3] = N,CA,C; structural
    # branch [2L + n_struct_tgt, 3]. Target rows shift by (sum_k - A_alloc).
    shift = sum_k - A_alloc
    fbind = offset[:, None] + jnp.arange(3, dtype=feat.frame_atom_index.dtype)[None, :]
    new_frame = jnp.concatenate([fbind.astype(feat.frame_atom_index.dtype), feat.frame_atom_index[L:] + shift], axis=0)
    sfr_bind = (offset[:, None, None] + pick("s_frame_off")).reshape(2 * L, 3)
    new_sframe = jnp.concatenate(
        [sfr_bind.astype(feat.structural_frame_atom_index.dtype), feat.structural_frame_atom_index[2 * L:] + shift],
        axis=0,
    )

    return eqx.tree_at(
        lambda f: (
            f.ref_pos, f.ref_element, f.ref_charge, f.ref_atom_name_chars, f.ref_mask,
            f.atom_to_token_idx, f.atom_to_tokatom_idx, f.d_lm, f.v_lm, f.pad_info,
            f.atom_to_structural_token_idx, f.atom_to_structural_tokatom_idx,
            f.distogram_rep_atom_mask, f.pae_rep_atom_mask, f.frame_atom_index,
            f.structural_distogram_rep_atom_mask, f.structural_pae_rep_atom_mask,
            f.structural_frame_atom_index,
        ),
        feat,
        (
            new_pos, new_el, new_ch, new_nm, new_mask,
            new_a2t, new_tka, d_lm, v_lm, pad,
            new_astok, new_astka,
            new_drep, new_prep, new_frame,
            new_sdrep, new_sprep, new_sframe,
        ),
    )


class OpenDDEDesignFeatures(eqx.Module):
    """A design bundle from `OpenDDEModel.binder_features`: the poly-Trp jopendde
    `Features` together with everything the in-loop refresh needs.

    The inner `Features` is a max-atom (poly-Trp) placeholder; `atom_templates` +
    `binder_length` / `binder_atom_alloc` are the per-residue conformer tables
    and (static) binder extents `refresh_binder_geometry` consumes. Bundling
    keeps that metadata attached to the features so `set_binder_sequence` can
    *always* rebuild the binder geometry from the designed PSSM -- there's no way
    to set the binder identity without also refreshing its side chains.
    """

    features: Features
    atom_templates: OpenDDEAtomTemplates
    binder_length: int = eqx.field(static=True, default=0)
    binder_atom_alloc: int = eqx.field(static=True, default=0)


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: OpenDDEDesignFeatures,
    key: jax.Array,
) -> Features:
    """Set the binder identity of design `features` and refresh its geometry.

    Injects the soft PSSM into the binder's `restype` / `profile` / MSA-query
    channels (`_inject_soft_pssm`) and then rewrites the binder's atom +
    structural-token features from `argmax(pssm)` (`refresh_binder_geometry`, under
    `stop_gradient`), so the trunk / diffusion always see the *designed* residues'
    side chains rather than the poly-Trp placeholder's. Returns the refreshed plain
    jopendde `Features`, ready for `get_pairformer_output`.

    `key` drives the per-residue random rigid augmentation of the reference
    conformers (the featurizer re-orients them every call; the model is trained
    robust to it).
    """
    if not isinstance(features, OpenDDEDesignFeatures):
        raise TypeError(
            "set_binder_sequence expects OpenDDEDesignFeatures (from "
            "OpenDDEModel.binder_features); got "
            f"{type(features).__name__}. Target-only Features have no binder to set."
        )
    if new_sequence.shape[-1] != 20:
        raise ValueError(
            "OpenDDE binder PSSM must have 20 columns; "
            f"got {new_sequence.shape[-1]}"
        )
    if new_sequence.shape[0] != features.binder_length:
        raise ValueError(
            "OpenDDE binder PSSM length does not match design features: "
            f"expected {features.binder_length}, got {new_sequence.shape[0]}"
        )
    feat = _inject_soft_pssm(new_sequence, features.features)
    return refresh_binder_geometry(
        feat, new_sequence, features.atom_templates, key,
        binder_length=features.binder_length,
        binder_atom_alloc=features.binder_atom_alloc,
    )

# `model_output` and the loss builders.
_DESIGN_NUM_CYCLES = 4
# `predict`: upstream's `N_cycle` and `N_step` (`config/model_base.py`).
_INFERENCE_NUM_CYCLES = 10
_UPSTREAM_SAMPLING_STEPS = 200

# `opendde_v1` relative-position-encoding parameters.
_RELP_R_MAX = 32
_RELP_S_MAX = 2

_MSA_SERVER = "https://api.colabfold.com"


def _target_a3m(sequence: str) -> str:
    """Fetch (and cache) an unpaired a3m for `sequence` from the ColabFold MMseqs2
    server, returning the a3m path. Cached by sequence hash under
    `MOSAIC_CACHE_DIR/msa`. Only called for chains with `use_msa=True`."""
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    msa_cache = cache_dir() / "msa"
    msa_cache.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    a3m_path = msa_cache / f"{seq_hash}.a3m"
    if not a3m_path.exists():
        result = run_mmseqs2(
            [sequence],
            prefix=str(msa_cache / seq_hash),
            use_env=True,
            use_filter=True,
            use_pairing=False,
            host_url=_MSA_SERVER,
        )
        # run_mmseqs2 returns list[str] or a tuple of lists depending on version.
        a3m_strings = result[0] if isinstance(result, tuple) else result
        a3m_path.write_text(a3m_strings[0])
    return str(a3m_path)


def _set_asset_cache_dir(configs, cache_root: Path):
    root = cache_root / "opendde"
    configs.load_checkpoint_dir = str(root / "checkpoint")
    configs.data.ccd_components_file = str(root / "common" / "components.cif")
    configs.data.ccd_components_rdkit_mol_file = str(
        root / "common" / "components.cif.rdkit_mol.pkl"
    )


def _build_dense_atom_to_atom37() -> np.ndarray:
    """`[restype(32), tokatom_idx] -> atom37 slot`, -1 where invalid.

    `atom_to_tokatom_idx` indexes each atom into its residue's dense atom list
    (`ATOM14_PADDED` order); map each dense atom name to its atom37 slot. Only the
    20 protein rows are populated (UNK/nucleic/gap -> -1). OpenDDE's atom37
    ordering matches mosaic's, so the slots line up directly.
    """
    from opendde.data.constants import (
        ATOM14_PADDED,
        ATOM37_ORDER,
        PROTEIN_COMMON_ONE_TO_THREE,
        PROTEIN_TYPES_ONE_LETTER,
    )

    max_dense = max(len(v) for v in ATOM14_PADDED.values())
    table = -np.ones((32, max_dense), dtype=np.int32)
    for rt, letter in enumerate(PROTEIN_TYPES_ONE_LETTER):  # rt 0..19
        names = ATOM14_PADDED[PROTEIN_COMMON_ONE_TO_THREE[letter]]
        for j, name in enumerate(names):
            if name and name in ATOM37_ORDER:
                table[rt, j] = ATOM37_ORDER[name]
    return table


def _build_spec(chains: list[TargetChain], *, name: str = "design", seed: int = 0):
    """OpenDDE/AF3 input spec (chains in order; binder first for a design run)."""
    sequences = []
    for c in chains:
        if c.polymer_type != PolymerType.PROTEIN:
            raise NotImplementedError(
                f"OpenDDE wrapper supports protein chains only; got {c.polymer_type}"
            )
        chain = {"sequence": c.sequence, "count": 1}
        if c.use_msa:
            chain["unpairedMsaPath"] = _target_a3m(c.sequence)
        sequences.append({"proteinChain": chain})
    return [{"name": name, "modelSeeds": [seed], "sequences": sequences}]


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cached_featurize_config(cache_root: str):
    """Build the immutable base config for an architecture and cache root."""
    from opendde.config.inference import (
        build_inference_config,
        update_gpu_compatible_configs,
    )
    from opendde.utils.download import download_inference_cache

    cfg = build_inference_config(
        model_name="opendde_v1", fill_required_with_null=True
    )
    _set_asset_cache_dir(cfg, Path(cache_root))
    cfg.use_msa = cfg.use_template = cfg.use_rna_msa = False
    cfg.triangle_multiplicative = cfg.triangle_attention = "torch"
    cfg.enable_diffusion_shared_vars_cache = False
    cfg.enable_efficient_fusion = False
    cfg = update_gpu_compatible_configs(cfg)
    download_inference_cache(cfg)
    return cfg


def _featurize_configs():
    """Return an isolated `opendde_v1` featurization config."""
    return copy.deepcopy(_cached_featurize_config(str(cache_dir())))


def _relp(feat_dict: dict) -> dict:
    """Add the `opendde_v1` relative-position feature to a raw feature dict."""
    import types

    from opendde.model.opendde import RelativePositionEncoding

    ns = types.SimpleNamespace(r_max=_RELP_R_MAX, s_max=_RELP_S_MAX)
    return RelativePositionEncoding.generate_relp(ns, feat_dict, lazy=False)


def _target_template_features(
    chains: list[TargetChain], *, max_templates: int = 4
) -> dict[str, np.ndarray] | None:
    """Convert Mosaic's in-memory Gemmi chains to OpenDDE template features."""
    if not any(c.template_chain is not None for c in chains):
        return None

    from opendde.data.constants import ATOM14_PADDED, STD_RESIDUES_WITH_GAP
    from opendde.data.template.template_featurizer import Templates

    n_res = sum(len(c.sequence) for c in chains)
    gap = STD_RESIDUES_WITH_GAP["-"]
    # OpenDDE represents an empty template row with GAP and pads additional rows
    # with ALA. Padded identities still affect embeddings despite zero masks.
    aatype = np.zeros((max_templates, n_res), dtype=np.int32)
    aatype[0] = gap
    positions = np.zeros((max_templates, n_res, 24, 3), dtype=np.float32)
    mask = np.zeros((max_templates, n_res, 24), dtype=bool)

    offset = 0
    for chain in chains:
        template_chain = chain.template_chain
        if template_chain is not None:
            polymer = template_chain.get_polymer()
            template_sequence = gemmi.one_letter_code([res.name for res in polymer])
            if template_sequence != chain.sequence:
                raise ValueError(
                    "OpenDDE template sequence does not match its target sequence: "
                    f"{template_sequence!r} != {chain.sequence!r}"
                )
            for i, residue in enumerate(polymer):
                restype = STD_RESIDUES_WITH_GAP.get(
                    residue.name, STD_RESIDUES_WITH_GAP["UNK"]
                )
                aatype[0, offset + i] = restype
                atom_slots = {
                    name: j
                    for j, name in enumerate(ATOM14_PADDED.get(residue.name, ()))
                    if name
                }
                for atom in residue:
                    atom_name = atom.name.upper()
                    if atom_name == "SE" and residue.name == "MSE":
                        atom_name = "SD"
                    if atom_name in atom_slots:
                        j = atom_slots[atom_name]
                        positions[0, offset + i, j] = atom.pos.tolist()
                        mask[0, offset + i, j] = True
        offset += len(chain.sequence)

    return Templates(
        aatype=aatype, atom_positions=positions, atom_mask=mask
    ).as_opendde_dict()


def _featurize(chains: list[TargetChain]):
    """Featurize protein chains without loading model weights.

    Returns JOpenDDE features and the Biotite atom array for native output.
    """
    import torch
    from jopendde.features import Features
    from jopendde.inference import _spec_tempfile, _to_numpy
    from opendde.data.inference.infer_dataloader import get_inference_dataloader
    from opendde.model.opendde import update_input_feature_dict

    cfg = _featurize_configs()
    # Per-chain a3m presence (set in `_build_spec`) controls MSA depth; the
    # featurizer gate is on when any chain supplies one.
    cfg.use_msa = any(c.use_msa for c in chains)
    spec = _build_spec(chains)
    seed = int(spec[0]["modelSeeds"][0])
    with _spec_tempfile(spec) as json_path:
        cfg.input_json_path = json_path
        cfg.seeds = [seed]
        data, atom_array, err = next(iter(get_inference_dataloader(configs=cfg)))[0]
        if err:
            raise RuntimeError(f"OpenDDE featurization failed: {err}")
    raw = copy.deepcopy(data["input_feature_dict"])
    raw["inference_seed"] = torch.tensor(seed, dtype=torch.long)
    with torch.no_grad():
        raw = _relp(raw)
    raw = update_input_feature_dict(raw)
    raw_np = _to_numpy(raw)
    template_features = _target_template_features(chains)
    if template_features is not None:
        raw_np.update(template_features)
    return Features.from_dict(raw_np), atom_array


class OpenDDEModel(DiffusionStructurePredictionModel[Features | OpenDDEDesignFeatures, TrunkInputs]):
    """OpenDDE wrapped behind mosaic's structure-prediction interface.

    `recycling_steps` counts trunk passes (upstream's `N_cycle`).
    """

    model: JaxOpenDDE
    dense_atom_to_atom37: Int[Array, "32 Adense"]
    # Architecture-level residue atom/layout templates used to refresh design
    # geometry. These are not target-chain structural templates.
    atom_templates: OpenDDEAtomTemplates
    # (min_bin, max_bin, no_bins) for the confidence heads, read from the
    # checkpoint config at construction (static -> baked into the loss).
    pae_bin_params: tuple = eqx.field(static=True, default=(0.0, 32.0, 64))
    plddt_bin_params: tuple = eqx.field(static=True, default=(0.0, 1.0, 50))
    design_recycling_steps: int = eqx.field(static=True, default=_DESIGN_NUM_CYCLES)
    inference_recycling_steps: int = eqx.field(static=True, default=_INFERENCE_NUM_CYCLES)
    backprop_all_passes: bool = eqx.field(static=True, default=False)

    # ------------------------------------------------------------------ features
    def target_only_features(self, chains: list[TargetChain]):
        """Featurize real protein chains. Returns `(feat, atom_array)` where `feat`
        is the jopendde `Features` and `atom_array` is the biotite writer for
        native structure output. Model weights are not loaded by this method."""
        return _featurize(chains)

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        """Featurize a poly-Trp binder followed by target chains for design.

        Returns an `OpenDDEDesignFeatures` bundle whose static binder extents and
        atom templates allow geometry to be refreshed from each candidate PSSM.
        """
        if binder_length < 1:
            raise ValueError("OpenDDE binder_length must be positive")
        binder = TargetChain(sequence="W" * binder_length, use_msa=False)
        feat, atom_array = self.target_only_features([binder, *chains])
        observed_length, binder_atom_alloc = _binder_extents(feat)
        if observed_length != binder_length:
            raise ValueError(
                "OpenDDE featurization changed the binder length: "
                f"expected {binder_length}, got {observed_length}"
            )
        design_features = OpenDDEDesignFeatures(
            features=feat,
            atom_templates=self.atom_templates,
            binder_length=observed_length,
            binder_atom_alloc=binder_atom_alloc,
        )
        return design_features, atom_array

    # ------------------------------------------------------------------ loss
    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        if num_steps is None:
            num_steps = _UPSTREAM_SAMPLING_STEPS
        return opendde_sampler(self.model, num_steps, schedule=schedule)

    # -------------------------------------------------- structure-model contract
    def set_sequence(
        self, features: OpenDDEDesignFeatures, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ) -> OpenDDEDesignFeatures:
        """Set the binder's sequence channels and rebuild its geometry from
        `argmax(sequence)`. Needs the bundle from `binder_features`."""
        return eqx.tree_at(
            lambda f: f.features, features, set_binder_sequence(sequence, features, key)
        )

    def embed(self, features, *, key: PRNGKeyArray) -> TrunkInputs:
        del key
        return self.model.embed(_feat(features))

    def init_state(self, features, inputs: TrunkInputs, *, key: PRNGKeyArray) -> TrunkState:
        del features, key
        state = self.model.init_state(inputs)
        return TrunkState(pair=state.z, single=state.s)

    def step(self, features, inputs: TrunkInputs, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        del key
        out = self.model.step(_feat(features), inputs, _jopendde_state(state))
        return TrunkState(pair=out.z, single=out.s)

    def structure_denoiser(self, trunk: Trunk) -> tuple[Denoise, Float[Array, " A"]]:
        """Diffusion runs on the structural-token branch."""
        return opendde_denoiser(
            self.model, _feat(trunk.features), trunk.inputs.s_inputs,
            trunk.state.single, trunk.state.pair,
        )

    def distogram(self, trunk: Trunk) -> Distogram:
        m = self.model
        return Distogram(
            logits=m.distogram_head(trunk.state.pair),
            bins=_bin_centers(m.dist_min_bin, m.dist_max_bin, m.dist_no_bins),
        )

    def confidence(self, trunk: Trunk, coords: Float[Array, "A 3"]) -> Confidence:
        feat = _feat(trunk.features)
        plddt_logits, pae_logits, _pde, _resolved = self.model.run_confidence_head(
            feat, trunk.inputs.s_inputs, trunk.state.single, trunk.state.pair, coords[None]
        )
        # per-token pLDDT in [0, 1] from each token's representative atom
        rep_idx = jnp.nonzero(feat.distogram_rep_atom_mask.astype(bool), size=feat.restype.shape[0])[0]
        plddt = jax.nn.softmax(plddt_logits[0][rep_idx], axis=-1) @ _bin_centers(*self.plddt_bin_params)
        return Confidence(
            plddt=plddt, pae_logits=pae_logits[0], pae_bins=_bin_centers(*self.pae_bin_params)
        )

    def tokens(self, features) -> TokenInfo:
        feat = _feat(features)
        return TokenInfo(
            sequence=feat.restype[:, :20],
            asym_id=feat.asym_id.astype(jnp.int32),
            residue_index=feat.residue_index.astype(jnp.int32),
            mask=jnp.ones(feat.asym_id.shape, bool),
        )

    def atom37(self, features, coords: Float[Array, "A 3"]) -> Atom37:
        feat = _feat(features)
        a2t = feat.atom_to_token_idx.astype(jnp.int32)
        restype = feat.restype[:, :20].argmax(-1)
        idx = self.dense_atom_to_atom37[restype[a2t], feat.atom_to_tokatom_idx.astype(jnp.int32)]
        idx = jnp.where(feat.ref_mask > 0.5, idx, jnp.int32(-1))
        atom37_coords, atom37_mask = scatter_atom37(coords, a2t, idx, feat.restype.shape[0])
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(self, features, writer, output: StructureModelOutput, sequence) -> gemmi.Structure:
        """Written from the atom37 view."""
        del features, writer, sequence
        return jax.tree.map(np.asarray, output).to_structure()


def _feat(features: Features | OpenDDEDesignFeatures) -> Features:
    return features.features if isinstance(features, OpenDDEDesignFeatures) else features


def _jopendde_state(state: TrunkState) -> JaxTrunkState:
    return JaxTrunkState(s=state.single, z=state.pair)


def _binder_extents(features) -> tuple[int, int]:
    """Static (host) binder token count + atom budget of a design featurization: chain-0
    (asym_id 0) token count, and the number of atoms belonging to those tokens.
    Used to parameterize the in-loop geometry refresh."""
    asym = np.asarray(features.asym_id)
    binder_length = int((asym == asym[0]).sum())
    a2t = np.asarray(features.atom_to_token_idx)
    binder_atom_alloc = int((a2t < binder_length).sum())
    return binder_length, binder_atom_alloc


# Atom templates are architecture data derived from OpenDDE CCD features.
_ATOM_TEMPLATE_ARCH = "opendde_v1"
_ATOM_TEMPLATE_SCHEMA_VERSION = 1
_ATOM_TEMPLATE_CACHE: dict[tuple[str, str, str, int], OpenDDEAtomTemplates] = {}


def _jopendde_build_id() -> str:
    """Return the installed JOpenDDE package revision or version."""
    try:
        dist = distribution("jopendde")
    except PackageNotFoundError:
        return "unknown"
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except json.JSONDecodeError:
            commit = None
        if commit:
            return commit
    return dist.version


def _atom_template_fields() -> list[str]:
    return [
        field.name
        for field in dataclasses.fields(OpenDDEAtomTemplates)
        if field.name not in ("max_atoms", "max_struct")
    ]


def _atom_template_shapes() -> dict[str, tuple[int, ...]]:
    atom = MAX_ATOMS_PER_RES
    struct = MAX_STRUCT_PER_RES
    return {
        "ref_pos": (2, 20, atom, 3),
        "ref_element": (2, 20, atom, 128),
        "ref_charge": (2, 20, atom),
        "ref_atom_name_chars": (2, 20, atom, 4, 64),
        "n_atoms": (2, 20),
        "disto_off": (2, 20),
        "pae_off": (2, 20),
        "a_struct_tok": (2, 20, atom),
        "a_struct_tokatom": (2, 20, atom),
        "s_disto_off": (2, 20, struct),
        "s_pae_off": (2, 20, struct),
        "s_frame_off": (2, 20, struct, 3),
        "s_valid": (2, 20, struct),
    }


def _validate_atom_template_cache(data, build_id: str) -> dict[str, np.ndarray]:
    metadata = {
        "_schema_version": str(_ATOM_TEMPLATE_SCHEMA_VERSION),
        "_architecture": _ATOM_TEMPLATE_ARCH,
        "_jopendde_build": build_id,
    }
    for name, expected in metadata.items():
        if name not in data or str(np.asarray(data[name]).item()) != expected:
            raise ValueError(f"invalid OpenDDE atom-template metadata: {name}")

    arrays = {}
    float_fields = {"ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars"}
    for name, shape in _atom_template_shapes().items():
        if name not in data:
            raise ValueError(f"missing OpenDDE atom-template field: {name}")
        value = np.asarray(data[name])
        if value.shape != shape:
            raise ValueError(
                f"invalid OpenDDE atom-template shape for {name}: "
                f"expected {shape}, got {value.shape}"
            )
        expected_kind = "f" if name in float_fields else "i"
        if value.dtype.kind != expected_kind:
            raise ValueError(
                f"invalid OpenDDE atom-template dtype for {name}: {value.dtype}"
            )
        arrays[name] = value
    return arrays


def _write_atom_template_cache(
    path: Path, templates: OpenDDEAtomTemplates, build_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: np.asarray(getattr(templates, name))
        for name in _atom_template_fields()
    }
    payload.update(
        {
            "_schema_version": np.asarray(str(_ATOM_TEMPLATE_SCHEMA_VERSION)),
            "_architecture": np.asarray(_ATOM_TEMPLATE_ARCH),
            "_jopendde_build": np.asarray(build_id),
        }
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_atom_template_cache(
    path: Path, build_id: str
) -> OpenDDEAtomTemplates:
    with np.load(path, allow_pickle=False) as data:
        arrays = _validate_atom_template_cache(data, build_id)
    return OpenDDEAtomTemplates(
        **{name: jnp.asarray(value) for name, value in arrays.items()}
    )


def _build_atom_template_cache(
    path: Path, build_id: str
) -> OpenDDEAtomTemplates:
    def featurize_one(sequence: str):
        features, _ = _featurize(
            [TargetChain(sequence=sequence, use_msa=False)]
        )
        return features

    templates = build_opendde_atom_templates(featurize_one)
    _write_atom_template_cache(path, templates, build_id)
    return templates


def _get_atom_templates() -> OpenDDEAtomTemplates:
    root = cache_dir()
    build_id = _jopendde_build_id()
    cache_key = (
        _ATOM_TEMPLATE_ARCH,
        str(root),
        build_id,
        _ATOM_TEMPLATE_SCHEMA_VERSION,
    )
    if cache_key in _ATOM_TEMPLATE_CACHE:
        return _ATOM_TEMPLATE_CACHE[cache_key]

    filename = (
        f"{_ATOM_TEMPLATE_ARCH}-v{_ATOM_TEMPLATE_SCHEMA_VERSION}-"
        f"{build_id[:12]}.npz"
    )
    path = root / "opendde" / "atom_templates" / filename
    if path.exists():
        try:
            templates = _load_atom_template_cache(path, build_id)
        except (OSError, ValueError, KeyError):
            path.unlink(missing_ok=True)
            templates = _build_atom_template_cache(path, build_id)
    else:
        templates = _build_atom_template_cache(path, build_id)

    _ATOM_TEMPLATE_CACHE[cache_key] = templates
    return templates


def _build_model(model_name: str, checkpoint_file: str | None = None) -> OpenDDEModel:
    from jopendde.inference import Predictor

    predictor = Predictor.from_checkpoint(
        model_name,
        checkpoint_file=checkpoint_file,
        asset_cache_dir=cache_dir() / "opendde",
    )
    sp = predictor.summary_params
    return OpenDDEModel(
        model=predictor.model,
        dense_atom_to_atom37=jnp.array(_build_dense_atom_to_atom37()),
        atom_templates=_get_atom_templates(),
        pae_bin_params=tuple(sp.pae_bins),
        plddt_bin_params=tuple(sp.plddt_bins),
    )


def OpenDDEModelV1() -> OpenDDEModel:
    return _build_model(
        "opendde_v1",
        checkpoint_file="opendde.pt",
    )


def OpenDDEModelAbag() -> OpenDDEModel:
    """Load the ABAG-optimized weights (`opendde_abag.pt`) for antibody-antigen
    complexes."""
    return _build_model(
        "opendde_v1",
        checkpoint_file="opendde_abag.pt",
    )
