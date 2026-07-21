"""Mosaic loss integration for OpenDDE (the `jopendde` JAX/Equinox port).

The differentiable design signal rides the distogram
(`distogram_logits`; cf. `DistogramIPTMProxy`, `BinderTargetContact`); pae/pLDDT
are consumed as scored values, not as a gradient channel.

The binder is allocated at a fixed poly-Trp (max-atom) budget so a candidate
panel shares one JIT compile (`OpenDDEModel.binder_features` -> an
`OpenDDEDesignFeatures`). `set_binder_sequence` injects the designed PSSM *and*
rewrites the binder's atom + structural-token features from `argmax(PSSM)` each
step via `refresh_binder_geometry` (under `stop_gradient`), so the trunk /
diffusion always see the *designed* residues' side chains -- not the
placeholder's -- while the gradient still flows only through `restype` /
`profile` / MSA. `OpenDDEDesignFeatures` carries atom templates and binder
extents the refresh needs (see `OpenDDEDesignFeatures`, `OpenDDEAtomTemplates`).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from jopendde.features import Features
from jopendde.model import OpenDDE as JaxOpenDDE
from jopendde.transformer import rearrange_qk_to_dense_trunk

from mosaic.common import TOKENS, LinearCombination, LossTerm
from mosaic.losses.atom37 import scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput, reduce_samples

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


def opendde_forward_from_trunk(
    model: JaxOpenDDE,
    feat: Features,
    s_inputs: Float[Array, "N Cs_in"],
    s: Float[Array, "N Cs"],
    z: Float[Array, "N N Cz"],
    key: jax.Array,
    *,
    n_step: int,
    dense_atom_to_atom37: Int[Array, "32 Adense"],
    pae_bin_params: tuple[float, float, int],
    plddt_bin_params: tuple[float, float, int],
    stop_grad_conf_coords: bool = False,
) -> StructureModelOutput:
    """Run the distogram head, one diffusion sample, and the confidence head from
    a pre-computed residue-branch trunk state -> `StructureModelOutput`.

    Diffusion runs on the structural-token branch; the distogram + confidence
    heads read the residue branch. A single sample is drawn (mosaic's B=1
    convention); the caller vmaps over keys for multi-sample losses.
    """
    n_token = int(feat.restype.shape[0])

    distogram_logits = model.distogram_head(z)
    distogram_bins = _bin_centers(
        model.dist_min_bin, model.dist_max_bin, model.dist_no_bins
    )

    struct_feat, si_st, s_st, z_st, attn_bias = model.expand_to_structural_tokens(
        feat, s_inputs, s, z
    )
    coords_ns = model.sample_coordinates(
        struct_feat, si_st, s_st, z_st, key,
        n_sample=1, n_step=n_step, extra_attn_bias=attn_bias,
    )  # [1, N_atom, 3]
    coords = coords_ns[0]

    # For design, optionally detach the diffusion coords so a confidence-head loss
    # backprops to the sequence only through the trunk reps (z), not the
    # stochastic sampler -- a deterministic, lower-variance gradient.
    conf_coords = jax.lax.stop_gradient(coords_ns) if stop_grad_conf_coords else coords_ns
    plddt_logits, pae_logits, _pde, _resolved = model.run_confidence_head(
        feat, s_inputs, s, z, conf_coords
    )
    plddt_logits = plddt_logits[0]  # [N_atom, Bplddt]
    pae_logits = pae_logits[0]      # [N, N, Bpae]

    a2t = feat.atom_to_token_idx.astype(jnp.int32)
    tokatom = feat.atom_to_tokatom_idx.astype(jnp.int32)

    # per-token pLDDT in [0, 1] from each token's representative atom
    plddt_centers = _bin_centers(*plddt_bin_params)
    rep_mask = feat.distogram_rep_atom_mask.astype(bool)
    rep_idx = jnp.nonzero(rep_mask, size=n_token)[0]
    plddt = jax.nn.softmax(plddt_logits[rep_idx], axis=-1) @ plddt_centers  # [N]

    pae_centers = _bin_centers(*pae_bin_params)
    pae = jax.nn.softmax(pae_logits, axis=-1) @ pae_centers  # [N, N]

    # atom37 view: map each atom to its slot via (designed restype, tokatom idx).
    restype_idx = feat.restype[:, :20].argmax(-1)  # [N] designed identity (0..19)
    atom37_idx = dense_atom_to_atom37[restype_idx[a2t], tokatom]  # [N_atom]
    atom37_idx = jnp.where(feat.ref_mask > 0.5, atom37_idx, jnp.int32(-1))
    atom37_coords, atom37_mask = scatter_atom37(coords, a2t, atom37_idx, n_token)
    # backbone N, CA, C, O == atom37 slots 0, 1, 2, 4
    backbone = atom37_coords[:, jnp.array([0, 1, 2, 4])]

    return StructureModelOutput(
        distogram_logits=distogram_logits,
        distogram_bins=distogram_bins,
        plddt=plddt,
        pae=pae,
        pae_logits=pae_logits,
        pae_bins=pae_centers,
        structure_coordinates=coords,
        backbone_coordinates=backbone,
        full_sequence=feat.restype[:, :20],
        asym_id=feat.asym_id.astype(jnp.int32),
        residue_idx=feat.residue_index.astype(jnp.int32),
        atom37_coords=atom37_coords,
        atom37_mask=atom37_mask,
    )


class MultiSampleOpenDDELoss(LossTerm):
    """Run the trunk once, then vmap diffusion, confidence, and loss.

    Vmap trades memory for throughput across samples.
    """

    model: JaxOpenDDE
    # Poly-Trp design features plus atom templates and binder extents. Setting
    # the binder sequence always refreshes its atom + structural-token geometry,
    # so the trunk sees the designed side chains (see `OpenDDEDesignFeatures`).
    features: OpenDDEDesignFeatures
    loss: LossTerm | LinearCombination
    dense_atom_to_atom37: Int[Array, "32 Adense"]
    num_cycles: int = eqx.field(static=True, default=4)
    n_step: int = eqx.field(static=True, default=20)
    num_samples: int = eqx.field(static=True, default=1)
    pae_bin_params: tuple = eqx.field(static=True, default=(0.0, 32.0, 64))
    plddt_bin_params: tuple = eqx.field(static=True, default=(0.0, 1.0, 50))
    stop_grad_conf_coords: bool = eqx.field(static=True, default=False)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)

    def __call__(self, sequence: Float[Array, "N 20"], key):
        # One geometry draw shared across diffusion samples (the featurizer
        # augments once per featurization); diffusion noise varies per sample below.
        key, geom_key = jax.random.split(key)
        feat = set_binder_sequence(sequence, self.features, geom_key)
        s_inputs, s, z = self.model.get_pairformer_output(feat, self.num_cycles)

        def single_sample(key):
            model_key, loss_key = jax.random.split(key)
            output = opendde_forward_from_trunk(
                self.model, feat, s_inputs, s, z, model_key,
                n_step=self.n_step,
                dense_atom_to_atom37=self.dense_atom_to_atom37,
                pae_bin_params=self.pae_bin_params,
                plddt_bin_params=self.plddt_bin_params,
                stop_grad_conf_coords=self.stop_grad_conf_coords,
            )
            return self.loss(sequence=sequence, output=output, key=loss_key)

        vs, auxs = jax.vmap(single_sample)(jax.random.split(key, self.num_samples))
        return reduce_samples(vs, auxs, self.reduction, self.num_samples)
