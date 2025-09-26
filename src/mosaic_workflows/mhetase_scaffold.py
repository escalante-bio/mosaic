import os
from pathlib import Path
from typing import Tuple, TYPE_CHECKING, cast, List, Optional, Any
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LossTerm
from mosaic.models.af2 import AlphaFold2 as AF2Model
from mosaic.losses.transformations import ClippedLoss, ClippedGradient
from mosaic.structure_prediction import TargetChain
from mosaic.losses.boltz2 import (
    Boltz2Loss,
    load_boltz2,
    load_features_and_structure_writer as load_boltz2_features,
)
from mosaic_workflows.design import run_workflow
from mosaic_workflows.optimizers import simplex_APGM_adapter as simplex_apgm
from mosaic_workflows.transforms import (
    per_position_allowed_tokens,
    per_position_allowed_probs,
    position_mask,
    temperature_on_logits,
    e_soft_on_logits,
    gradient_normalizer,
)
from mosaic.proteinmpnn.mpnn import ProteinMPNN
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
 
import mosaic.losses.structure_prediction as sp

"""MHETase scaffolding (ColabDesign-style, Mosaic minimal).

Single-phase masked contact loss; optional motif distogram CCE from PDB.
Backends: AF2 or Boltz2.
"""


class MotifDistogramCCE(LossTerm):
    motif_positions: Tuple[int, ...]
    motif_template_ca: np.ndarray
    max_pair_distance: float
    def __init__(self, *, motif_positions: Tuple[int, ...], motif_template_ca: np.ndarray, max_pair_distance: float = 20.0):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))
        object.__setattr__(self, "max_pair_distance", float(max_pair_distance))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        logits = jnp.nan_to_num(output.distogram_logits[:L, :L], nan=0.0, posinf=0.0, neginf=0.0)
        bins = output.distogram_bins

        idx = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        Q = self.motif_template_ca  # [K,3]
        K = Q.shape[0]

        dmat = jnp.sqrt(jnp.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=-1))
        mask_pairs = (dmat <= self.max_pair_distance) & (~jnp.eye(K, dtype=bool))

        logits_sub = logits[idx[:, None], idx[None, :], :]
        logp = jax.nn.log_softmax(logits_sub, axis=-1)
        t_idx = jnp.argmin(jnp.abs(dmat[..., None] - bins[None, None, :]), axis=-1)
        oh = jax.nn.one_hot(t_idx, logp.shape[-1], dtype=logp.dtype)
        nll = -(oh * logp).sum(axis=-1)
        mask_f = mask_pairs.astype(nll.dtype)
        denom = mask_f.sum()
        loss = jnp.where(denom > 0, (nll * mask_f).sum() / denom, 0.0)
        return loss, {"motif_cce": loss}


def _cd_np_kabsch(a: jax.Array, b: jax.Array) -> jax.Array:
    _np = jnp
    ab = a.swapaxes(-1, -2) @ b
    u, s, vh = _np.linalg.svd(ab, full_matrices=False)
    flip = _np.linalg.det(u @ vh) < 0
    u_ = _np.where(flip, -u[...,-1].T, u[...,-1].T).T
    u = u.at[...,-1].set(u_)
    return u @ vh


def _cd_weighted_rmsd(true_xyz: jax.Array, pred_xyz: jax.Array, weights_1d: jax.Array) -> jax.Array:
    w = weights_1d / (jnp.sum(weights_1d) + 1e-8)                     # [K]
    T_mu = jnp.sum(true_xyz * w[:, None], axis=-2, keepdims=True)     # [1,3]
    P_mu = jnp.sum(pred_xyz * w[:, None], axis=-2, keepdims=True)     # [1,3]
    aln = _cd_np_kabsch((pred_xyz - P_mu) * w[:, None], (true_xyz - T_mu))
    align = lambda x: (x - P_mu) @ aln + T_mu
    sd = jnp.sum(jnp.square(align(pred_xyz) - true_xyz), axis=-1)     # [K]
    msd = jnp.sum(w * sd)                                             # ()
    return jnp.sqrt(msd + 1e-8)


class MotifRMSDCA(LossTerm):
    motif_positions: Tuple[int, ...]
    motif_template_ca: jax.Array
    hinge_tau: float | None
    def __init__(self, *, motif_positions: Tuple[int, ...], motif_template_ca: np.ndarray, hinge_tau: float | None = None):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))
        object.__setattr__(self, "hinge_tau", None if hinge_tau is None else float(hinge_tau))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        ca = output.backbone_coordinates[:L, 1, :]  # [L,3]
        idx = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        P = ca[idx]                  # [K,3]
        Q = self.motif_template_ca   # [K,3]
        w = jnp.ones(P.shape[-2], dtype=P.dtype)
        rmsd = _cd_weighted_rmsd(Q, P, w)
        return rmsd, {"motif_rmsd": rmsd}



class MotifSidechainRMSD(LossTerm):
    """Side-chain RMSD over motif residues using Joltz/Boltz atom ordering.

    Requires a Boltz2Output providing structure_coordinates and features with
    atom_to_token mapping. Falls back to zero if unavailable.
    """
    motif_positions: Tuple[int, ...]
    motif_sidechains: Tuple[Tuple[str, Tuple[str, ...], jax.Array], ...]
    def __init__(
        self,
        *,
        motif_positions: Tuple[int, ...],
        motif_sidechains: Tuple[Tuple[str, Tuple[str, ...], np.ndarray], ...],
    ):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        packed = []
        for resname, names, coords in motif_sidechains:
            packed.append((str(resname), tuple(names), jnp.asarray(coords, dtype=jnp.float32)))
        object.__setattr__(self, "motif_sidechains", tuple(packed))

    def __call__(self, sequence, output, key):
        import boltz.data.const as bconst  # type: ignore

        L = sequence.shape[0]
        features = jax.tree.map(lambda x: x[0], output.features)
        first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(features["atom_to_token"].T)
        all_atom_coords = output.structure_coordinates[0]

        Q_list = []
        P_list = []

        for pos_i, (resname_tpl, atom_names_tpl, coords_tpl) in zip(self.motif_positions, self.motif_sidechains):
            if coords_tpl.shape[0] == 0:
                continue
            ref_list = bconst.ref_atoms.get(resname_tpl, [])
            idx_offsets_py = []
            Q_atoms_py = []
            for nm, q_coord in zip(atom_names_tpl, coords_tpl):
                if nm in ("N", "CA", "C", "O"):
                    continue
                if nm in ref_list:
                    idx_offsets_py.append(ref_list.index(nm))
                    Q_atoms_py.append(q_coord)
            if not idx_offsets_py:
                continue
            start = first_atom_idx[pos_i]
            offsets = jnp.asarray(idx_offsets_py, dtype=jnp.int32)
            atom_indices = start + offsets
            P_atoms = all_atom_coords[atom_indices]
            Q_atoms = jnp.asarray(Q_atoms_py, dtype=P_atoms.dtype)
            if P_atoms.shape[0] != Q_atoms.shape[0]:
                n = min(P_atoms.shape[0], Q_atoms.shape[0])
                P_atoms = P_atoms[:n]
                Q_atoms = Q_atoms[:n]
            P_list.append(P_atoms)
            Q_list.append(Q_atoms)

        if not P_list:
            z = jnp.asarray(0.0, dtype=jnp.float32)
            return z, {"motif_sc_rmsd": z}

        P = jnp.concatenate(P_list, axis=0)
        Q = jnp.concatenate(Q_list, axis=0)
        w = jnp.ones((P.shape[-2],), dtype=P.dtype)
        rmsd = _cd_weighted_rmsd(Q, P, w)
        return rmsd, {"motif_sc_rmsd": rmsd}

class MotifAutoDistogramCCE(LossTerm):
    """Automatic motif placement using pairwise soft-min CCE over all positions.

    Implements a differentiable surrogate of the triplet-based satisfaction in Wang et al. (2022),
    but using pairs for efficiency: for each motif residue pair (a,b) within 20Å in the template,
    take a soft minimum (temperature beta) of the negative log-likelihood over all binder pairs (i!=j).
    The final loss is the average over motif pairs.
    """
    motif_template_ca: jax.Array
    beta: float

    def __init__(self, *, motif_template_ca: np.ndarray, beta: float = 10.0):
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))
        object.__setattr__(self, "beta", float(beta))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        logits = jnp.nan_to_num(output.distogram_logits[:L, :L], nan=0.0, posinf=0.0, neginf=0.0)
        bins = output.distogram_bins  # [B]

        Q = self.motif_template_ca  # [K,3]
        K = Q.shape[0]
        dmat = jnp.sqrt(jnp.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=-1))  # [K,K]
        mask_pairs = (dmat <= 20.0) & (~jnp.eye(K, dtype=bool))

        # Compute per-(i,j) NLL for a given target distance d using nearest bin
        # logits: [L,L,B] -> logp: [L,L,B]
        logp = jax.nn.log_softmax(logits, axis=-1)

        def pair_softmin_loss(d):
            t_idx = jnp.argmin(jnp.abs(d - bins))  # scalar bin index for target distance
            nll = -logp[..., t_idx]  # [L,L]
            # Exclude diagonal (i==j)
            nll = jnp.where(~jnp.eye(L, dtype=bool), nll, jnp.inf)
            # Soft-min with temperature beta
            m = jnp.min(nll)
            z = jnp.sum(jnp.exp(-self.beta * (nll - m)))
            return m - (1.0 / self.beta) * jnp.log(z + 1e-9)

        # Average over all motif pairs within 20Å (avoid Python conditionals on tracers)
        pair_losses_all = jax.vmap(lambda d: pair_softmin_loss(d))(dmat.reshape(-1))
        mask_flat = mask_pairs.reshape(-1).astype(pair_losses_all.dtype)
        denom = jnp.sum(mask_flat)
        loss = jnp.sum(pair_losses_all * mask_flat) / jnp.maximum(denom, 1.0)
        return loss, {"motif_cce": loss}


class NoCysteine(LossTerm):
    def __call__(self, sequence, output, key):
        vocab = "ARNDCQEGHILKMFPSTWYV"
        idx_c = vocab.index("C")
        p_cys = jnp.sum(sequence[:, idx_c])
        return p_cys, {"p_cys": p_cys}


class ContactLoss(LossTerm):
    cutoff: float
    binary: bool
    num: int
    num_pos: int
    seqsep: int
    exclude_positions: Tuple[int, ...]
    def __init__(self, *, cutoff: float = 14.0, binary: bool = True, num: int = 2, num_pos: int = 1, seqsep: int = 9, exclude_positions: Tuple[int, ...] = ()):        
        object.__setattr__(self, "cutoff", float(cutoff))
        object.__setattr__(self, "binary", bool(binary))
        object.__setattr__(self, "num", int(num))
        object.__setattr__(self, "num_pos", int(num_pos))
        object.__setattr__(self, "seqsep", int(seqsep))
        object.__setattr__(self, "exclude_positions", tuple(int(p) for p in exclude_positions))

    @staticmethod
    def _min_k(x, k: int, mask: jax.Array | None = None):
        y = jnp.sort(x if mask is None else jnp.where(mask, x, jnp.nan))
        k_mask = jnp.logical_and(jnp.arange(y.shape[-1]) < k, jnp.isnan(y) == False)
        return jnp.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        dgram = output.distogram_logits[:L, :L, :]
        dgram_bins = output.distogram_bins

        bins_mask = dgram_bins < self.cutoff
        px = jax.nn.softmax(dgram, axis=-1)
        px_cut = jax.nn.softmax(dgram - 1e7 * (1.0 - bins_mask[None, None, :]), axis=-1)

        con_loss_cat_ent = -(px_cut * jax.nn.log_softmax(dgram, axis=-1)).sum(-1)
        con_loss_bin_ent = -jnp.log((bins_mask[None, None, :] * px + 1e-8).sum(-1))
        con_mtx = jnp.where(self.binary, con_loss_bin_ent, con_loss_cat_ent)

        # seqsep and position masks
        idx = jnp.arange(L)
        offset = idx[:, None] - idx[None, :]
        m_seqsep = jnp.abs(offset) >= int(self.seqsep)
        excl = jnp.zeros((L,), dtype=bool).at[jnp.asarray(self.exclude_positions, dtype=jnp.int32)].set(True)
        mask_1d = ~excl
        mask_2d = mask_1d[:, None] & mask_1d[None, :]
        m2d = jnp.where(mask_2d, m_seqsep, False)

        # first reduce over columns with min_k, then reduce over rows with min_k
        p_row = self._min_k(con_mtx, self.num, m2d)
        loss = self._min_k(p_row, self.num_pos, mask_1d)
        return loss, {"con": loss}


class PAELoss(LossTerm):
    seqsep: int
    exclude_positions: Tuple[int, ...]
    def __init__(self, *, seqsep: int = 9, exclude_positions: Tuple[int, ...] = ()):        
        object.__setattr__(self, "seqsep", int(seqsep))
        object.__setattr__(self, "exclude_positions", tuple(int(p) for p in exclude_positions))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        logits = output.pae_logits[:L, :L, :]
        breaks = output.pae_bins  # use bins as centers
        prob = jax.nn.softmax(logits, axis=-1)
        exp_err = (prob * breaks[None, None, :]).sum(-1) / 31.0
        exp_err = (exp_err + exp_err.T) / 2.0

        idx = jnp.arange(L)
        offset = idx[:, None] - idx[None, :]
        m_seqsep = jnp.abs(offset) >= int(self.seqsep)
        excl = jnp.zeros((L,), dtype=bool).at[jnp.asarray(self.exclude_positions, dtype=jnp.int32)].set(True)
        mask_1d = ~excl
        mask_2d = mask_1d[:, None] & mask_1d[None, :]
        m = mask_2d & m_seqsep

        denom = jnp.sum(m)
        loss = jnp.where(denom > 0, (exp_err * m).sum() / (denom + 1e-8), 0.0)
        return loss, {"pae": loss}


class PLDDTLoss(LossTerm):
    exclude_positions: Tuple[int, ...]
    def __init__(self, *, exclude_positions: Tuple[int, ...] = ()):        
        object.__setattr__(self, "exclude_positions", tuple(int(p) for p in exclude_positions))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        plddt = output.plddt[:L]
        p = 1.0 - plddt
        mask = jnp.ones((L,), dtype=bool).at[jnp.asarray(self.exclude_positions, dtype=jnp.int32)].set(False)
        denom = jnp.sum(mask)
        loss = jnp.where(denom > 0, (p * mask).sum() / denom, p.mean())
        return loss, {"plddt": loss}


class AF2SidechainRMSD(LossTerm):
    """Sidechain RMSD at specified positions using AF2 outputs and template atoms.

    - Uses predicted atom37 positions from AF2.
    - Uses template atom37 positions from features (template_all_atom_positions) as references.
    - Excludes backbone atoms (N, CA, C, O) from the RMSD calculation.
    - If positions is None, includes any residue with a nonzero template sidechain mask.
    """
    positions: tuple[int, ...] | None = None

    def __call__(self, sequence, output, *, key):
        from mosaic.alphafold.common import residue_constants as rc

        import jax.numpy as jnp

        N = sequence.shape[0]
        pred_all37 = jnp.nan_to_num(output.output.structure_module.final_atom_positions[:N])

        # Template references: take first template (if any) and restrict to binder length
        tmpl_pos_all = jnp.nan_to_num(output.features["template_all_atom_positions"]) 
        tmpl_mask_all = output.features["template_all_atom_mask"]
        if tmpl_pos_all.shape[0] == 0:
            z = jnp.asarray(0.0, dtype=jnp.float32)
            return z, {"af2_sc_rmsd": z}
        tmpl_pos = tmpl_pos_all[0, :N]
        tmpl_mask = tmpl_mask_all[0, :N]

        # Exclude backbone atoms
        backbone_idx = [rc.atom_order[k] for k in ["N","CA","C","O"]]
        sidechain_mask = jnp.ones_like(tmpl_mask).at[:, backbone_idx].set(0)
        sidechain_mask = sidechain_mask * tmpl_mask

        # Optionally restrict to provided positions
        if self.positions is not None and len(self.positions) > 0:
            keep = jnp.zeros((N,), dtype=tmpl_mask.dtype).at[jnp.asarray(self.positions, dtype=jnp.int32)].set(1)
            sidechain_mask = sidechain_mask * keep[:, None]

        # Weighted Kabsch alignment and RMSD
        def _np_kabsch(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            ab = a.T @ b
            u, s, vh = jnp.linalg.svd(ab, full_matrices=False)
            flip = jnp.linalg.det(u @ vh) < 0
            u_ = jnp.where(flip, -u[:, -1], u[:, -1])
            u = u.at[:, -1].set(u_)
            return u @ vh

        P_all = jnp.nan_to_num(pred_all37.reshape(-1, 3))
        Q_all = jnp.nan_to_num(tmpl_pos.reshape(-1, 3))
        w_mask = sidechain_mask.reshape(-1).astype(P_all.dtype)
        total = w_mask.sum()

        def _zero():
            z = jnp.asarray(0.0, dtype=jnp.float32)
            return z, {"af2_sc_rmsd": z}

        def _compute():
            w = w_mask / (total + 1e-8)
            P_mu = (P_all * w[:, None]).sum(0, keepdims=True)
            Q_mu = (Q_all * w[:, None]).sum(0, keepdims=True)
            R = _np_kabsch((P_all - P_mu) * w[:, None], (Q_all - Q_mu))
            align = lambda x: (x - P_mu) @ R + Q_mu
            sd = jnp.square(align(P_all) - Q_all).sum(-1)
            msd = (w * sd).sum()
            rmsd = jnp.sqrt(msd + 1e-8)
            return rmsd, {"af2_sc_rmsd": rmsd}

        import jax
        rmsd, aux = jax.lax.cond(total > 0, _compute, _zero)
        return rmsd, aux
        
AF2SidechainRMSD_Outer = AF2SidechainRMSD



class AF2MotifFAPE(LossTerm):
    """ColabDesign-style FAPE over motif positions using AF2 outputs and template atoms.

    Uses N, CA, C to build local frames; clamps per-pair error at 10.0 and divides by 10.0.
    Requires AF2 features to provide template_all_atom_positions and template_all_atom_mask.
    """
    positions: Tuple[int, ...]
    clamp: float
    def __init__(self, *, positions: Tuple[int, ...], clamp: float = 10.0):
        object.__setattr__(self, "positions", tuple(int(p) for p in positions))
        object.__setattr__(self, "clamp", float(clamp))

    def __call__(self, sequence, output, key):
        from mosaic.alphafold.common import residue_constants as rc

        L = sequence.shape[0]
        pred = jnp.nan_to_num(output.output.structure_module.final_atom_positions[:L])
        tpos_all = jnp.nan_to_num(output.features["template_all_atom_positions"]) 
        tmask_all = output.features["template_all_atom_mask"]
        if tpos_all.shape[0] == 0:
            z = jnp.asarray(0.0, dtype=jnp.float32)
            return z, {"fape": z}
        true = tpos_all[0, :L]
        tmask = tmask_all[0, :L]

        idx = jnp.asarray(self.positions, dtype=jnp.int32)
        true = true[idx]
        pred = pred[idx]
        tmask = tmask[idx]

        N, CA, C = (rc.atom_order[k] for k in ["N","CA","C"])
        w = tmask[:, N] * tmask[:, CA] * tmask[:, C]

        def _robust_norm(x):
            return jnp.sqrt(jnp.square(x).sum(-1) + 1e-8)

        def _get_R(Nxyz, CAxyz, Cxyz):
            v1, v2 = (Cxyz - CAxyz), (Nxyz - CAxyz)
            e1 = v1 / _robust_norm(v1)[..., None]
            c = jnp.einsum('li,li->l', e1, v2)[..., None]
            e2 = v2 - c * e1
            e2 = e2 / _robust_norm(e2)[..., None]
            e3 = jnp.cross(e1, e2, axis=-1)
            return jnp.concatenate([e1[:, :, None], e2[:, :, None], e3[:, :, None]], axis=-1)

        def _get_ij(R, T):
            return jnp.einsum('rji,rsj->rsi', R, T[None, :] - T[:, None])

        true_R = _get_R(true[:, N], true[:, CA], true[:, C])
        pred_R = _get_R(pred[:, N], pred[:, CA], pred[:, C])
        true_T = true[:, CA]
        pred_T = pred[:, CA]

        t_ij = _get_ij(true_R, true_T)
        p_ij = _get_ij(pred_R, pred_T)
        fape = _robust_norm(t_ij - p_ij)
        fape = jnp.clip(fape, 0.0, float(self.clamp)) / 10.0

        wm = (w[:, None] * w[None, :])
        denom = jnp.sum(wm)
        loss = jnp.where(denom > 0, (fape * wm).sum() / (denom + 1e-8), 0.0)
        return loss, {"fape": loss}


class SeqEntropyLoss(LossTerm):
    exclude_positions: Tuple[int, ...]
    def __init__(self, *, exclude_positions: Tuple[int, ...] = ()):        
        object.__setattr__(self, "exclude_positions", tuple(int(p) for p in exclude_positions))
    def __call__(self, sequence, output, key):
        eps = 1e-8
        p = jnp.clip(sequence, eps, 1.0)
        ent = -jnp.sum(p * jnp.log(p), axis=-1)  # [L]
        L = sequence.shape[0]
        mask = jnp.ones((L,), dtype=bool).at[jnp.asarray(self.exclude_positions, dtype=jnp.int32)].set(False)
        denom = jnp.sum(mask)
        mean_ent = jnp.where(denom > 0, (ent * mask).sum() / denom, ent.mean())
        return -mean_ent, {"seq_entropy": mean_ent}


class MaskedDistogramRadiusOfGyration(LossTerm):
    exclude_positions: Tuple[int, ...]
    target_radius: float | None
    def __init__(self, *, exclude_positions: Tuple[int, ...] = (), target_radius: float | None = None):
        object.__setattr__(self, "exclude_positions", tuple(int(p) for p in exclude_positions))
        object.__setattr__(self, "target_radius", None if target_radius is None else float(target_radius))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        logits = jnp.nan_to_num(output.distogram_logits[:L, :L, :], nan=0.0, posinf=0.0, neginf=0.0)
        bins = jnp.nan_to_num(output.distogram_bins, nan=0.0, posinf=0.0, neginf=0.0)
        # expected squared distances
        probs = jax.nn.softmax(logits, axis=-1)
        exp_d2 = jnp.sum(probs * (bins[None, None, :] ** 2), axis=-1)
        # mask out excluded rows/cols and diagonal
        excl = jnp.zeros((L,), dtype=bool).at[jnp.asarray(self.exclude_positions, dtype=jnp.int32)].set(True)
        keep = ~excl
        pair_mask = (keep[:, None] & keep[None, :]) & (~jnp.eye(L, dtype=bool))
        denom = jnp.sum(pair_mask)
        mean_exp_d2 = jnp.where(denom > 0, jnp.sum(exp_d2 * pair_mask) / denom, exp_d2.mean())
        rg = jnp.sqrt(mean_exp_d2 + 1e-8)
        rg_th = 2.38 * (jnp.sum(keep) ** 0.365) if self.target_radius is None else self.target_radius
        loss = jax.nn.elu(rg - rg_th)
        return loss, {"rg": rg}


def _zero_loss() -> LossTerm:
    """Return a no-op loss term that evaluates to 0.0.

    Invariants: callable compatible with LossTerm; safe under jit and composition.
    """
    class _Zero(LossTerm):
        def __call__(self, *a, **k):
            return jnp.asarray(0.0, dtype=jnp.float32), {}
    return _Zero()


def _sum_losses(terms: List[Any]) -> LossTerm:
    """Combine a list of loss terms into a single LossTerm.

    Skips None entries. Returns a zero loss if the list is empty or all None.
    Assumes each term is a LossTerm and supports addition semantics.
    """
    usable = [t for t in terms if t is not None]
    if not usable:
        return _zero_loss()
    acc = usable[0]
    for t in usable[1:]:
        acc = acc + t
    return acc


def _build_mhetase_yaml(*, binder_len: int, enzyme_chain: str = "A", ligand_chain: str = "L", ligand_ccd: str | None = None, ligand_smiles: str | None = None, template_pdb_path: str | None = None, template_chain_id: str | None = None, template_force: bool = True, template_threshold: float = 2.0) -> str:
    if not (ligand_ccd or ligand_smiles):
        raise ValueError("Provide ligand_ccd or ligand_smiles")
    lines = ["version: 1", "sequences:"]
    lines.append(f"  - protein:\n      id: {enzyme_chain}\n      sequence: {'X'*binder_len}\n      msa: empty")
    if ligand_ccd:
        lines.append(f"  - ligand:\n      id: {ligand_chain}\n      ccd: {ligand_ccd}")
    else:
        lines.append(f"  - ligand:\n      id: {ligand_chain}\n      smiles: '{ligand_smiles}'")
    if template_pdb_path:
        lines.append("templates:")
        lines.append(f"  - pdb: {template_pdb_path}")
        if template_chain_id:
            lines.append(f"    chain: {template_chain_id}")
        if template_force:
            lines.append("    force: true")
            lines.append(f"    threshold: {float(template_threshold)}")
    return "\n".join(lines)


def _build_boltz2_loss(*, binder_len: int, enzyme_chain: str, ligand_chain: str, ligand_ccd: str | None, ligand_smiles: str | None, base_loss: LossTerm, template_pdb_path: str | None = None, template_chain_id: str | None = None) -> LossTerm:
    joltz2 = load_boltz2()
    es_yaml = _build_mhetase_yaml(
        binder_len=binder_len,
        enzyme_chain=enzyme_chain,
        ligand_chain=ligand_chain,
        ligand_ccd=ligand_ccd,
        ligand_smiles=ligand_smiles,
        template_pdb_path=template_pdb_path,
        template_chain_id=template_chain_id,
        template_force=True,
        template_threshold=2.0,
    )
    features, _ = load_boltz2_features(es_yaml, cache=Path(os.environ.get("BOLTZ_CACHE", "/root/.boltz")).expanduser())
    return Boltz2Loss(joltz2=joltz2, features=features, loss=base_loss, deterministic=True, recycling_steps=0, name="boltz2")


def _build_motif_sidechains_from_pdb(
    *,
    pdb_path: str | Path,
    chain_id: str,
    residue_numbers: Tuple[int, ...],
):
    """Return per-residue sidechain coordinates and names for motif residues.

    Returns a list length K (motif size). Each element is a tuple:
      (resname: str, atom_names: list[str], coords: np.ndarray[num_atoms,3])
    Backbone atoms N, CA, C, O are excluded. Gly returns an empty list.
    """
    import importlib
    gemmi = importlib.import_module("gemmi")
    st = gemmi.read_structure(str(pdb_path))
    chain = st[0][chain_id]
    out = []
    for rn in residue_numbers:
        res = next(r for r in chain if r.seqid.num == int(rn))
        resname = str(res.name).upper()
        atom_names = []
        coords = []
        for atom in res:
            n = str(atom.name)
            if n in ("N", "CA", "C", "O"):
                continue
            atom_names.append(n)
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        out.append((resname, atom_names, np.asarray(coords, dtype=np.float32)))
    return out


class CatalyticProximityCA(LossTerm):
    """Minimize CA-CA distances for catalytic pairs: (Ser,His) and (His,Asp).

    Uses CA coordinates for simplicity and stability across predictors.
    """
    ser_idx: int
    his_idx: int
    asp_idx: int

    def __init__(self, *, ser_idx: int, his_idx: int, asp_idx: int):
        object.__setattr__(self, "ser_idx", int(ser_idx))
        object.__setattr__(self, "his_idx", int(his_idx))
        object.__setattr__(self, "asp_idx", int(asp_idx))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        ca = jnp.nan_to_num(output.backbone_coordinates[:L, 1, :], nan=0.0, posinf=0.0, neginf=0.0)
        s = jnp.asarray(self.ser_idx, dtype=jnp.int32)
        h = jnp.asarray(self.his_idx, dtype=jnp.int32)
        d = jnp.asarray(self.asp_idx, dtype=jnp.int32)
        ds = jnp.linalg.norm(ca[s] - ca[h])
        dh = jnp.linalg.norm(ca[h] - ca[d])
        loss = (ds + dh) / 2.0
        return loss, {"cat_dist_sh": ds, "cat_dist_hd": dh}


def _build_motif_from_pdb(*, pdb_path: str | Path, chain_id: str, residue_numbers: Tuple[int, ...]) -> np.ndarray:
    import importlib
    gemmi = importlib.import_module("gemmi")
    st = gemmi.read_structure(str(pdb_path))
    chain = st[0][chain_id]
    ca = []
    for rn in residue_numbers:
        res = next(r for r in chain if r.seqid.num == int(rn))
        ca_atom = None
        for atom in res:
            if atom.name == "CA":
                ca_atom = atom
                break
        if ca_atom is None:
            raise RuntimeError(f"Missing CA for residue {rn}")
        ca.append([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z])
    return np.asarray(ca, dtype=np.float32)


def make_workflow(*, binder_len: int, tmol_context: dict, supervised_positions: Tuple[int, ...] | None = None, motif_roles: Tuple[str, ...] | None = None, motif_pdb_path: str | Path | None = None, motif_chain_id: str | None = None, motif_resnums: Tuple[int, ...] | None = None, optimizer=None, use_af2: bool = False, af2_num_recycles: int = 1, af2_params_dir: str | None = None, steps: int = 410, lr: float = 0.1, w_contact: float = 1.0, w_motif_cce: float = 1.0, w_motif_rmsd: float = 1.0, w_sc_rmsd: float = 0.1, w_plddt: float = 0.0, w_pae: float = 0.0, w_helix: float = 0.0, w_seq_ent: float = 0.1, w_cat_dist: float = 0.1, w_fape: float = 1.0, w_exp_res: float = 0.0, w_rg: float = 0.0, fix_supervised_identities: Optional[str] | None = None, freeze_supervised_positions: bool = False, apgm_stepsize_coef: float = 0.1):
    """Build a three-phase MHETase scaffolding workflow with shared losses.

    Args:
      binder_len: Designed binder length (L).
      tmol_context: Dict with ligand spec used by Boltz2 (enzyme_chain, ligand_chain, smiles/ccd).
      supervised_positions: Optional tuple of motif positions (0-indexed) in the binder.
      motif_roles: Optional labels mapping motif indices to roles; must include ser, his, asp when
        catalytic losses (w_cat_dist) or masked seq entropy (w_seq_ent) are enabled.
      motif_pdb_path/chain_id/resnums: Template PDB motif definition for motif losses and AF2 templates.
      optimizer: Callable optimizer adapter; defaults to simplex APGM.
      use_af2: If True, use AF2; otherwise use Boltz2.
      af2_num_recycles/af2_params_dir: AF2 runtime settings.
      steps/lr: Total steps and learning rate; per-phase splits are derived from steps.
      Loss weights: w_contact, w_motif_cce, w_motif_rmsd, w_sc_rmsd, w_plddt, w_pae, w_rg,
        w_seq_ent, w_cat_dist.
      fix_supervised_identities: Optional identities to clamp at supervised positions.
      freeze_supervised_positions: If True, masks gradients at supervised positions after warmup.
      apgm_stepsize_coef: Stepsize coefficient for simplex APGM.

    Invariants:
      - If any motif-weighted loss is nonzero, a motif template and either supervised positions or
        auto-placement is required; else a ValueError is raised.
      - If w_seq_ent or w_cat_dist is nonzero and supervised positions are used, motif_roles must
        include ser, his, asp.
      - AF2 and Boltz2 share the same loss composition; AF2 uses its features; Boltz2 uses Joltz2.
    """
    motif_positions_tuple = None
    motif_template_ca = None
    motif_sidechains_tpl = None
    if motif_pdb_path and motif_chain_id and motif_resnums:
        motif_template_ca = _build_motif_from_pdb(pdb_path=motif_pdb_path, chain_id=str(motif_chain_id), residue_numbers=tuple(int(x) for x in motif_resnums))
        # sidechains with names for RMSD on side-chains
        sc_list = _build_motif_sidechains_from_pdb(pdb_path=motif_pdb_path, chain_id=str(motif_chain_id), residue_numbers=tuple(int(x) for x in motif_resnums))
        motif_sidechains_tpl = tuple(sc_list)
    if motif_template_ca is not None and supervised_positions and len(supervised_positions) == int(motif_template_ca.shape[0]):
        motif_positions_tuple = tuple(int(x) for x in supervised_positions)

    # Catalytic triad positions from roles (ser, his, asp) when supervised
    def _triad_from_roles(mp: Tuple[int, ...], roles: Tuple[str, ...] | None) -> Tuple[int, int, int]:
        """Return (ser, his, asp) binder indices from motif roles.

        Requires roles to include 'ser','his','asp' exactly once each.
        """
        if roles is None:
            raise ValueError("motif_roles must include 'ser','his','asp' when catalytic terms or masking are used")
        roles_l = [str(r).lower() for r in roles]
        if not all(k in roles_l for k in ("ser","his","asp")):
            raise ValueError("motif_roles must include 'ser','his','asp'")
        return int(mp[roles_l.index("ser")]), int(mp[roles_l.index("his")]), int(mp[roles_l.index("asp")])

    cat_positions: Tuple[int, ...] = ()
    if motif_positions_tuple is not None and supervised_positions is not None and len(supervised_positions) > 0:
        if float(w_seq_ent) != 0.0 or float(w_cat_dist) != 0.0:
            s_i, h_i, a_i = _triad_from_roles(tuple(int(x) for x in supervised_positions), motif_roles)
            cat_positions = (s_i, h_i, a_i)

    # ColabDesign-style transforms: temperature on logits, straight-through hard mix, and sequence-grad norm
    def _pre_logits_temperature():
        def fn(logits, ctx):
            sched = (ctx or {}).get("schedule", {})
            t = float(sched.get("temp", sched.get("temperature", 1.0)))
            # keep temperature within [1e-2, +inf) to avoid extreme softmax
            t = float(jnp.maximum(1e-2, t))
            return logits / t
        return fn

    def _pre_logits_flatten():
        def fn(logits, ctx):
            sched = (ctx or {}).get("schedule", {})
            e = float(sched.get("e_soft", 1.0))
            return logits * e
        return fn

    def _pre_probs_soft_hard():
        def fn(probs, ctx):
            sched = (ctx or {}).get("schedule", {})
            hard = float(sched.get("hard", 0.0))
            if hard <= 0.0:
                # sanitize probabilities
                return jnp.nan_to_num(jnp.clip(probs, 1e-9, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
            idx = jnp.argmax(probs, axis=-1)
            hard_oh = jax.nn.one_hot(idx, probs.shape[-1])
            # straight-through: forward hard; backward through soft
            surrogate = jnp.nan_to_num(jnp.clip(probs, 1e-9, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
            st = hard_oh + (surrogate - jax.lax.stop_gradient(surrogate))
            return st
        return fn

    def _post_logits_sanitize():
        def fn(logits, ctx):
            # Replace non-finite and clip logits range to maintain numerical stability
            x = jnp.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            return jnp.clip(x, -20.0, 20.0)
        return fn

    def _grad_cd_seq_norm():
        def fn(grad, ctx):
            g = grad
            eff_L = (jnp.square(g).sum(axis=-1, keepdims=True) > 0).sum(axis=-2, keepdims=True)
            gn = jnp.linalg.norm(g, axis=(-1, -2), keepdims=True)
            return g * jnp.sqrt(eff_L) / (gn + 1e-7)
        return fn

    def motif_geo(include_cat: bool = True, include_af2_sc: bool = True):
        """Assemble motif-related losses (CCE/RMSD/sidechain/catalytic) or auto-placement CCE.

        Requires template CA coordinates; with supervised positions adds exact placement losses.
        With auto-placement, uses soft-min distogram CCE over all placements.
        Raises when motif-weighted losses are requested but motif is not configured.
        """
        if motif_template_ca is not None and motif_positions_tuple is not None:
            mp = tuple(int(x) for x in motif_positions_tuple)
            mt = motif_template_ca

            terms: List[LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms.append(float(w_motif_cce) * MotifDistogramCCE(motif_positions=mp, motif_template_ca=mt))
            if float(w_motif_rmsd) != 0.0:
                rmsd_term = MotifRMSDCA(motif_positions=mp, motif_template_ca=mt)
                terms.append(float(w_motif_rmsd) * ClippedLoss(loss=rmsd_term, l=0.0, u=10.0, name="motif_rmsd_clip"))
            # optional side-chain RMSD
            if motif_sidechains_tpl is not None and float(w_sc_rmsd) != 0.0:
                if include_af2_sc and use_af2:
                    terms.append(float(w_sc_rmsd) * AF2SidechainRMSD(positions=mp))
                elif not use_af2:
                    terms.append(float(w_sc_rmsd) * MotifSidechainRMSD(motif_positions=mp, motif_sidechains=motif_sidechains_tpl))
            # catalytic proximity on CA: (Ser,His) + (His,Asp)
            if include_cat and float(w_cat_dist) != 0.0:
                s_i, h_i, a_i = _triad_from_roles(mp, motif_roles)
                terms.append(float(w_cat_dist) * CatalyticProximityCA(ser_idx=s_i, his_idx=h_i, asp_idx=a_i))
            return _sum_losses(terms)
        if motif_template_ca is not None and motif_positions_tuple is None:
            # Automatic placement: use pairwise soft-min CCE
            terms_auto: List[LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms_auto.append(float(w_motif_cce) * MotifAutoDistogramCCE(motif_template_ca=motif_template_ca, beta=10.0))
            return _sum_losses(terms_auto)
        # If motif terms are requested but no template/positions are configured, raise
        if any(float(x) != 0.0 for x in (w_motif_cce, w_motif_rmsd, w_sc_rmsd, w_cat_dist)):
            raise ValueError("Motif losses requested but motif template/positions are not configured")
        return _zero_loss()

    def motif_only_loss_term():
        """Motif-only losses (no catalytic distance, no other terms):
        - MotifDistogramCCE
        - MotifRMSDCA (clipped)
        - SidechainRMSD (clipped) if template sidechains available
        """
        if motif_template_ca is not None and motif_positions_tuple is not None:
            mp = tuple(int(x) for x in motif_positions_tuple)
            mt = motif_template_ca
            terms: List[LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms.append(float(w_motif_cce) * MotifDistogramCCE(motif_positions=mp, motif_template_ca=mt))
            if float(w_motif_rmsd) != 0.0:
                rmsd_term = MotifRMSDCA(motif_positions=mp, motif_template_ca=mt)
                terms.append(float(w_motif_rmsd) * ClippedLoss(loss=rmsd_term, l=0.0, u=10.0, name="motif_rmsd_clip"))
            if motif_sidechains_tpl is not None and float(w_sc_rmsd) != 0.0:
                if not use_af2:
                    sc_term: LossTerm = MotifSidechainRMSD(motif_positions=mp, motif_sidechains=motif_sidechains_tpl)
                    terms.append(float(w_sc_rmsd) * ClippedLoss(loss=sc_term, l=0.0, u=10.0, name="motif_sc_rmsd_clip"))
            return _sum_losses(terms)
        return _zero_loss()

    excl = tuple(supervised_positions) if supervised_positions else ()
    # Unsupervised (ColabDesign-style)
    aux = float(w_contact) * ContactLoss(cutoff=14.0, binary=False, num=2, num_pos=1, seqsep=9, exclude_positions=excl)
    conf = _sum_losses([
        (float(w_plddt) * PLDDTLoss(exclude_positions=excl)) if float(w_plddt) != 0.0 and bool(use_af2) else None,
        (float(w_pae) * PAELoss(seqsep=9, exclude_positions=excl)) if float(w_pae) != 0.0 and bool(use_af2) else None,
    ])
    # Optional helix bias (no silent fallback)
    from mosaic.losses.structure_prediction import HelixLoss as _HelixLoss
    helix_term = float(w_helix) * _HelixLoss()

    seq_ent = float(w_seq_ent) * SeqEntropyLoss(exclude_positions=cat_positions)
    # Add FAPE supervised on motif positions (AF2-only)
    def motif_fape_term():
        if bool(use_af2) and motif_positions_tuple is not None and float(w_fape) != 0.0:
            return float(w_fape) * AF2MotifFAPE(positions=tuple(int(x) for x in motif_positions_tuple), clamp=10.0)
        return _zero_loss()

    # Add priors: ProteinMPNN inverse-folding and no-cysteine (no silent fallback)
    mpnn = ProteinMPNN.from_pretrained()
    mpnn_prior = InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.asarray(0.05), num_samples=8, jacobi_iterations=8)
    sequence_prior_term = ClippedLoss(loss=5.0 * mpnn_prior, l=-np.inf, u=100.0, name="mpnn_clipped")
    no_cys = 0.1 * NoCysteine()

    # Optional masked radius of gyration (exclude motif)
    rg_term = float(w_rg) * MaskedDistogramRadiusOfGyration(exclude_positions=excl)

    # Build two full-loss variants: B/C without catalytic & AF2 sidechain; D with catalytic, AF2 SC off by default
    common_terms = [aux, motif_geo(include_cat=False, include_af2_sc=False), seq_ent, conf, helix_term, rg_term, motif_fape_term(), sequence_prior_term, no_cys]
    struct_full = ClippedGradient(loss=_sum_losses(common_terms), max_norm=1.0)
    struct_full_with_cat = ClippedGradient(loss=_sum_losses([
        aux, motif_geo(include_cat=True, include_af2_sc=False), seq_ent, conf, helix_term, motif_fape_term(), sequence_prior_term, no_cys
    ]), max_norm=1.0)

    # ensure type checker sees a single type for loss
    loss: LossTerm
    if use_af2:
        model = AF2Model(data_dir=af2_params_dir or ".")
        # Build binder-only features with optional partial template (ColabDesign-style)
        use_partial_template = motif_template_ca is not None and motif_positions_tuple is not None
        if use_partial_template:
            # Construct a partial binder template chain with CA (and optional sidechains) at motif positions
            import importlib
            gemmi = importlib.import_module("gemmi")
            chain = gemmi.Chain("A")
            # create residues 1..L (1-based in gemmi)
            for i in range(int(binder_len)):
                res = gemmi.Residue()
                res.name = "GLY"
                res.seqid = gemmi.SeqId(int(i + 1), " ")
                chain.add_residue(res)
            # prepare backbone coords (N, CA, C) for motif residues from source PDB so FAPE can be computed
            bb_coords = {}
            if motif_pdb_path is not None and motif_positions_tuple is not None:
                st = gemmi.read_structure(str(motif_pdb_path))
                ch = st[0][str(motif_chain_id) if motif_chain_id is not None else "A"]
                # map residue numbers to (N,CA,C)
                rn_to_bb = {}
                for rn in (motif_resnums or ()):  # type: ignore[arg-type]
                    res = next(r for r in ch if r.seqid.num == int(rn))
                    pos = {}
                    for atom in res:
                        n = str(atom.name)
                        if n in ("N","CA","C"):
                            pos[n] = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                    if all(k in pos for k in ("N","CA","C")):
                        rn_to_bb[int(rn)] = pos
                # align by index order between supervised positions and motif_resnums
                for idx_local, rn in enumerate(tuple(int(x) for x in (motif_resnums or ())) ):
                    if rn in rn_to_bb:
                        bb_coords[idx_local] = rn_to_bb[rn]

            # place backbone (N, CA, C) and optional sidechains at motif positions
            if motif_template_ca is not None and motif_positions_tuple is not None:
                for idx_local, binder_pos in enumerate(motif_positions_tuple):
                    if 0 <= int(binder_pos) < int(binder_len):
                        res = chain[int(binder_pos)]
                        # Set residue name from template PDB, even if no sidechain atoms (improves template_aatype)
                        if motif_sidechains_tpl is not None and idx_local < len(motif_sidechains_tpl):
                            resname, _, _ = motif_sidechains_tpl[idx_local]
                            res.name = str(resname)
                        # N, CA, C from PDB if available; otherwise at least CA from motif_template_ca
                        bb = bb_coords.get(idx_local)
                        if bb is not None:
                            for name in ("N","CA","C"):
                                a = gemmi.Atom(); a.name = name
                                x,y,z = bb[name]
                                a.pos.x, a.pos.y, a.pos.z = x,y,z
                                res.add_atom(a)
                        else:
                            a_ca = gemmi.Atom(); a_ca.name = "CA"
                            xyz = motif_template_ca[idx_local]
                            a_ca.pos.x, a_ca.pos.y, a_ca.pos.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                            res.add_atom(a_ca)
                        # Optional: add sidechain atoms if provided
                        if motif_sidechains_tpl is not None and idx_local < len(motif_sidechains_tpl):
                            _, atom_names, coords = motif_sidechains_tpl[idx_local]
                            for nm, q in zip(atom_names, coords):
                                if nm in ("N", "CA", "C", "O"):
                                    continue
                                a2 = gemmi.Atom()
                                a2.name = str(nm)
                                a2.pos.x, a2.pos.y, a2.pos.z = float(q[0]), float(q[1]), float(q[2])
                                res.add_atom(a2)
            # Wrap as TargetChain with template
            feats, _ = model.target_only_features([
                TargetChain(sequence="G" * int(binder_len), use_msa=False, template_chain=chain)
            ])
        else:
            feats, _ = model.binder_features(int(binder_len), chains=[])
        # Template injection for binder is not supported in the standardized AF2 interface.
        loss_full = cast(LossTerm, model.build_loss(
            loss=struct_full,  # type: ignore[arg-type]
            features=feats,
            recycling_steps=int(af2_num_recycles),
        ))
        loss_full_with_cat = cast(LossTerm, model.build_loss(
            loss=struct_full_with_cat,  # type: ignore[arg-type]
            features=feats,
            recycling_steps=int(af2_num_recycles),
        ))
        loss_motif_only = cast(LossTerm, model.build_loss(
            loss=ClippedGradient(loss=motif_only_loss_term(), max_norm=1.0),  # type: ignore[arg-type]
            features=feats,
            recycling_steps=int(af2_num_recycles),
        ))
        # Motif-only warmup phase (stabilizes early optimization)
        loss_warmup = loss_motif_only
    else:
        ligand = tmol_context.get("ligand", {})
        # If motif template is available, pass it to Boltz2 as a template
        tpl_pdb = str(motif_pdb_path) if motif_pdb_path else None
        tpl_chain = str(motif_chain_id) if motif_chain_id else None
        loss_full = cast(LossTerm, _build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=struct_full,
            template_pdb_path=tpl_pdb,
            template_chain_id=tpl_chain,
        ))
        loss_full_with_cat = cast(LossTerm, _build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=struct_full_with_cat,
            template_pdb_path=tpl_pdb,
            template_chain_id=tpl_chain,
        ))
        loss_motif_only = cast(LossTerm, _build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=motif_only_loss_term(),
            template_pdb_path=tpl_pdb,
            template_chain_id=tpl_chain,
        ))
        # Motif-only warmup phase (stabilizes early optimization)
        loss_warmup = loss_motif_only

    # Optional fixing/clamping of supervised positions (old-style transforms)
    grad_chain_warm = [gradient_normalizer(mode="l2_effL")]
    grad_chain_late = [gradient_normalizer(mode="l2_effL")]
    if freeze_supervised_positions and supervised_positions is not None and len(supervised_positions) > 0:
        mask = np.ones(int(binder_len), dtype=np.float32)
        for p in supervised_positions:
            if 0 <= int(p) < int(binder_len):
                mask[int(p)] = 0.0
        grad_chain_late = [position_mask(mask)] + grad_chain_late
    pre_logits_chain = [temperature_on_logits(), e_soft_on_logits()]
    pre_probs_chain: List[Any] = []
    post_logits_chain: List[Any] = []

    # If requested, clamp identities at supervised positions to provided letters
    if fix_supervised_identities and supervised_positions is not None and len(supervised_positions) > 0:
        vocab = "ARNDCQEGHILKMFPSTWYV"
        allowed = np.ones((int(binder_len), 20), dtype=np.float32)
        ids = [s.strip().upper() for s in str(fix_supervised_identities).split(',') if s.strip()]
        # Map each provided identity to the corresponding supervised position by index order
        for i, sup_pos in enumerate(tuple(int(x) for x in supervised_positions)):
            if i < len(ids) and ids[i] in vocab and 0 <= int(sup_pos) < int(binder_len):
                allowed[int(sup_pos), :] = 0.0
                allowed[int(sup_pos), vocab.index(ids[i])] = 1.0
        # Enforce at both logits and probability levels
        post_logits_chain = post_logits_chain + [per_position_allowed_tokens(allowed)]
        pre_probs_chain = pre_probs_chain + [per_position_allowed_probs(allowed)]

    # 3-phase (old-style): warmup 20%, soft 60%, anneal 20%
    total = max(1, int(steps))
    warmup_steps = max(1, int(round(total * 0.20)))
    soft_steps = max(1, int(round(total * 0.60)))
    anneal_steps = max(1, int(total - (warmup_steps + soft_steps)))

    def phase_dict(name: str, build_loss, n_steps: int, temperature: float, e_soft: float, anneal: bool = False, scale: float = 1.1, stepsize_coef: float | None = None, use_mask: bool = False):
        return {
            "name": name,
            "build_loss": build_loss,
            "optimizer": (optimizer or simplex_apgm),
            "steps": int(n_steps),
            "schedule": (lambda g, p: {
                "lr": float(lr),
                "stepsize": float((apgm_stepsize_coef if stepsize_coef is None else stepsize_coef)) * float(jnp.sqrt(jnp.maximum(1, binder_len))),
                "scale": float(scale),
                "temperature": float(jnp.maximum(0.05, (temperature if not anneal else temperature * jnp.exp(-3.0 * (g / jnp.maximum(1.0, n_steps)))))),
                "e_soft": float(e_soft),
            }),
            "transforms": {
                "pre_logits": pre_logits_chain,
                "pre_probs": pre_probs_chain,
                "grad": (grad_chain_late if use_mask else grad_chain_warm),
                "post_logits": post_logits_chain,
            },
            "analyzers": [],
            "analyze_every": 1,
        }

    phases = [
        phase_dict("motif_lock", lambda: loss_warmup, warmup_steps, temperature=1.0, e_soft=0.8, anneal=False, scale=1.1, stepsize_coef=0.1, use_mask=False),
        phase_dict("soft",       lambda: loss_full,   soft_steps,  temperature=1.0, e_soft=0.8, anneal=False, scale=1.1, stepsize_coef=0.1, use_mask=bool(freeze_supervised_positions)),
        phase_dict("anneal",     lambda: loss_full_with_cat, anneal_steps, temperature=1.0, e_soft=1.0, anneal=True,  scale=1.5, stepsize_coef=0.2, use_mask=bool(freeze_supervised_positions)),
    ]

    return {"phases": phases, "binder_len": int(binder_len), "seed": 0}


def run(binder_len: int, tmol_context: dict, supervised_positions: Tuple[int, ...] | None = None, motif_pdb_path: str | Path | None = None, motif_chain_id: str | None = None, motif_resnums: Tuple[int, ...] | None = None, initial_x: np.ndarray | None = None):
    wf = make_workflow(binder_len=binder_len, tmol_context=tmol_context, supervised_positions=supervised_positions, motif_pdb_path=motif_pdb_path, motif_chain_id=motif_chain_id, motif_resnums=motif_resnums)
    wf["initial_x"] = initial_x if initial_x is not None else np.random.randn(binder_len, 20).astype(np.float32) * 0.1
    return run_workflow(wf)

