import os
from pathlib import Path
from typing import Tuple, TYPE_CHECKING, cast, List, Optional, Any
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LossTerm
from mosaic.af2.alphafold2 import AF2
from mosaic.losses.af2 import AlphaFoldLoss, AF2SidechainRMSD
from mosaic.af2.featurization import inject_partial_template_from_pdb
from mosaic.losses.boltz2 import (
    Boltz2Loss,
    load_boltz2,
    load_features_and_structure_writer as load_boltz2_features,
)
from mosaic_workflows.design import run_workflow
from mosaic_workflows.optimizers import sgd_logits_adapter as sgd_logits
from mosaic_workflows.optimizers import simplex_APGM_adapter as simplex_apgm
from mosaic_workflows.transforms import (
    temperature_on_logits,
    e_soft_on_logits,
    gradient_normalizer,
    position_mask,
    per_position_allowed_tokens,
)
from mosaic.proteinmpnn.mpnn import ProteinMPNN
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import ClippedLoss, ClippedGradient
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
        # store as tuples to keep pytree static
        packed = []
        for resname, names, coords in motif_sidechains:
            packed.append((str(resname), tuple(names), jnp.asarray(coords, dtype=jnp.float32)))
        object.__setattr__(self, "motif_sidechains", tuple(packed))

    def __call__(self, sequence, output, key):
        # Only supported for Boltz2Output which has structure_coordinates/features
        if not hasattr(output, "structure_coordinates"):
            z = jnp.asarray(0.0, dtype=jnp.float32)
            return z, {"motif_sc_rmsd": z}

        import boltz.data.const as bconst  # type: ignore

        L = sequence.shape[0]
        features = jax.tree.map(lambda x: x[0], output.features)
        # first atom index per token
        first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(features["atom_to_token"].T)
        all_atom_coords = output.structure_coordinates[0]

        Q_list = []
        P_list = []

        for pos_i, (resname_tpl, atom_names_tpl, coords_tpl) in zip(self.motif_positions, self.motif_sidechains):
            # Skip residues without sidechain atoms in template
            if coords_tpl.shape[0] == 0:
                continue
            # Use template residue name to define reference atom ordering;
            # we fix identities at motif positions so predicted tokens should match template.
            ref_list = bconst.ref_atoms.get(resname_tpl, [])
            # Build offsets for sidechain atom names present in template according to ref order
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
            # If counts mismatch due to template naming, align again by name order
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


def make_workflow(*, binder_len: int, tmol_context: dict, supervised_positions: Tuple[int, ...] | None = None, motif_roles: Tuple[str, ...] | None = None, motif_pdb_path: str | Path | None = None, motif_chain_id: str | None = None, motif_resnums: Tuple[int, ...] | None = None, optimizer=None, use_af2: bool = False, af2_num_recycles: int = 1, af2_params_dir: str | None = None, steps: int = 200, lr: float = 0.1, w_contact: float = 1.0, w_motif_cce: float = 1.0, w_motif_rmsd: float = 0.0, w_sc_rmsd: float = 0.1, w_plddt: float = 0.0, w_pae: float = 0.0, w_rg: float = 0.0, w_seq_ent: float = 0.0, w_cat_dist: float = 0.0, fix_supervised_identities: Tuple[str, ...] | None = None, freeze_supervised_positions: bool = False, apgm_stepsize_coef: float = 0.1):
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

    # Transforms helpers
    def _build_grad_transforms() -> tuple[list, list]:
        """Build gradient transform chains for warmup and later phases.

        Always includes gradient_normalizer; adds position_mask after warmup when freezing
        supervised_positions is requested.
        """
        warm = [gradient_normalizer(mode="l2_effL")]
        late = [gradient_normalizer(mode="l2_effL")]
        if supervised_positions and freeze_supervised_positions:
            mask = np.ones(int(binder_len), dtype=np.float32)
            for p in supervised_positions:
                if 0 <= int(p) < int(binder_len):
                    mask[int(p)] = 0.0
            late = [position_mask(mask)] + late
        return warm, late

    def _build_post_logits_transforms() -> list:
        """Build post-logits transforms for identity clamping at supervised positions.

        Returns a single per_position_allowed_tokens transform if identities are provided; otherwise empty.
        """
        if supervised_positions and fix_supervised_identities:
            vocab = "ARNDCQEGHILKMFPSTWYV"
            allowed = np.ones((int(binder_len), 20), dtype=np.float32)
            for pos_i, aa in zip(supervised_positions, fix_supervised_identities):
                if aa in vocab and 0 <= int(pos_i) < int(binder_len):
                    idx = vocab.index(aa)
                    allowed[int(pos_i), :] = 0.0
                    allowed[int(pos_i), idx] = 1.0
            return [per_position_allowed_tokens(allowed)]
        return []

    def motif_geo():
        """Assemble motif-related losses (CCE/RMSD/sidechain/catalytic) or auto-placement CCE.

        Requires template CA coordinates; with supervised positions adds exact placement losses.
        With auto-placement, uses soft-min distogram CCE over all placements.
        Raises when motif-weighted losses are requested but motif is not configured.
        """
        if motif_template_ca is not None and motif_positions_tuple is not None:
            mp = tuple(int(x) for x in motif_positions_tuple)
            mt = jnp.asarray(motif_template_ca, dtype=jnp.float32)

            terms: List[LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms.append(float(w_motif_cce) * MotifDistogramCCE(motif_positions=mp, motif_template_ca=mt))
            if float(w_motif_rmsd) != 0.0:
                terms.append(float(w_motif_rmsd) * MotifRMSDCA(motif_positions=mp, motif_template_ca=mt))
            # optional motif side-chain RMSD
            if motif_sidechains_tpl is not None and float(w_sc_rmsd) != 0.0:
                if use_af2:
                    terms.append(float(w_sc_rmsd) * AF2SidechainRMSD(positions=mp))
                else:
                    terms.append(float(w_sc_rmsd) * MotifSidechainRMSD(motif_positions=mp, motif_sidechains=motif_sidechains_tpl))
            # catalytic proximity on CA: (Ser,His) + (His,Asp)
            if float(w_cat_dist) != 0.0:
                s_i, h_i, a_i = _triad_from_roles(mp, motif_roles)
                terms.append(float(w_cat_dist) * CatalyticProximityCA(ser_idx=s_i, his_idx=h_i, asp_idx=a_i))
            return _sum_losses(terms)
        if motif_template_ca is not None and motif_positions_tuple is None:
            # Automatic placement: use pairwise soft-min CCE
            terms: List[LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms.append(float(w_motif_cce) * MotifAutoDistogramCCE(motif_template_ca=motif_template_ca, beta=10.0))
            return _sum_losses(terms)
        # If motif terms are requested but no template/positions are configured, raise
        if any(float(x) != 0.0 for x in (w_motif_cce, w_motif_rmsd, w_sc_rmsd, w_cat_dist)):
            raise ValueError("Motif losses requested but motif template/positions are not configured")
        return _zero_loss()

    excl = tuple(supervised_positions) if supervised_positions else ()
    aux = float(w_contact) * ContactLoss(cutoff=14.0, binary=True, num=2, num_pos=1, seqsep=9, exclude_positions=excl)
    conf = _sum_losses([
        (float(w_plddt) * PLDDTLoss(exclude_positions=excl)) if float(w_plddt) != 0.0 else None,
        (float(w_pae) * PAELoss(seqsep=9, exclude_positions=excl)) if float(w_pae) != 0.0 else None,
    ])
    # Add priors: ProteinMPNN inverse-folding sequence recovery and no-cysteine penalty (shared by both backends)
    mpnn = ProteinMPNN.from_pretrained()
    seq_prior = 5.0 * InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.asarray(0.05), num_samples=8, jacobi_iterations=8)
    seq_prior = ClippedLoss(loss=seq_prior, l=2.0, u=100.0, name="seq_prior_clip")
    no_cys = 0.1 * NoCysteine()

    # Optional masked radius of gyration (exclude motif) and priors
    rg_term = float(w_rg) * MaskedDistogramRadiusOfGyration(exclude_positions=excl)
    seq_ent = float(w_seq_ent) * SeqEntropyLoss(exclude_positions=cat_positions)
    common_terms = [aux, motif_geo(), seq_ent, conf, rg_term, seq_prior, no_cys]
    struct_full = ClippedGradient(loss=_sum_losses(common_terms), max_norm=1.0)

    # ensure type checker sees a single type for loss
    loss: LossTerm
    if use_af2:
        af2 = AF2(num_recycle=int(af2_num_recycles), data_dir=af2_params_dir or ".")
        # Build empty features; then inject partial template only at supervised positions
        feats, _ = af2.build_features(["X" * int(binder_len)], template_chains={})
        if motif_positions_tuple is not None and motif_pdb_path and motif_chain_id and motif_resnums is not None:
            feats = inject_partial_template_from_pdb(
                feats,
                np.asarray(motif_positions_tuple, dtype=np.int32),
                str(motif_pdb_path),
                str(motif_chain_id),
                np.asarray(motif_resnums, dtype=np.int32),
            )
        loss_full = cast(LossTerm, AlphaFoldLoss(
            forward=af2.jitted_apply,
            stacked_params=af2.stacked_model_params,
            features=feats,
            losses=struct_full,  # type: ignore[arg-type]
            name="af2",
        ))
        loss_motif_only = cast(LossTerm, AlphaFoldLoss(
            forward=af2.jitted_apply,
            stacked_params=af2.stacked_model_params,
            features=feats,
            losses=ClippedGradient(loss=_sum_losses([motif_geo(), seq_ent]), max_norm=1.0),  # type: ignore[arg-type]
            name="af2",
        ))
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
        loss_motif_only = cast(LossTerm, _build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=ClippedGradient(loss=_sum_losses([motif_geo(), seq_ent]), max_norm=1.0),
            template_pdb_path=tpl_pdb,
            template_chain_id=tpl_chain,
        ))
        loss_warmup = loss_motif_only

    # Optional fixing of supervised positions (centralized via helpers)
    grad_chain_warmup, grad_chain_later = _build_grad_transforms()
    post_logits = _build_post_logits_transforms()

    # Split steps similar to CD: warmup ~20%, soft ~33%, anneal rest
    warmup_steps = max(1, int(steps // 5))
    soft_steps = max(1, int(steps // 3))
    anneal_steps = max(1, int(steps - (warmup_steps + soft_steps)))

    def phase_dict(name: str, build_loss, n_steps: int, temperature: float, e_soft: float, anneal: bool = False, scale: float = 1.1, stepsize_coef: float | None = None, use_mask: bool = False):
        return {
            "name": name,
            "build_loss": build_loss,
            # Pluggable optimizer; default to simplex APGM (as in notebooks)
            "optimizer": optimizer or simplex_apgm,
            "steps": int(n_steps),
            "schedule": (lambda g, p: {
                "lr": float(lr),  # used by logits-based adapters
                # parameters used by simplex_APGM_adapter
                "stepsize": float((apgm_stepsize_coef if stepsize_coef is None else stepsize_coef)) * float(jnp.sqrt(jnp.maximum(1, binder_len))),
                "scale": float(scale),
                "temperature": float(jnp.maximum(0.05, (temperature if not anneal else temperature * jnp.exp(-3.0 * (g / jnp.maximum(1.0, steps)))))),
                "e_soft": float(e_soft),
            }),
            "transforms": {
                "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                "grad": (grad_chain_later if use_mask else grad_chain_warmup),
                "post_logits": post_logits,
            },
            "analyzers": [],
            "analyze_every": 1,
        }

    phases = [
        phase_dict("motif_lock", lambda: loss_warmup, warmup_steps, temperature=1.0, e_soft=0.8, anneal=False, scale=1.1, stepsize_coef=0.1, use_mask=False),
        phase_dict("soft", lambda: loss_full, soft_steps, temperature=1.0, e_soft=0.8, anneal=False, scale=1.1, stepsize_coef=0.1, use_mask=True),
        phase_dict("anneal", lambda: loss_full, anneal_steps, temperature=1.0, e_soft=1.0, anneal=True, scale=1.5, stepsize_coef=0.2, use_mask=True),
    ]

    return {"phases": phases, "binder_len": int(binder_len), "seed": 0}


def run(binder_len: int, tmol_context: dict, supervised_positions: Tuple[int, ...] | None = None, motif_pdb_path: str | Path | None = None, motif_chain_id: str | None = None, motif_resnums: Tuple[int, ...] | None = None, initial_x: np.ndarray | None = None):
    wf = make_workflow(binder_len=binder_len, tmol_context=tmol_context, supervised_positions=supervised_positions, motif_pdb_path=motif_pdb_path, motif_chain_id=motif_chain_id, motif_resnums=motif_resnums)
    wf["initial_x"] = initial_x if initial_x is not None else np.random.randn(binder_len, 20).astype(np.float32) * 0.1
    return run_workflow(wf)

