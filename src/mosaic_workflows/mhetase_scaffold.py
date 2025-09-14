import os
from pathlib import Path
from typing import Tuple, TYPE_CHECKING, cast
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LossTerm
from mosaic.af2.alphafold2 import AF2
from mosaic.losses.af2 import AlphaFoldLoss
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
    def __call__(self, sequence, output, key):
        eps = 1e-8
        p = jnp.clip(sequence, eps, 1.0)
        ent = -jnp.sum(p * jnp.log(p), axis=-1)
        mean_ent = jnp.mean(ent)
        return -mean_ent, {"seq_entropy": mean_ent}


def _build_mhetase_yaml(*, binder_len: int, enzyme_chain: str = "A", ligand_chain: str = "L", ligand_ccd: str | None = None, ligand_smiles: str | None = None) -> str:
    if not (ligand_ccd or ligand_smiles):
        raise ValueError("Provide ligand_ccd or ligand_smiles")
    lines = ["version: 1", "sequences:"]
    lines.append(f"  - protein:\n      id: {enzyme_chain}\n      sequence: {'X'*binder_len}\n      msa: empty")
    if ligand_ccd:
        lines.append(f"  - ligand:\n      id: {ligand_chain}\n      ccd: {ligand_ccd}")
    else:
        lines.append(f"  - ligand:\n      id: {ligand_chain}\n      smiles: '{ligand_smiles}'")
    return "\n".join(lines)


def _build_boltz2_loss(*, binder_len: int, enzyme_chain: str, ligand_chain: str, ligand_ccd: str | None, ligand_smiles: str | None, base_loss: LossTerm) -> LossTerm:
    joltz2 = load_boltz2()
    es_yaml = _build_mhetase_yaml(binder_len=binder_len, enzyme_chain=enzyme_chain, ligand_chain=ligand_chain, ligand_ccd=ligand_ccd, ligand_smiles=ligand_smiles)
    features, _ = load_boltz2_features(es_yaml, cache=Path(os.environ.get("BOLTZ_CACHE", "/root/.boltz")).expanduser())
    return Boltz2Loss(joltz2=joltz2, features=features, loss=base_loss, deterministic=True, recycling_steps=0, name="boltz2")


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


def make_workflow(*, binder_len: int, tmol_context: dict, supervised_positions: Tuple[int, ...] | None = None, motif_pdb_path: str | Path | None = None, motif_chain_id: str | None = None, motif_resnums: Tuple[int, ...] | None = None, optimizer=None, use_af2: bool = False, af2_num_recycles: int = 1, af2_params_dir: str | None = None, steps: int = 200, lr: float = 0.1, w_contact: float = 1.0, w_motif_cce: float = 1.0, w_motif_rmsd: float = 0.0, w_plddt: float = 0.0, w_pae: float = 0.0, w_seq_ent: float = 0.0, fix_supervised_identities: Tuple[str, ...] | None = None, freeze_supervised_positions: bool = False, apgm_stepsize_coef: float = 0.1):
    motif_positions_tuple = None
    motif_template_ca = None
    if motif_pdb_path and motif_chain_id and motif_resnums:
        motif_template_ca = _build_motif_from_pdb(pdb_path=motif_pdb_path, chain_id=str(motif_chain_id), residue_numbers=tuple(int(x) for x in motif_resnums))
    if motif_template_ca is not None and supervised_positions and len(supervised_positions) == int(motif_template_ca.shape[0]):
        motif_positions_tuple = tuple(int(x) for x in supervised_positions)

    def motif_geo():
        if motif_template_ca is not None and motif_positions_tuple is not None:
            mp = tuple(int(x) for x in motif_positions_tuple)
            mt = jnp.asarray(motif_template_ca, dtype=jnp.float32)

            class _MotifCE_SP(LossTerm):
                def __call__(self, sequence, output, key):
                    L = sequence.shape[0]
                    logits = jnp.nan_to_num(output.distogram_logits[:L, :L], nan=0.0, posinf=0.0, neginf=0.0)
                    bins = output.distogram_bins
                    idx = jnp.asarray(mp, dtype=jnp.int32)
                    Q = mt
                    K = Q.shape[0]
                    dmat = jnp.sqrt(jnp.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=-1))
                    mask_pairs = (dmat <= 20.0) & (~jnp.eye(K, dtype=bool))
                    logits_sub = logits[idx[:, None], idx[None, :], :]
                    logp = jax.nn.log_softmax(logits_sub, axis=-1)
                    t_idx = jnp.argmin(jnp.abs(dmat[..., None] - bins[None, None, :]), axis=-1)
                    oh = jax.nn.one_hot(t_idx, logp.shape[-1], dtype=logp.dtype)
                    nll = -(oh * logp).sum(axis=-1)
                    mask_f = mask_pairs.astype(nll.dtype)
                    denom = mask_f.sum()
                    v = jnp.where(denom > 0, (nll * mask_f).sum() / denom, 0.0)
                    return v, {"motif_cce": v}

            class _MotifRMSD(LossTerm):
                def __call__(self, sequence, output, key):
                    L = sequence.shape[0]
                    ca = jnp.nan_to_num(output.backbone_coordinates[:L, 1, :], nan=0.0, posinf=0.0, neginf=0.0)
                    idx = jnp.asarray(mp, dtype=jnp.int32)
                    P = ca[idx]
                    Q = mt
                    w = jnp.ones(P.shape[-2], dtype=P.dtype)
                    rmsd = _cd_weighted_rmsd(Q, P, w)
                    return rmsd, {"motif_rmsd": rmsd}

            terms = []
            if float(w_motif_cce) != 0.0:
                terms.append(float(w_motif_cce) * _MotifCE_SP())
            if float(w_motif_rmsd) != 0.0:
                terms.append(float(w_motif_rmsd) * ClippedLoss(loss=_MotifRMSD(), l=0.0, u=5.0, name="motif_rmsd_clip"))
            if not terms:
                class _Zero(LossTerm):
                    def __call__(self, *a, **k):
                        return jnp.asarray(0.0, dtype=jnp.float32), {}
                return _Zero()
            r = terms[0]
            for t in terms[1:]:
                r = r + t
            return r
        class _Zero(LossTerm):
            def __call__(self, *a, **k):
                return jnp.asarray(0.0, dtype=jnp.float32), {}
        return _Zero()

    excl = tuple(supervised_positions) if supervised_positions else ()
    aux = float(w_contact) * ContactLoss(cutoff=14.0, binary=True, num=2, num_pos=1, seqsep=9, exclude_positions=excl)
    conf_terms = []
    if float(w_plddt) != 0.0:
        conf_terms.append(float(w_plddt) * PLDDTLoss(exclude_positions=excl))
    if float(w_pae) != 0.0:
        conf_terms.append(float(w_pae) * PAELoss(seqsep=9, exclude_positions=excl))
    conf = None
    if conf_terms:
        conf = conf_terms[0]
        for t in conf_terms[1:]:
            conf = conf + t
    if conf is None:
        class _Zero(LossTerm):
            def __call__(self, *a, **k):
                return jnp.asarray(0.0, dtype=jnp.float32), {}
        conf = _Zero()
    # Add priors: ProteinMPNN inverse-folding sequence recovery and no-cysteine penalty
    mpnn = ProteinMPNN.from_pretrained()
    seq_prior = 5.0 * InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.asarray(0.05), num_samples=8, jacobi_iterations=8)
    # Clip the prior like in the notebooks to avoid over-optimization pathologies
    seq_prior = ClippedLoss(loss=seq_prior, l=2.0, u=100.0, name="seq_prior_clip")
    no_cys = 0.1 * NoCysteine()

    # Structural + confidence + priors (no explicit sequence entropy)
    struct_full = aux + motif_geo() + conf + seq_prior + no_cys
    # Stabilize gradients as in notebooks
    struct_full = ClippedGradient(loss=struct_full, max_norm=1.0)

    # ensure type checker sees a single type for loss
    loss: LossTerm
    if use_af2:
        af2 = AF2(num_recycle=int(af2_num_recycles), data_dir=af2_params_dir or ".")
        feats, _ = af2.build_features(["X" * int(binder_len)], template_chains={})
        loss_full = cast(LossTerm, AlphaFoldLoss(
            forward=af2.jitted_apply,
            stacked_params=af2.stacked_model_params,
            features=feats,
            losses=struct_full,
            name="af2",
        ))
        loss_warmup = cast(LossTerm, AlphaFoldLoss(
            forward=af2.jitted_apply,
            stacked_params=af2.stacked_model_params,
            features=feats,
            losses=ClippedGradient(loss=aux, max_norm=1.0),
            name="af2",
        ))
    else:
        ligand = tmol_context.get("ligand", {})
        loss_full = cast(LossTerm, _build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=struct_full,
        ))
        loss_warmup = cast(LossTerm, _build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=ClippedGradient(loss=aux, max_norm=1.0),
        ))

    # Optional fixing of supervised positions
    post_logits = []
    grad_chain = [gradient_normalizer(mode="l2_effL")]
    if supervised_positions and freeze_supervised_positions:
        mask = np.ones(int(binder_len), dtype=np.float32)
        for p in supervised_positions:
            if 0 <= int(p) < int(binder_len):
                mask[int(p)] = 0.0
        grad_chain = [position_mask(mask)] + grad_chain
        # If identities to freeze are provided, also clamp tokens
        if fix_supervised_identities:
            vocab = "ARNDCQEGHILKMFPSTWYV"
            allowed = np.ones((int(binder_len), 20), dtype=np.float32)
            for pos_i, aa in zip(supervised_positions, fix_supervised_identities):
                if aa in vocab and 0 <= int(pos_i) < int(binder_len):
                    idx = vocab.index(aa)
                    allowed[int(pos_i), :] = 0.0
                    allowed[int(pos_i), idx] = 1.0
            post_logits.append(per_position_allowed_tokens(allowed))

    # Split steps similar to CD: warmup ~20%, soft ~33%, anneal rest
    warmup_steps = max(1, int(steps // 5))
    soft_steps = max(1, int(steps // 3))
    anneal_steps = max(1, int(steps - (warmup_steps + soft_steps)))

    def phase_dict(name: str, build_loss, n_steps: int, temperature: float, e_soft: float, anneal: bool = False, scale: float = 1.1, stepsize_coef: float | None = None):
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
                "grad": grad_chain,
                "post_logits": post_logits,
            },
            "analyzers": [],
            "analyze_every": 1,
        }

    phases = [
        phase_dict("warmup", lambda: loss_warmup, warmup_steps, temperature=1.0, e_soft=0.8, anneal=False, scale=1.1, stepsize_coef=0.1),
        phase_dict("soft", lambda: loss_full, soft_steps, temperature=1.0, e_soft=0.8, anneal=False, scale=1.1, stepsize_coef=0.1),
        phase_dict("anneal", lambda: loss_full, anneal_steps, temperature=1.0, e_soft=1.0, anneal=True, scale=1.5, stepsize_coef=0.2),
    ]

    return {"phases": phases, "binder_len": int(binder_len), "seed": 0}


def run(binder_len: int, tmol_context: dict, supervised_positions: Tuple[int, ...] | None = None, motif_pdb_path: str | Path | None = None, motif_chain_id: str | None = None, motif_resnums: Tuple[int, ...] | None = None, initial_x: np.ndarray | None = None):
    wf = make_workflow(binder_len=binder_len, tmol_context=tmol_context, supervised_positions=supervised_positions, motif_pdb_path=motif_pdb_path, motif_chain_id=motif_chain_id, motif_resnums=motif_resnums)
    wf["initial_x"] = initial_x if initial_x is not None else np.random.randn(binder_len, 20).astype(np.float32) * 0.1
    return run_workflow(wf)

