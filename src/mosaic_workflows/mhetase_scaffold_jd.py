import os
from pathlib import Path
from typing import Tuple, Optional, Any, List, cast
import jax
import jax.numpy as jnp
import numpy as np

# Reuse all loss building blocks and helpers from the baseline scaffold
from mosaic_workflows import mhetase_scaffold as base

# Optimizer: JD with conflict-aware aggregation
from mosaic_workflows.optimizers import jacobian_descent_adapter, jd_pcgrad_aggregator, jd_upgrad_aggregator

# Transforms: same CD-style stack plus convergence metric and freeze-on-threshold
from mosaic_workflows.transforms import (
    per_position_allowed_tokens,
    per_position_allowed_probs,
    position_mask,
    temperature_on_logits,
    e_soft_on_logits,
    gradient_normalizer,
    germinal_softmax_convergence,
)

from functools import partial


"""MHETase scaffolding (JD variant).

Identical losses and features as base `mhetase_scaffold`, but:
- Uses Jacobian Descent with PCGrad aggregation to reduce cross-objective interference
- Records a per-step convergence metric and freezes gradients when peaked
"""


def make_workflow(
    *,
    binder_len: int,
    tmol_context: dict,
    supervised_positions: Tuple[int, ...] | None = None,
    motif_roles: Tuple[str, ...] | None = None,
    motif_pdb_path: str | Path | None = None,
    motif_chain_id: str | None = None,
    motif_resnums: Tuple[int, ...] | None = None,
    optimizer=None,
    use_af2: bool = False,
    af2_num_recycles: int = 1,
    af2_params_dir: str | None = None,
    af2_model_idx: int | None = None,
    steps: int = 410,
    lr: float = 0.1,
    w_contact: float = 1.0,
    w_motif_cce: float = 1.0,
    w_motif_rmsd: float = 0.0,
    w_sc_rmsd: float = 0.1,
    w_plddt: float = 0.0,
    w_pae: float = 0.0,
    w_helix: float = 0.0,
    w_helix_cap: float = 0.0,
    w_seq_ent: float = 0.1,
    w_cat_dist: float = 0.1,
    w_fape: float = 1.0,
    w_exp_res: float = 0.0,
    w_rg: float = 0.0,
    fix_supervised_identities: Optional[str] | None = None,
    freeze_supervised_positions: bool = False,
    apgm_stepsize_coef: float = 0.1,
    jd_agg: str = "pcgrad",
    jd_lr_scale: float = 0.5,
    # New orientation-invariant motif geometry losses
    w_motif_orient: float = 0.0,
    w_motif_lddt: float = 0.0,
    w_motif_cabeta_dir: float = 0.0,
    w_motif_pair_orient: float = 0.2,
    w_motif_cb_bins: float = 0.0,
):
    motif_positions_tuple = None
    motif_template_ca = None
    motif_sidechains_tpl = None
    if motif_pdb_path and motif_chain_id and motif_resnums:
        motif_template_ca = base._build_motif_from_pdb(
            pdb_path=motif_pdb_path,
            chain_id=str(motif_chain_id),
            residue_numbers=tuple(int(x) for x in motif_resnums),
        )
        sc_list = base._build_motif_sidechains_from_pdb(
            pdb_path=motif_pdb_path,
            chain_id=str(motif_chain_id),
            residue_numbers=tuple(int(x) for x in motif_resnums),
        )
        motif_sidechains_tpl = tuple(sc_list)
    if motif_template_ca is not None and supervised_positions and len(supervised_positions) == int(motif_template_ca.shape[0]):
        motif_positions_tuple = tuple(int(x) for x in supervised_positions)

    def _triad_from_roles(mp: Tuple[int, ...], roles: Tuple[str, ...] | None) -> Tuple[int, int, int]:
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

    # CD-style transforms + JD convergence instrumentation
    def _pre_logits_temperature():
        def fn(logits, ctx):
            sched = (ctx or {}).get("schedule", {})
            t = float(sched.get("temp", sched.get("temperature", 1.0)))
            t = float(jnp.maximum(1e-2, t))
            return logits / t
        return fn

    def _pre_logits_flatten():
        def fn(logits, ctx):
            sched = (ctx or {}).get("schedule", {})
            e = float(sched.get("e_soft", 1.0))
            return logits * e
        return fn

    def _post_logits_sanitize():
        def fn(logits, ctx):
            x = jnp.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            return jnp.clip(x, -20.0, 20.0)
        return fn

    # Loss assembly (reusing base components)
    excl = tuple(supervised_positions) if supervised_positions else ()
    aux = float(w_contact) * base.ContactLoss(cutoff=14.0, binary=False, num=2, num_pos=1, seqsep=9, exclude_positions=excl)
    conf = base._sum_losses([
        (float(w_plddt) * base.PLDDTLoss(exclude_positions=excl)) if float(w_plddt) != 0.0 and bool(use_af2) else None,
        (float(w_pae) * base.PAELoss(seqsep=9, exclude_positions=excl)) if float(w_pae) != 0.0 and bool(use_af2) else None,
    ])

    from mosaic.losses.structure_prediction import HelixLoss as _HelixLoss
    from mosaic.losses.structure_prediction import HelixCapAtSerLoss as _HelixCapAtSerLoss
    helix_term = float(w_helix) * _HelixLoss()
    helix_cap_term = base._zero_loss()
    if float(w_helix_cap) != 0.0 and supervised_positions is not None and len(supervised_positions) > 0 and motif_roles is not None:
        # Determine catalytic ser index in binder coordinates
        mp_tmp = tuple(int(x) for x in (motif_positions_tuple or ())) if motif_positions_tuple is not None else tuple(int(x) for x in supervised_positions)
        try:
            s_i, _, _ = _triad_from_roles(mp_tmp, motif_roles)
            helix_cap_term = float(w_helix_cap) * _HelixCapAtSerLoss(ser_index=int(s_i), upstream_len=3, downstream_len=2, contact_dist=6.0, weight_downstream=1.0)
        except Exception:
            helix_cap_term = base._zero_loss()
    seq_ent = float(w_seq_ent) * base.SeqEntropyLoss(exclude_positions=cat_positions)

    def motif_geo(include_cat: bool = True, include_af2_sc: bool = True):
        if motif_template_ca is not None and motif_positions_tuple is not None:
            mp = tuple(int(x) for x in motif_positions_tuple)
            mt = motif_template_ca
            terms: List[base.LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms.append(float(w_motif_cce) * base.MotifDistogramCCE(motif_positions=mp, motif_template_ca=mt))
            if float(w_motif_rmsd) != 0.0:
                rmsd_term = base.MotifRMSDCA(motif_positions=mp, motif_template_ca=mt)
                terms.append(float(w_motif_rmsd) * base.ClippedLoss(loss=rmsd_term, l=0.0, u=10.0, name="motif_rmsd_clip"))
            # Orientation/graph losses (rotation-invariant)
            if float(w_motif_orient) != 0.0:
                terms.append(float(w_motif_orient) * MotifAngleGraphLoss(motif_positions=mp, motif_template_ca=mt))
            if float(w_motif_lddt) != 0.0:
                terms.append(float(w_motif_lddt) * MotifMotifLDDTLoss(motif_positions=mp, motif_template_ca=mt, radius=12.0))
            if float(w_motif_pair_orient) != 0.0:
                terms.append(float(w_motif_pair_orient) * MotifInterVecInFrameLoss(motif_positions=mp, motif_template_ca=mt))
            if float(w_motif_cabeta_dir) != 0.0:
                if motif_sidechains_tpl is None:
                    raise ValueError("w_motif_cabeta_dir requires motif_sidechains_tpl from template PDB")
                terms.append(float(w_motif_cabeta_dir) * MotifCABetaDirLoss(motif_positions=mp, motif_template_ca=mt, motif_sidechains_tpl=motif_sidechains_tpl))
            if float(w_motif_cb_bins) != 0.0:
                if motif_sidechains_tpl is None:
                    raise ValueError("w_motif_cb_bins requires motif_sidechains_tpl from template PDB")
                terms.append(float(w_motif_cb_bins) * MotifCBDistanceBinsLoss(motif_positions=mp, motif_template_ca=mt, motif_sidechains_tpl=motif_sidechains_tpl, thresholds=(0.5,1.0,2.0,4.0)))
            if motif_sidechains_tpl is not None and float(w_sc_rmsd) != 0.0:
                if include_af2_sc and use_af2:
                    terms.append(float(w_sc_rmsd) * base.AF2SidechainRMSD(positions=mp))
                elif not use_af2:
                    terms.append(float(w_sc_rmsd) * base.MotifSidechainRMSD(motif_positions=mp, motif_sidechains=motif_sidechains_tpl))
            if include_cat and float(w_cat_dist) != 0.0:
                s_i, h_i, a_i = _triad_from_roles(mp, motif_roles)
                terms.append(float(w_cat_dist) * base.CatalyticProximityCA(ser_idx=s_i, his_idx=h_i, asp_idx=a_i))
            return base._sum_losses(terms)
        if motif_template_ca is not None and motif_positions_tuple is None:
            terms_auto: List[base.LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms_auto.append(float(w_motif_cce) * base.MotifAutoDistogramCCE(motif_template_ca=motif_template_ca, beta=10.0))
            return base._sum_losses(terms_auto)
        if any(float(x) != 0.0 for x in (w_motif_cce, w_motif_rmsd, w_sc_rmsd, w_cat_dist)):
            raise ValueError("Motif losses requested but motif template/positions are not configured")
        return base._zero_loss()

    def motif_only_loss_term():
        if motif_template_ca is not None and motif_positions_tuple is not None:
            mp = tuple(int(x) for x in motif_positions_tuple)
            mt = motif_template_ca
            terms: List[base.LossTerm] = []
            if float(w_motif_cce) != 0.0:
                terms.append(float(w_motif_cce) * base.MotifDistogramCCE(motif_positions=mp, motif_template_ca=mt))
            if float(w_motif_rmsd) != 0.0:
                rmsd_term = base.MotifRMSDCA(motif_positions=mp, motif_template_ca=mt)
                terms.append(float(w_motif_rmsd) * base.ClippedLoss(loss=rmsd_term, l=0.0, u=10.0, name="motif_rmsd_clip"))
            if motif_sidechains_tpl is not None and float(w_sc_rmsd) != 0.0 and not use_af2:
                sc_term: base.LossTerm = base.MotifSidechainRMSD(motif_positions=mp, motif_sidechains=motif_sidechains_tpl)
                terms.append(float(w_sc_rmsd) * base.ClippedLoss(loss=sc_term, l=0.0, u=10.0, name="motif_sc_rmsd_clip"))
            return base._sum_losses(terms)
        return base._zero_loss()

    # Priors
    mpnn = base.ProteinMPNN.from_pretrained()
    mpnn_prior = base.InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.asarray(0.05), num_samples=8, jacobi_iterations=8)
    sequence_prior_term = base.ClippedLoss(loss=5.0 * mpnn_prior, l=-np.inf, u=100.0, name="mpnn_clipped")
    no_cys = 0.1 * base.NoCysteine()
    rg_term = float(w_rg) * base.MaskedDistogramRadiusOfGyration(exclude_positions=excl)

    # Build a single fixed-order list of component terms (unweighted). We will swap weights per phase.
    from mosaic.common import LinearCombination as _LC

    # Individual AF2 confidence terms (keep structure stable)
    plddt_term = base.PLDDTLoss(exclude_positions=excl) if bool(use_af2) else base._zero_loss()
    pae_term = base.PAELoss(seqsep=9, exclude_positions=excl) if bool(use_af2) else base._zero_loss()

    # Motif terms expanded individually (keep present even if weight=0)
    motif_terms_list: List[base.LossTerm] = []
    if motif_template_ca is not None and motif_positions_tuple is not None:
        mp = tuple(int(x) for x in motif_positions_tuple)
        mt = motif_template_ca
        motif_terms_list.append(base.MotifDistogramCCE(motif_positions=mp, motif_template_ca=mt))             # idx: motif_cce
        motif_rmsd_raw = base.MotifRMSDCA(motif_positions=mp, motif_template_ca=mt)
        motif_terms_list.append(base.ClippedLoss(loss=motif_rmsd_raw, l=0.0, u=10.0, name="motif_rmsd_clip")) # idx: motif_rmsd
        motif_terms_list.append(MotifAngleGraphLoss(motif_positions=mp, motif_template_ca=mt))                 # idx: motif_orient
        motif_terms_list.append(MotifMotifLDDTLoss(motif_positions=mp, motif_template_ca=mt, radius=12.0))     # idx: motif_lddt
        motif_terms_list.append(MotifInterVecInFrameLoss(motif_positions=mp, motif_template_ca=mt))            # idx: motif_pair_orient
        if motif_sidechains_tpl is not None:
            motif_terms_list.append(MotifCABetaDirLoss(motif_positions=mp, motif_template_ca=mt, motif_sidechains_tpl=motif_sidechains_tpl)) # idx: motif_cabeta_dir
            motif_terms_list.append(MotifCBDistanceBinsLoss(motif_positions=mp, motif_template_ca=mt, motif_sidechains_tpl=motif_sidechains_tpl, thresholds=(0.5,1.0,2.0,4.0))) # idx: motif_cb_bins
        else:
            motif_terms_list.append(base._zero_loss())  # motif_cabeta_dir placeholder
            motif_terms_list.append(base._zero_loss())  # motif_cb_bins placeholder
        # Sidechain RMSD (AF2-backed)
        motif_terms_list.append(base.AF2SidechainRMSD(positions=mp) if bool(use_af2) else base._zero_loss())   # idx: sc_rmsd
        # Catalytic proximity (always include; weight can be 0)
        s_i, h_i, a_i = _triad_from_roles(mp, motif_roles) if motif_roles is not None else (0, 0, 0)
        motif_terms_list.append(base.CatalyticProximityCA(ser_idx=s_i, his_idx=h_i, asp_idx=a_i))              # idx: cat_dist
        # FAPE term
        motif_terms_list.append(base.AF2MotifFAPE(positions=mp, clamp=10.0) if bool(use_af2) else base._zero_loss()) # idx: fape
    else:
        # Placeholders to keep structure consistent
        motif_terms_list.extend([
            base._zero_loss(),  # motif_cce
            base._zero_loss(),  # motif_rmsd
            base._zero_loss(),  # motif_orient
            base._zero_loss(),  # motif_lddt
            base._zero_loss(),  # motif_pair_orient
            base._zero_loss(),  # motif_cabeta_dir
            base._zero_loss(),  # motif_cb_bins
            base._zero_loss(),  # sc_rmsd
            base._zero_loss(),  # cat_dist
            base._zero_loss(),  # fape
        ])

    # Assemble full fixed term list
    terms_all: List[base.LossTerm] = [
        aux,               # 0 contact
        seq_ent,           # 1 sequence entropy
        helix_term,        # 2 helix suppression/encourage
        helix_cap_term,    # 3 helix N-cap at Ser
        rg_term,           # 4 radius of gyration
        plddt_term,        # 5 AF2 pLDDT
        pae_term,          # 6 AF2 PAE
    ] + motif_terms_list + [
        sequence_prior_term, # last-2: mpnn prior (clipped)
        no_cys,              # last-1: no cysteine prior
    ]

    # Indices (for readability)
    idx_contact = 0
    idx_seqent = 1
    idx_helix = 2
    idx_helixcap = 3
    idx_rg = 4
    idx_plddt = 5
    idx_pae = 6
    # Motif block starts at 7
    idx_motif_cce = 7
    idx_motif_rmsd = 8
    idx_motif_orient = 9
    idx_motif_lddt = 10
    idx_motif_pair_orient = 11
    idx_motif_cabeta = 12
    idx_motif_cb_bins = 13
    idx_sc_rmsd = 14
    idx_cat_dist = 15
    idx_fape = 16
    idx_mpnn = 17
    idx_nocys = 18

    # Phase weight vectors (same length as terms_all)
    def _zeros():
        return jnp.zeros((len(terms_all),), dtype=jnp.float32)

    w_warm = _zeros()
    w_warm = w_warm.at[idx_contact].set(float(w_contact))
    w_warm = w_warm.at[idx_seqent].set(float(w_seq_ent))
    w_warm = w_warm.at[idx_helix].set(float(w_helix))
    w_warm = w_warm.at[idx_helixcap].set(float(w_helix_cap))
    w_warm = w_warm.at[idx_rg].set(float(w_rg))
    w_warm = w_warm.at[idx_plddt].set(float(w_plddt))
    w_warm = w_warm.at[idx_pae].set(float(w_pae))
    w_warm = w_warm.at[idx_motif_cce].set(float(w_motif_cce))
    w_warm = w_warm.at[idx_motif_rmsd].set(float(w_motif_rmsd))
    w_warm = w_warm.at[idx_motif_orient].set(float(w_motif_orient))
    w_warm = w_warm.at[idx_motif_lddt].set(float(w_motif_lddt))
    w_warm = w_warm.at[idx_motif_pair_orient].set(float(w_motif_pair_orient))
    w_warm = w_warm.at[idx_motif_cabeta].set(float(w_motif_cabeta_dir))
    w_warm = w_warm.at[idx_motif_cb_bins].set(float(w_motif_cb_bins))
    w_warm = w_warm.at[idx_sc_rmsd].set(float(w_sc_rmsd))
    w_warm = w_warm.at[idx_cat_dist].set(0.0)
    w_warm = w_warm.at[idx_fape].set(0.0)
    w_warm = w_warm.at[idx_mpnn].set(1.0)
    w_warm = w_warm.at[idx_nocys].set(1.0)

    w_soft = w_warm.at[idx_fape].set(float(w_fape))

    w_anneal = w_soft.at[idx_cat_dist].set(float(w_cat_dist))

    # Single AF2-backed loss with fixed structure
    combined = _LC(weights=w_warm, l=terms_all)
    struct_all = base.ClippedGradient(loss=combined, max_norm=1.0)  # type: ignore[arg-type]

    # Build predictor-backed losses (single compiled graph reused across phases)
    loss: base.LossTerm
    if use_af2:
        model = base.AF2Model(data_dir=af2_params_dir or ".")
        use_partial_template = motif_template_ca is not None and motif_positions_tuple is not None
        if use_partial_template:
            import importlib
            gemmi = importlib.import_module("gemmi")
            chain = gemmi.Chain("A")
            for i in range(int(binder_len)):
                res = gemmi.Residue(); res.name = "GLY"; res.seqid = gemmi.SeqId(int(i + 1), " "); chain.add_residue(res)
            bb_coords = {}
            if motif_pdb_path is not None and motif_positions_tuple is not None:
                st = gemmi.read_structure(str(motif_pdb_path))
                ch = st[0][str(motif_chain_id) if motif_chain_id is not None else "A"]
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
                for idx_local, rn in enumerate(tuple(int(x) for x in (motif_resnums or ()) ) ):
                    if rn in rn_to_bb:
                        bb_coords[idx_local] = rn_to_bb[rn]
            if motif_template_ca is not None and motif_positions_tuple is not None:
                for idx_local, binder_pos in enumerate(motif_positions_tuple):
                    if 0 <= int(binder_pos) < int(binder_len):
                        res = chain[int(binder_pos)]
                        if motif_sidechains_tpl is not None and idx_local < len(motif_sidechains_tpl):
                            resname, _, _ = motif_sidechains_tpl[idx_local]
                            res.name = str(resname)
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
                        if motif_sidechains_tpl is not None and idx_local < len(motif_sidechains_tpl):
                            _, atom_names, coords = motif_sidechains_tpl[idx_local]
                            for nm, q in zip(atom_names, coords):
                                if nm in ("N", "CA", "C", "O"):
                                    continue
                                a2 = gemmi.Atom(); a2.name = str(nm)
                                a2.pos.x, a2.pos.y, a2.pos.z = float(q[0]), float(q[1]), float(q[2])
                                res.add_atom(a2)
            # Use upstream AF2 binder_features: binder + target chain with matching sequence
            target_seq = gemmi.one_letter_code([r.name for r in chain])
            feats, _ = model.binder_features(int(binder_len), chains=[base.TargetChain(sequence=target_seq, use_msa=False, template_chain=chain)])
        else:
            feats, _ = model.binder_features(int(binder_len), chains=[])
        loss_all = cast(base.LossTerm, model.build_loss(loss=struct_all, features=feats, recycling_steps=int(af2_num_recycles), model_idx=(int(af2_model_idx) if af2_model_idx is not None else None)))
        # Phase-specific views by swapping only the weights leaf. Keep structure identical to avoid recompiles.
        import equinox as eqx
        loss_warmup = eqx.tree_at(lambda m: m.loss.loss.weights, loss_all, w_warm)
        loss_soft = eqx.tree_at(lambda m: m.loss.loss.weights, loss_all, w_soft)
        loss_anneal = eqx.tree_at(lambda m: m.loss.loss.weights, loss_all, w_anneal)

        # Prime JIT once for all AF2-backed losses to avoid per-phase compile pauses
        def _prime_jit(loss_term: base.LossTerm):
            try:
                probs0 = jnp.full((int(binder_len), 20), 1.0 / 20.0, dtype=jnp.float32)
                key0 = jax.random.PRNGKey(0)
                _ = loss_term(probs0, key=key0)
            except Exception:
                pass
        _prime_jit(loss_warmup)
        _prime_jit(loss_soft)
        _prime_jit(loss_anneal)
    else:
        ligand = tmol_context.get("ligand", {})
        tpl_pdb = str(motif_pdb_path) if motif_pdb_path else None
        tpl_chain = str(motif_chain_id) if motif_chain_id else None
        # Non-AF2 path: build a simpler fixed-structure base loss and wrap with Boltz2
        base_loss_nonaf2 = base.ClippedGradient(loss=base._sum_losses([
            aux, seq_ent, helix_term, helix_cap_term, rg_term, sequence_prior_term, no_cys
        ]), max_norm=1.0)
        loss_full = cast(base.LossTerm, base._build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=base_loss_nonaf2,
            template_pdb_path=tpl_pdb,
            template_chain_id=tpl_chain,
        ))
        base_loss_with_cat = base.ClippedGradient(loss=base._sum_losses([
            aux, seq_ent, conf, helix_term, helix_cap_term, rg_term,
            (base.CatalyticProximityCA(ser_idx=cat_positions[0], his_idx=cat_positions[1], asp_idx=cat_positions[2]) if len(cat_positions)==3 else base._zero_loss()),
            sequence_prior_term, no_cys
        ]), max_norm=1.0)
        loss_full_with_cat = cast(base.LossTerm, base._build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=base_loss_with_cat,
            template_pdb_path=tpl_pdb,
            template_chain_id=tpl_chain,
        ))
        loss_motif_only = cast(base.LossTerm, base._build_boltz2_loss(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            base_loss=motif_only_loss_term(),
            template_pdb_path=tpl_pdb,
            template_chain_id=tpl_chain,
        ))
        loss_warmup = loss_motif_only

    # JD optimizer and convergence transforms
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

    # Convergence: exclude supervised (often fixed) positions to avoid trivial early stop
    conv_mask = None
    conv_key = "probs_max_mean"
    if supervised_positions is not None and len(supervised_positions) > 0:
        conv_mask = np.ones((int(binder_len),), dtype=np.float32)
        for p in supervised_positions:
            if 0 <= int(p) < int(binder_len):
                conv_mask[int(p)] = 0.0
        conv_key = "probs_max_mean_designable"
    # Germinal-style convergence check on masked positions, threshold=0.1
    pre_probs_chain = pre_probs_chain + [germinal_softmax_convergence(mask=conv_mask, threshold=0.10, key=conv_key)]

    if fix_supervised_identities and supervised_positions is not None and len(supervised_positions) > 0:
        vocab = "ARNDCQEGHILKMFPSTWYV"
        allowed = np.ones((int(binder_len), 20), dtype=np.float32)
        ids = [s.strip().upper() for s in str(fix_supervised_identities).split(',') if s.strip()]
        for i, sup_pos in enumerate(tuple(int(x) for x in supervised_positions)):
            if i < len(ids) and ids[i] in vocab and 0 <= int(sup_pos) < int(binder_len):
                allowed[int(sup_pos), :] = 0.0
                allowed[int(sup_pos), vocab.index(ids[i])] = 1.0
        post_logits_chain = post_logits_chain + [per_position_allowed_tokens(allowed)]
        pre_probs_chain = pre_probs_chain + [per_position_allowed_probs(allowed)]

    total = max(1, int(steps))
    warmup_steps = max(1, int(round(total * 0.20)))
    soft_steps = max(1, int(round(total * 0.60)))
    anneal_steps = max(1, int(total - (warmup_steps + soft_steps)))

    def phase_dict(name: str, build_loss, n_steps: int, temperature: float, e_soft: float, anneal: bool = False, scale: float = 1.1):
        agg = jd_pcgrad_aggregator if str(jd_agg).lower() == "pcgrad" else jd_upgrad_aggregator
        return {
            "name": name,
            "build_loss": build_loss,
            # Prefer caller-supplied optimizer else JD+PCGrad to damp oscillations
            "optimizer": (optimizer or partial(jacobian_descent_adapter, aggregator=agg)),
            "steps": int(n_steps),
            "schedule": (lambda g, p: {
                # Smaller steps for JD to reduce oscillations
                "lr": float(lr) * float(jd_lr_scale),
                "stepsize": float(apgm_stepsize_coef) * float(jnp.sqrt(jnp.maximum(1, binder_len))),
                "scale": float(scale),
                "temperature": float(jnp.maximum(0.05, (temperature if not anneal else temperature * jnp.exp(-3.0 * (g / jnp.maximum(1.0, n_steps)))))),
                "e_soft": float(e_soft),
                # Gate convergence stopping until a few steps into a phase
                "min_stop_step": 5,
            }),
            "transforms": {
                "pre_logits": pre_logits_chain,
                "pre_probs": pre_probs_chain,
                "grad": grad_chain_late if name != "motif_lock" and bool(freeze_supervised_positions) else grad_chain_warm,
                "post_logits": post_logits_chain + [_post_logits_sanitize()],
            },
            "analyzers": [],
            "analyze_every": 1,
        }

    phases = [
        phase_dict("motif_lock", lambda: loss_warmup, warmup_steps, temperature=1.0, e_soft=0.8, anneal=False, scale=1.1),
        phase_dict("soft",       lambda: loss_soft,   soft_steps,  temperature=1.0, e_soft=0.8, anneal=False, scale=1.1),
        phase_dict("anneal",     lambda: loss_anneal, anneal_steps, temperature=1.0, e_soft=1.0, anneal=True,  scale=1.5),
    ]

    return {"phases": phases, "binder_len": int(binder_len), "seed": 0}


def run(binder_len: int, tmol_context: dict, supervised_positions: Tuple[int, ...] | None = None, motif_pdb_path: str | Path | None = None, motif_chain_id: str | None = None, motif_resnums: Tuple[int, ...] | None = None, initial_x: np.ndarray | None = None):
    wf = make_workflow(binder_len=binder_len, tmol_context=tmol_context, supervised_positions=supervised_positions, motif_pdb_path=motif_pdb_path, motif_chain_id=motif_chain_id, motif_resnums=motif_resnums)
    wf["initial_x"] = initial_x if initial_x is not None else np.random.randn(binder_len, 20).astype(np.float32) * 0.1
    from .design import run_workflow
    return run_workflow(wf)


# ---------- New rotation-invariant motif geometry losses ----------

class MotifAngleGraphLoss(base.LossTerm):
    """Compare motif geometry via angles between inter-residue vectors (rotation-invariant).

    For each center residue c in motif, take all distinct pairs (a,b) from motif\{c} and compare
    the angle at c between vectors (a->c) and (b->c) to the template angle.
    """
    motif_positions: Tuple[int, ...]
    motif_template_ca: jax.Array
    def __init__(self, *, motif_positions: Tuple[int, ...], motif_template_ca: np.ndarray):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))

    @staticmethod
    def _get_ca_coords(output, L: int) -> jax.Array:
        # Prefer generic backbone_coordinates if present ([L, atoms?, 3] with CA at index 1)
        try:
            ca = jnp.nan_to_num(output.backbone_coordinates[:L, 1, :])
            return ca
        except Exception:
            # AF2 fallback
            from mosaic.alphafold.common import residue_constants as rc  # type: ignore
            all37 = jnp.nan_to_num(output.output.structure_module.final_atom_positions[:L])
            return all37[:, rc.atom_order["CA"], :]

    @staticmethod
    def _angle(u: jax.Array, v: jax.Array) -> jax.Array:
        u = u / (jnp.linalg.norm(u) + 1e-8)
        v = v / (jnp.linalg.norm(v) + 1e-8)
        cos = jnp.clip(jnp.sum(u * v), -1.0, 1.0)
        return jnp.arccos(cos)

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        mp = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        K = mp.shape[0]
        ca_pred = self._get_ca_coords(output, L)
        ca_tpl = self.motif_template_ca  # [K,3] assumed template order matches mp order

        def angles_for_center(coords: jax.Array):
            # coords: [K,3] in motif order
            def angle_at(c_idx):
                c = coords[c_idx]
                others = jnp.arange(K)
                others = others[others != c_idx]
                # accumulate mean angle error over all unordered pairs of others
                def pair_angle(ix):
                    a_idx = others[ix // (K - 1 - 0)]  # not used; we'll vectorize below
                    return a_idx
                return c  # placeholder (unused)
            return coords  # placeholder

        # Compute mean absolute angle difference across all centers and unordered pairs
        def mean_angle_diff(coords_pred: jax.Array, coords_tpl: jax.Array) -> jax.Array:
            diffs = []
            for ci in range(int(K)):
                c_p = coords_pred[ci]
                c_t = coords_tpl[ci]
                others = [int(x) for x in range(int(K)) if x != ci]
                for i in range(len(others)):
                    for j in range(i + 1, len(others)):
                        a, b = others[i], others[j]
                        ang_p = self._angle(coords_pred[a] - c_p, coords_pred[b] - c_p)
                        ang_t = self._angle(coords_tpl[a] - c_t, coords_tpl[b] - c_t)
                        diffs.append(jnp.abs(ang_p - ang_t))
            if len(diffs) == 0:
                return jnp.asarray(0.0, dtype=jnp.float32)
            return jnp.mean(jnp.stack(diffs))

        # Gather predicted motif CA in motif order
        ca_pred_motif = ca_pred[mp]
        val = mean_angle_diff(ca_pred_motif, ca_tpl)
        return val, {"motif_angle_graph": val}


class MotifMotifLDDTLoss(base.LossTerm):
    """Local LDDT-style loss restricted to motif residues (superposition-free).

    For each motif residue, compare pairwise CA distances within the motif neighborhood and score
    across thresholds {0.5,1,2,4} Å; minimize 1 - mean score.
    """
    motif_positions: Tuple[int, ...]
    motif_template_ca: jax.Array
    radius: float
    def __init__(self, *, motif_positions: Tuple[int, ...], motif_template_ca: np.ndarray, radius: float = 12.0):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))
        object.__setattr__(self, "radius", float(radius))

    @staticmethod
    def _get_ca_coords(output, L: int) -> jax.Array:
        try:
            return jnp.nan_to_num(output.backbone_coordinates[:L, 1, :])
        except Exception:
            from mosaic.alphafold.common import residue_constants as rc  # type: ignore
            all37 = jnp.nan_to_num(output.output.structure_module.final_atom_positions[:L])
            return all37[:, rc.atom_order["CA"], :]

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        mp = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        K = mp.shape[0]
        ca_pred = self._get_ca_coords(output, L)[mp]
        ca_tpl = self.motif_template_ca  # [K,3]

        # pairwise distances within motif
        def pdist(x):
            diff = x[:, None, :] - x[None, :, :]
            return jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-8)

        Dp = pdist(ca_pred)
        Dt = pdist(ca_tpl)
        # mask neighborhood by radius around each center in template space
        M = (Dt <= float(self.radius)) & (~jnp.eye(K, dtype=bool))
        d = jnp.abs(Dp - Dt)
        # LDDT-like per-center average over neighbors j
        thr = jnp.asarray([0.5, 1.0, 2.0, 4.0], dtype=Dp.dtype)
        s = (d[..., None] < thr).astype(Dp.dtype).mean(axis=-1)  # [K,K]
        denom = jnp.maximum(jnp.sum(M, axis=-1), 1.0)
        per_center = jnp.sum(s * M.astype(Dp.dtype), axis=-1) / denom
        lddt_mean = jnp.mean(per_center)
        loss = 1.0 - lddt_mean
        return loss, {"motif_lddt": lddt_mean}


class _BackboneFrames:
    @staticmethod
    def get_backbone(output, L: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Prefer generic backbone_coordinates with N(0), CA(1), C(2)
        X = output.backbone_coordinates[:L]
        N = jnp.nan_to_num(X[:, 0, :])
        CA = jnp.nan_to_num(X[:, 1, :])
        C = jnp.nan_to_num(X[:, 2, :])
        return N, CA, C

    @staticmethod
    def frame_from_ncac(N: jax.Array, CA: jax.Array, C: jax.Array) -> jax.Array:
        # Returns rotation matrix [3,3] with columns (e1,e2,e3)
        e1 = C - CA; e1 = e1 / (jnp.linalg.norm(e1) + 1e-8)
        v2 = N - CA
        e2 = v2 - jnp.dot(v2, e1) * e1
        e2 = e2 / (jnp.linalg.norm(e2) + 1e-8)
        e3 = jnp.cross(e1, e2)
        return jnp.stack([e1, e2, e3], axis=-1)

    @staticmethod
    def frame_stack(N: jax.Array, CA: jax.Array, C: jax.Array) -> jax.Array:
        # [L,3,3]
        def one(i):
            return _BackboneFrames.frame_from_ncac(N[i], CA[i], C[i])
        return jax.vmap(one)(jnp.arange(N.shape[0]))

    @staticmethod
    def virtual_cb(N: jax.Array, CA: jax.Array, C: jax.Array) -> jax.Array:
        # Construct pseudo-CB direction from N,CA,C triad (unit length)
        v1 = N - CA
        v2 = C - CA
        v1 = v1 / (jnp.linalg.norm(v1) + 1e-8)
        v2 = v2 / (jnp.linalg.norm(v2) + 1e-8)
        b = (v1 + v2); b = b / (jnp.linalg.norm(b) + 1e-8)
        n = jnp.cross(v1, v2); n = n / (jnp.linalg.norm(n) + 1e-8)
        # combination coefficients approximate tetrahedral geometry; direction only
        vc = -0.58273431 * b + 0.56802827 * n
        return vc / (jnp.linalg.norm(vc) + 1e-8)


class MotifCABetaDirLoss(base.LossTerm):
    motif_positions: Tuple[int, ...]
    motif_template_ca: jax.Array
    motif_sidechains_tpl: Tuple[Tuple[str, Tuple[str, ...], jax.Array], ...]
    def __init__(self, *, motif_positions: Tuple[int, ...], motif_template_ca: np.ndarray, motif_sidechains_tpl: Tuple[Tuple[str, Tuple[str, ...], np.ndarray], ...]):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))
        packed = []
        for resname, names, coords in motif_sidechains_tpl:
            packed.append((str(resname), tuple(names), jnp.asarray(coords, dtype=jnp.float32)))
        object.__setattr__(self, "motif_sidechains_tpl", tuple(packed))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        mp = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        K = int(mp.shape[0])
        N, CA, C = _BackboneFrames.get_backbone(output, L)
        N_m, CA_m, C_m = N[mp], CA[mp], C[mp]
        vcb_pred = _BackboneFrames.virtual_cb(N_m, CA_m, C_m)  # [K,3]

        # Template sidechain centroid direction from CA
        present = []
        vcb_tpl = []
        for idx in range(K):
            _, atom_names, coords = self.motif_sidechains_tpl[idx]
            if coords.shape[0] == 0:
                present.append(False)
                continue
            centroid = jnp.mean(coords, axis=0)
            dirv = centroid - self.motif_template_ca[idx]
            dirv = dirv / (jnp.linalg.norm(dirv) + 1e-8)
            vcb_tpl.append(dirv)
            present.append(True)

        num_present = int(sum(1 for b in present if b))
        if num_present == 0:
            # No sidechain template info; make this term a no-op
            loss = jnp.asarray(0.0, dtype=jnp.float32)
            return loss, {"motif_cabeta_dir": 1.0 - loss}

        idxs = jnp.asarray([i for i, b in enumerate(present) if b], dtype=jnp.int32)
        vcb_pred_sel = vcb_pred[idxs]
        vcb_tpl = jnp.stack(vcb_tpl, axis=0)

        cos = jnp.sum(vcb_pred_sel * vcb_tpl, axis=-1)
        loss = jnp.mean(1.0 - cos)
        return loss, {"motif_cabeta_dir": 1.0 - loss}


class MotifInterVecInFrameLoss(base.LossTerm):
    motif_positions: Tuple[int, ...]
    motif_template_ca: jax.Array
    def __init__(self, *, motif_positions: Tuple[int, ...], motif_template_ca: np.ndarray):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        mp = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        K = int(mp.shape[0])
        N, CA, C = _BackboneFrames.get_backbone(output, L)
        N_m, CA_m, C_m = N[mp], CA[mp], C[mp]
        R = _BackboneFrames.frame_stack(N_m, CA_m, C_m)  # [K,3,3]

        ca_tpl = self.motif_template_ca  # [K,3]
        # Compare unit inter-residue vectors expressed in the local frame of i
        diffs = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                v_pred = CA_m[j] - CA_m[i]
                v_pred = v_pred / (jnp.linalg.norm(v_pred) + 1e-8)
                v_pred_f = R[i].T @ v_pred
                v_tpl = ca_tpl[j] - ca_tpl[i]
                v_tpl = v_tpl / (jnp.linalg.norm(v_tpl) + 1e-8)
                # Build a template frame at i from three motif CA points if available? Not available; compare raw vector in global template coords to predicted frame vector via cosine only using direction
                # Project template vector into predicted frame for direction comparison
                v_tpl_f = R[i].T @ v_tpl
                diffs.append(1.0 - jnp.sum(v_pred_f * v_tpl_f) / (jnp.linalg.norm(v_pred_f) * jnp.linalg.norm(v_tpl_f) + 1e-8))
        loss = jnp.mean(jnp.stack(diffs)) if len(diffs) > 0 else jnp.asarray(0.0, dtype=jnp.float32)
        return loss, {"motif_pair_orient": 1.0 - loss}


class MotifCBDistanceBinsLoss(base.LossTerm):
    motif_positions: Tuple[int, ...]
    motif_template_ca: jax.Array
    motif_sidechains_tpl: Tuple[Tuple[str, Tuple[str, ...], jax.Array], ...]
    thresholds: Tuple[float, float, float, float]
    def __init__(self, *, motif_positions: Tuple[int, ...], motif_template_ca: np.ndarray, motif_sidechains_tpl: Tuple[Tuple[str, Tuple[str, ...], np.ndarray], ...], thresholds: Tuple[float, float, float, float] = (0.5,1.0,2.0,4.0)):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))
        packed = []
        for resname, names, coords in motif_sidechains_tpl:
            packed.append((str(resname), tuple(names), jnp.asarray(coords, dtype=jnp.float32)))
        object.__setattr__(self, "motif_sidechains_tpl", tuple(packed))
        object.__setattr__(self, "thresholds", tuple(float(x) for x in thresholds))

    def __call__(self, sequence, output, key):
        L = sequence.shape[0]
        mp = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        K = int(mp.shape[0])
        N, CA, C = _BackboneFrames.get_backbone(output, L)
        N_m, CA_m, C_m = N[mp], CA[mp], C[mp]
        # predicted CB pseudo positions
        CB_pred = CA_m + _BackboneFrames.virtual_cb(N_m, CA_m, C_m)

        # template CB centroids
        present = []
        CB_tpl = []
        for idx in range(K):
            _, atom_names, coords = self.motif_sidechains_tpl[idx]
            if coords.shape[0] == 0:
                present.append(False)
                continue
            centroid = jnp.mean(coords, axis=0)
            CB_tpl.append(centroid)
            present.append(True)

        idxs = jnp.asarray([i for i, b in enumerate(present) if b], dtype=jnp.int32)
        if idxs.shape[0] < 2:
            # Not enough residues with sidechains to form pairs; no-op
            loss = jnp.asarray(0.0, dtype=jnp.float32)
            return loss, {"motif_cb_bins": 1.0 - loss}

        CB_tpl = jnp.stack(CB_tpl, axis=0)
        CB_pred = CB_pred[idxs]

        def pdist(x):
            diff = x[:, None, :] - x[None, :, :]
            return jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-8)

        Dp = pdist(CB_pred)
        Dt = pdist(CB_tpl)
        Kp = int(CB_pred.shape[0])
        M = (~jnp.eye(Kp, dtype=bool))
        d = jnp.abs(Dp - Dt)
        thr = jnp.asarray(self.thresholds, dtype=Dp.dtype)
        s = (d[..., None] < thr).astype(Dp.dtype).mean(axis=-1)
        denom = jnp.maximum(jnp.sum(M, axis=-1), 1.0)
        per_center = jnp.sum(s * M.astype(Dp.dtype), axis=-1) / denom
        score = jnp.mean(per_center)
        loss = 1.0 - score
        return loss, {"motif_cb_bins": score}


