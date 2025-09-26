import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable

from .outer import run_many
from functools import lru_cache


def _add_bindcraft_paths(repo_root: str) -> None:
    src = Path(repo_root) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _bindcraft_dirs(base: str) -> Dict[str, str]:
    names = [
        "Accepted", "Trajectory", "MPNN", "Rejected", "AF2", "CSV", "logs", "settings",
        "Accepted/Ranked", "Accepted/Animation", "Accepted/Plots", "Accepted/Pickle",
        "Trajectory/Pickle", "Trajectory/Relaxed", "Trajectory/Plots", "Trajectory/Clashing",
        "Trajectory/LowConfidence", "Trajectory/Animation",
        "MPNN/Binder", "MPNN/Sequences", "MPNN/Relaxed",
        "AF2/Ranked",
    ]
    paths: Dict[str, str] = {}
    for n in names:
        p = Path(base) / n
        p.mkdir(parents=True, exist_ok=True)
        paths[n] = str(p)
    # canonical CSV names
    paths["trajectory_csv"] = str(Path(base) / "trajectory_stats.csv")
    paths["mpnn_csv"] = str(Path(base) / "mpnn_design_stats.csv")
    paths["final_csv"] = str(Path(base) / "final_design_stats.csv")
    paths["failure_csv"] = str(Path(base) / "failure_csv.csv")
    return paths


def _init_csvs(paths: Dict[str, str]) -> None:
    # Import BindCraft functions for headers and CSV creation
    from BindCraft.functions import generate_dataframe_labels, create_dataframe, generate_filter_pass_csv

    traj_labels, design_labels, final_labels = generate_dataframe_labels()
    create_dataframe(paths["trajectory_csv"], traj_labels)
    create_dataframe(paths["mpnn_csv"], design_labels)
    create_dataframe(paths["final_csv"], final_labels)
    # failure csv based on filter settings is created by generate_filter_pass_csv; we call it in emit


def _append_csv_row(csv_path: str, row: List[Any]) -> None:
    import csv
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def _make_design_name(binder_name: str, length: int, seed: int) -> str:
    return f"{binder_name}_l{length}_s{seed}"



def default_emit_row(*, kind: str, row: dict, paths: Dict[str, str], target_settings: Dict[str, Any], advanced_settings: Dict[str, Any], filters: Dict[str, Any]) -> None:
    """BindCraft-parity emitter: relax/DSSP/interface scoring + CSV writes.

    - kind: "parent" or "child"
    - row: {"spec", "best_sequence", "metrics", ...}
    - paths: directory and csv paths from _bindcraft_dirs
    """
    from BindCraft.functions import (
        openmm_relax, score_interface, calc_ss_percentage,
        generate_dataframe_labels, insert_data, generate_filter_pass_csv,
        calculate_averages, check_filters, update_failures,
        target_pdb_rmsd, unaligned_rmsd, save_fasta,
    )

    # Initialize failure CSV once
    generate_filter_pass_csv(paths["failure_csv"], advanced_settings.get("filters_json", ""))

    binder_name = target_settings.get("binder_name", target_settings.get("task_name", "design"))
    length = int(row.get("spec", {}).get("binder_len", 0))
    seed = int(row.get("spec", {}).get("seed", 0))
    design_name = _make_design_name(binder_name, length, seed)

    # Save FASTA for child rows when present
    seq = row.get("best_sequence")
    if seq:
        save_fasta(design_name, seq, paths)

    # Expect structure_path provided upstream or created by predictor-specific emit
    pdb_in = row.get("structure_path")

    # If no structure_path yet, do a quick AF/ColabDesign complex prediction for the parent
    if kind == "parent" and (not pdb_in) and seq:
        from BindCraft.functions import mk_afdesign_model
        # build prediction model and prep inputs
        # Force multimer for parent quick predict to ensure PAE is available
        model = mk_afdesign_model(
            protocol="binder",
            debug=False,
            data_dir=advanced_settings.get("af_params_dir", "/root/BindCraft/params"),
            use_multimer=True,
            num_recycles=advanced_settings.get("num_recycles_validation", 1),
        )
        model.prep_inputs(
            pdb_filename=target_settings.get("starting_pdb"),
            chain=target_settings.get("chains", "A"),
            binder_len=length,
            rm_target_seq=advanced_settings.get("rm_template_seq_predict", False),
            rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
        )
        # predict one model and save to Trajectory
        model.predict(seq=seq, models=[0], num_recycles=advanced_settings.get("num_recycles_validation", 1), verbose=False)
        traj_pdb = os.path.join(paths["Trajectory"], f"{design_name}.pdb")
        model.save_pdb(traj_pdb)
        pdb_in = traj_pdb
        row["structure_path"] = pdb_in
        # Extract AF2 metrics from model aux if available
        metrics: Dict[str, Any] = {}
        aux = getattr(model, "aux", {}) or {}
        def _mean(x):
            try:
                import numpy as _np
                return float(_np.asarray(x).mean())
            except Exception:
                return None
        if isinstance(aux, dict):
            if "plddt" in aux:
                metrics["predict.plddt_mean"] = _mean(aux.get("plddt"))
            if "ptm" in aux:
                try:
                    metrics["predict.ptm"] = float(aux.get("ptm"))
                except Exception:
                    pass
            if "iptm" in aux:
                try:
                    metrics["predict.i_ptm"] = float(aux.get("iptm"))
                except Exception:
                    pass
            if "pae" in aux:
                metrics["predict.pae_mean"] = _mean(aux.get("pae"))
        row["metrics"] = {**row.get("metrics", {}), **metrics}
    pdb_relaxed = None
    ss = None
    if pdb_in:
        relaxed_out = os.path.join(paths["Trajectory/Relaxed"], f"{design_name}.pdb")
        ok_relax = openmm_relax(pdb_in, relaxed_out)
        pdb_relaxed = relaxed_out if ok_relax else None
        row["pdb_relaxed"] = pdb_relaxed
        # sanitize PDB to avoid parser issues
        # Skip PDB cleaning due to sporadic file descriptor errors on remote volumes

    # DSSP + Interface scoring if relaxed available
    if pdb_relaxed and os.path.exists(pdb_relaxed) and os.path.getsize(pdb_relaxed) > 0:
        chain = advanced_settings.get("binder_chain", "B")
        ss = calc_ss_percentage(pdb_relaxed, advanced_settings, chain)
        row["dssp"] = ss
        # Attempt to pass PAE to enable ipSAE metrics
        pae_matrix = None
        try:
            from adaptyv_bindcraft.bindcraft_pipeline import extract_pae
            pae_matrix, pae_logits, breaks, chains = extract_pae(model if 'model' in locals() else row, length, chain)
        except Exception:
            pae_matrix = None
            chains = None
        iface_tuple = score_interface(
            pdb_relaxed,
            chain,
            pae_matrix=pae_matrix,
            chains=chains,
            pae_cutoff=advanced_settings.get("pae_cutoff", 10.0),
            pae_logits=None,
            breaks=None,
        )
        if isinstance(iface_tuple, tuple) and len(iface_tuple) >= 3:
            iface_scores, iface_AA, iface_residues = iface_tuple
            row["interface"] = iface_scores
            row["interface_AA"] = iface_AA
            row["interface_residues"] = iface_residues
        else:
            row["interface"] = iface_tuple
        # Compute clashes on parent (unrelaxed and relaxed)
        try:
            from BindCraft.functions.openmm_utils import calculate_clash_score, target_pdb_rmsd
            row["unrelaxed_clashes"] = calculate_clash_score(pdb_in)
            row["relaxed_clashes"] = calculate_clash_score(pdb_relaxed)
            row["target_rmsd"] = target_pdb_rmsd(pdb_relaxed, target_settings.get("starting_pdb"), target_settings.get("chains", "A"))
        except Exception:
            pass

    # Write to CSVs depending on kind
    traj_labels, design_labels, final_labels = generate_dataframe_labels()
    if kind == "parent":
        # Minimal trajectory row: fill known positions, leave others blank/zero
        # ['Design','Protocol','Length','Seed','Helicity','Target_Hotspot','Sequence','InterfaceResidues', ...]
        data = [
            design_name,
            advanced_settings.get("design_protocol", "Default"),
            length,
            seed,
            advanced_settings.get("weights_helicity", 0),
            target_settings.get("target_hotspot_residues", ""),
            seq or "",
            row.get("interface_residues", ""),
        ]
        # Fill confidence/interface/ss metrics
        metrics_map = {
            "pLDDT": row.get("metrics", {}).get("predict.plddt_mean"),
            "pTM": row.get("metrics", {}).get("predict.ptm"),
            "i_pTM": row.get("metrics", {}).get("predict.i_ptm"),
            "pAE": row.get("metrics", {}).get("predict.pae_mean"),
            "i_pAE": row.get("metrics", {}).get("predict.i_pae"),
            "i_pLDDT": row.get("metrics", {}).get("predict.i_plddt"),
            "ss_pLDDT": row.get("metrics", {}).get("predict.ss_plddt"),
        }
        data.extend([metrics_map.get(k) for k in ["pLDDT","pTM","i_pTM","pAE","i_pAE","i_pLDDT","ss_pLDDT"]])
        # Clashes pre/post relax (if present)
        data.extend([row.get("unrelaxed_clashes"), row.get("relaxed_clashes")])
        # Binder energy (from interface binder_score)
        iface_parent = row.get("interface", {}) or {}
        data.append(iface_parent.get('binder_score'))
        # Interface metrics (some may be None)
        iface = row.get("interface", {}) or {}
        data.extend([
            iface.get('surface_hydrophobicity'),
            iface.get('interface_sc'),
            iface.get('interface_packstat'),
            iface.get('interface_dG'),
            iface.get('interface_dSASA'),
            iface.get('interface_dG_SASA_ratio'),
            iface.get('interface_fraction'),
            iface.get('interface_hydrophobicity'),
            iface.get('interface_nres'),
            iface.get('interface_interface_hbonds'),
            iface.get('interface_hbond_percentage'),
            iface.get('interface_delta_unsat_hbonds'),
            iface.get('interface_delta_unsat_hbonds_percentage'),
        ])
        # SS breakdown
        if isinstance(ss, (list, tuple)) and len(ss) >= 8:
            alpha, beta, loops, alpha_i, beta_i, loops_i, i_plddt, ss_plddt = ss
            data.extend([alpha_i, beta_i, loops_i, alpha, beta, loops])
        else:
            data.extend([None, None, None, None, None, None])
        # InterfaceAAs (string) and target RMSD if available
        iface_AA = row.get("interface_AA", {}) or {}
        iface_aas_str = ",".join([f"{aa}:{cnt}" for aa, cnt in sorted(iface_AA.items())]) if isinstance(iface_AA, dict) else ""
        data.append(iface_aas_str)
        data.append(row.get("target_rmsd"))
        # Electric fields (optional)
        data.extend([
            iface.get('electric_field_mean'), iface.get('electric_field_max'), iface.get('electric_field_std'),
            iface.get('electric_field_divergence'), iface.get('electric_field_alignment'),
            iface.get('electrostatic_complementarity'), iface.get('field_projection'), iface.get('field_gradient')
        ])
        # ipSAE metrics in BindCraft order
        data.extend([
            iface.get('ipsae_d0chn'),
            iface.get('ipsae_d0dom'),
            iface.get('ipsae_d0res'),
            iface.get('ipsae_max'),
            iface.get('ipsae_min'),
            iface.get('ipsae_d0chn_max'),
            iface.get('ipsae_d0dom_max'),
            iface.get('ipsae_d0res_max'),
            iface.get('ipsae_d0chn_min'),
            iface.get('ipsae_d0dom_min'),
            iface.get('ipsae_d0res_min'),
        ])
        # Trajectory time and notes & placeholders for settings and filters
        # Sanitize settings to exclude non-JSON-serializable values (e.g., callables)
        adv_safe = {k: v for k, v in advanced_settings.items() if not callable(v)}
        tgt_safe = {k: v for k, v in target_settings.items() if not callable(v)}
        data.extend(["", row.get("notes", ""), json.dumps(tgt_safe), json.dumps(filters or {}), json.dumps(adv_safe)])
        _append_csv_row(paths["trajectory_csv"], data)
    else:
        # Child redesign: run BindCraft AF predictions, relax, DSSP/interface, build full MPNN row
        from BindCraft.functions import (
            mk_afdesign_model, clear_mem, calculate_averages,
            load_af2_models, check_filters, insert_data,
        )
        from adaptyv_bindcraft.bindcraft_pipeline import prepare_mpnn_data, save_successful_design, handle_failed_design

        # Required inputs
        # Prefer explicit sequence from spec (MPNN output) over any model-decoded string
        import re
        seq_raw = row.get("spec", {}).get("sequence") or row.get("best_sequence") or ""
        seq_child = re.sub("[^A-Z]", "", seq_raw.upper())
        idx = int(row.get("spec", {}).get("idx", 0)) + 1
        mpnn_design_name = f"{design_name}_mpnn{idx}"
        length_child = len(seq_child)
        # Use parent's design name for trajectory reference (length may differ from child)
        parent_spec = row.get("parent_spec", {}) or {}
        parent_len = int(parent_spec.get("binder_len", length))
        parent_seed = int(parent_spec.get("seed", seed))
        parent_name = _make_design_name(binder_name, parent_len, parent_seed)
        # Prefer relaxed parent complex if present
        traj_relaxed_path = os.path.join(paths["Trajectory/Relaxed"], f"{parent_name}.pdb")
        traj_unrelaxed_path = os.path.join(paths["Trajectory"], f"{parent_name}.pdb")
        trajectory_pdb = traj_relaxed_path if os.path.exists(traj_relaxed_path) else traj_unrelaxed_path

        # AF models selection
        design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings.get("use_multimer_design", True))

        # Compile prediction models
        clear_mem()
        complex_prediction_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=advanced_settings.get("num_recycles_validation", 1),
            data_dir=advanced_settings.get("af_params_dir", "/root/BindCraft/params"),
            use_multimer=multimer_validation,
            use_initial_guess=False,
            initial_guess=False,
            use_initial_atom_pos=False,
        )
        complex_prediction_model.prep_inputs(
            pdb_filename=target_settings.get("starting_pdb"),
            chain=target_settings.get("chains", "A"),
            binder_len=length_child,
            rm_target_seq=advanced_settings.get("rm_template_seq_predict", False),
            rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
        )

        # Predict complex and relax (BindCraft utility handles AF2 filters and relaxation)
        from BindCraft.functions.colabdesign_utils import predict_binder_complex, predict_binder_alone
        # AF2 prefilters: default to disabled to keep predictions for scoring
        af2_filters = {}
        if advanced_settings.get("enable_af2_prefilters", False):
            try:
                import json as _json
                filters_json_path = advanced_settings.get("filters_json", "")
                with open(filters_json_path, "r") as _f:
                    af2_filters = _json.load(_f)
            except Exception:
                af2_filters = {}

        prediction_stats, pass_af2_filters, complex_prediction_model = predict_binder_complex(
            complex_prediction_model,
            seq_child,
            mpnn_design_name,
            target_settings.get("starting_pdb"),
            target_settings.get("chains", "A"),
            length_child,
            trajectory_pdb,
            prediction_models,
            advanced_settings,
            af2_filters,
            paths,
            paths["failure_csv"],
        )
        # Note: even if base AF2 filters fail, continue to enrich stats and write MPNN CSV.

        # Enrich per-model stats with interface, clashes, SS, RMSDs like adaptyv_bindcraft
        best_model_pdb = None
        highest_plddt = -1.0
        for model_num in prediction_models:
            model_key = model_num + 1
            mpnn_design_pdb = os.path.join(paths["MPNN"], f"{mpnn_design_name}_model{model_key}.pdb")
            mpnn_design_relaxed = os.path.join(paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_key}.pdb")
            # Ensure relaxed structure exists; if missing but unrelaxed exists, relax now
            if (not os.path.exists(mpnn_design_relaxed)) and os.path.exists(mpnn_design_pdb):
                try:
                    openmm_relax(mpnn_design_pdb, mpnn_design_relaxed)
                except Exception:
                    pass
            # Fallback: if neither unrelaxed nor relaxed exist, try to source from AF2 outputs
            if (not os.path.exists(mpnn_design_relaxed)) and (not os.path.exists(mpnn_design_pdb)):
                try:
                    import glob, shutil
                    # Search AF2 and AF2/Ranked for this design's model
                    af2_candidates = []
                    af2_dir = paths.get("AF2")
                    af2_ranked = os.path.join(paths.get("AF2"), "Ranked") if paths.get("AF2") else None
                    patterns = [
                        f"{mpnn_design_name}_model{model_key}.pdb",
                        f"{mpnn_design_name}*model{model_key}*.pdb",
                        f"{mpnn_design_name}*.pdb",
                    ]
                    for d in [af2_dir, af2_ranked]:
                        if d and os.path.isdir(d):
                            for pat in patterns:
                                af2_candidates.extend(glob.glob(os.path.join(d, pat)))
                    if af2_candidates:
                        source_pdb = sorted(af2_candidates)[0]
                        # Copy as unrelaxed and relax into MPNN/Relaxed
                        shutil.copyfile(source_pdb, mpnn_design_pdb)
                        try:
                            openmm_relax(mpnn_design_pdb, mpnn_design_relaxed)
                        except Exception:
                            pass
                except Exception:
                    pass
            if os.path.exists(mpnn_design_relaxed):
                # Clashes
                from BindCraft.functions.openmm_utils import calculate_clash_score, score_interface, unaligned_rmsd, target_pdb_rmsd
                num_clashes_unrelaxed = calculate_clash_score(mpnn_design_pdb) if os.path.exists(mpnn_design_pdb) else 0
                num_clashes_relaxed = calculate_clash_score(mpnn_design_relaxed)
                # PAE/ipSAE-aware interface metrics
                pae_matrix = None
                pae_logits = None
                breaks = None
                chains = None
                try:
                    # Attempt to pull PAE from the compiled model when available
                    from adaptyv_bindcraft.bindcraft_pipeline import extract_pae
                    pae_matrix, pae_logits, breaks, chains = extract_pae(complex_prediction_model, length_child, advanced_settings.get("binder_chain", "B"))
                except Exception:
                    pass
                iface_scores, iface_AA, iface_residues = score_interface(
                    mpnn_design_relaxed,
                    advanced_settings.get("binder_chain", "B"),
                    pae_matrix=pae_matrix,
                    chains=chains,
                    pae_cutoff=advanced_settings.get("pae_cutoff", 10.0),
                    pae_logits=pae_logits,
                    breaks=breaks,
                )
                # Secondary structure for binder and interface
                from BindCraft.functions import calc_ss_percentage
                b_alpha, b_beta, b_loops, i_alpha, i_beta, i_loops, i_plddt, ss_plddt = calc_ss_percentage(
                    mpnn_design_relaxed, advanced_settings, advanced_settings.get("binder_chain", "B")
                )
                # RMSDs
                rmsd_site = unaligned_rmsd(trajectory_pdb, mpnn_design_relaxed, advanced_settings.get("binder_chain", "B"), advanced_settings.get("binder_chain", "B"))
                target_rmsd = target_pdb_rmsd(mpnn_design_relaxed, target_settings.get("starting_pdb"), target_settings.get("chains", "A"))

                # Update prediction_stats for this model with the extended metrics
                st = prediction_stats.get(model_key, {})
                st.update({
                    'i_pLDDT': i_plddt,
                    'ss_pLDDT': ss_plddt,
                    'Unrelaxed_Clashes': num_clashes_unrelaxed,
                    'Relaxed_Clashes': num_clashes_relaxed,
                    'Binder_Energy_Score': iface_scores.get('binder_score', 0.0),
                    'Surface_Hydrophobicity': iface_scores.get('surface_hydrophobicity', 0.0),
                    'ShapeComplementarity': iface_scores.get('interface_sc', 0.0),
                    'PackStat': iface_scores.get('interface_packstat', 0.0),
                    'dG': iface_scores.get('interface_dG', 0.0),
                    'dSASA': iface_scores.get('interface_dSASA', 0.0),
                    'dG/dSASA': iface_scores.get('interface_dG_SASA_ratio', 0.0),
                    'Interface_SASA_%': iface_scores.get('interface_fraction', 0.0),
                    'Interface_Hydrophobicity': iface_scores.get('interface_hydrophobicity', 0.0),
                    'n_InterfaceResidues': iface_scores.get('interface_nres', 0),
                    'n_InterfaceHbonds': iface_scores.get('interface_interface_hbonds', 0),
                    'InterfaceHbondsPercentage': iface_scores.get('interface_hbond_percentage', 0.0),
                    'n_InterfaceUnsatHbonds': iface_scores.get('interface_delta_unsat_hbonds', 0),
                    'InterfaceUnsatHbondsPercentage': iface_scores.get('interface_delta_unsat_hbonds_percentage', 0.0),
                    'InterfaceAAs': ",".join([f"{aa}:{cnt}" for aa, cnt in sorted((iface_AA or {}).items())]) if iface_AA else "",
                    'Interface_Helix%': i_alpha,
                    'Interface_BetaSheet%': i_beta,
                    'Interface_Loop%': i_loops,
                    'Binder_Helix%': b_alpha,
                    'Binder_BetaSheet%': b_beta,
                    'Binder_Loop%': b_loops,
                    'Hotspot_RMSD': rmsd_site,
                    'Target_RMSD': target_rmsd,
                    # Electric field and ipsae already integrated in score_interface when available
                    'electric_field_mean': iface_scores.get('electric_field_mean', 0.0),
                    'electric_field_max': iface_scores.get('electric_field_max', 0.0),
                    'electric_field_std': iface_scores.get('electric_field_std', 0.0),
                    'electric_field_divergence': iface_scores.get('electric_field_divergence', 0.0),
                    'electric_field_alignment': iface_scores.get('electric_field_alignment', 0.0),
                    'electrostatic_complementarity': iface_scores.get('electrostatic_complementarity', 0.0),
                    'field_projection': iface_scores.get('field_projection', 0.0),
                    'field_gradient': iface_scores.get('field_gradient', 0.0),
                    'ipsae_d0chn': iface_scores.get('ipsae_d0chn', 0.0),
                    'ipsae_d0dom': iface_scores.get('ipsae_d0dom', 0.0),
                    'ipsae_d0res': iface_scores.get('ipsae_d0res', 0.0),
                    'ipsae_max': iface_scores.get('ipsae_max', 0.0),
                    'ipsae_min': iface_scores.get('ipsae_min', 0.0),
                    'ipsae_d0chn_max': iface_scores.get('ipsae_d0chn_max', 0.0),
                    'ipsae_d0dom_max': iface_scores.get('ipsae_d0dom_max', 0.0),
                    'ipsae_d0res_max': iface_scores.get('ipsae_d0res_max', 0.0),
                    'ipsae_d0chn_min': iface_scores.get('ipsae_d0chn_min', 0.0),
                    'ipsae_d0dom_min': iface_scores.get('ipsae_d0dom_min', 0.0),
                    'ipsae_d0res_min': iface_scores.get('ipsae_d0res_min', 0.0),
                })
                prediction_stats[model_key] = st
                # Track best model by pLDDT
                cur_plddt = st.get('pLDDT', 0.0)
                if cur_plddt is not None and cur_plddt > highest_plddt:
                    highest_plddt = cur_plddt
                    best_model_pdb = mpnn_design_relaxed

        # Recompute averages after enriching per-model statistics
        mpnn_complex_averages = calculate_averages(prediction_stats, handle_aa=True)

        # Binder-alone prediction
        binder_prediction_model = mk_afdesign_model(
            protocol="hallucination",
            use_templates=False,
            initial_guess=False,
            use_initial_atom_pos=False,
            num_recycles=advanced_settings.get("num_recycles_validation", 1),
            data_dir=advanced_settings.get("af_params_dir", "/root/BindCraft/params"),
            use_multimer=multimer_validation,
        )
        binder_prediction_model.prep_inputs(length=length_child)
        binder_stats = predict_binder_alone(
            binder_prediction_model,
            seq_child,
            mpnn_design_name,
            length_child,
            trajectory_pdb,
            advanced_settings.get("binder_chain", "B"),
            prediction_models,
            advanced_settings,
            paths,
        )
        # Enrich binder stats with per-model Binder_RMSD like adaptyv_bindcraft
        from BindCraft.functions.openmm_utils import unaligned_rmsd
        binder_chain = advanced_settings.get("binder_chain", "B")
        for model_num in prediction_models:
            model_key = model_num + 1
            mpnn_binder_pdb = os.path.join(paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_key}.pdb")
            if os.path.exists(mpnn_binder_pdb):
                try:
                    rmsd_binder = unaligned_rmsd(trajectory_pdb, mpnn_binder_pdb, binder_chain, "A")
                    binder_stats.setdefault(model_key, {})['Binder_RMSD'] = rmsd_binder
                except Exception:
                    pass
        binder_averages = calculate_averages(binder_stats)

        # Build MPNN row using BindCraft helper
        analysis_results = {
            "binder_chain": advanced_settings.get("binder_chain", "B"),
            "trajectory_interface_residues": row.get("interface_residues", ""),
        }
        mpnn_sequence = {
            "seq": seq_child,
            "score": row.get("spec", {}).get("mpnn_score", 0.0),
            "seqid": row.get("spec", {}).get("mpnn_seqid", 0),
        }
        mpnn_data = prepare_mpnn_data(
            mpnn_design_name,
            advanced_settings,
            length_child,
            seed,
            analysis_results,
            mpnn_sequence,
            mpnn_complex_averages,
            prediction_stats,
            binder_averages,
            binder_stats,
            target_settings,
            prediction_models,
            "unknown",
            "",
        )

        # Append to MPNN CSV
        insert_data(paths["mpnn_csv"], mpnn_data)

        # Filter and accept/reject
        from BindCraft.functions import generate_dataframe_labels
        _, design_labels, _ = generate_dataframe_labels()
        filter_conditions = check_filters(mpnn_data, design_labels, filters or {})
        if filter_conditions is True:
            # Prefer best_model_pdb if determined and exists, else select first existing relaxed model
            if best_model_pdb and os.path.exists(best_model_pdb):
                best_path = best_model_pdb
            else:
                existing = None
                for model_num in prediction_models:
                    cand = os.path.join(paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                    if os.path.exists(cand):
                        existing = cand
                        break
                best_path = existing or os.path.join(paths["MPNN/Relaxed"], f"{mpnn_design_name}_model1.pdb")
            save_successful_design(mpnn_design_name, best_path, design_name, paths, mpnn_data, paths["final_csv"])
            row["accepted"] = True
        else:
            if best_model_pdb and os.path.exists(best_model_pdb):
                fail_path = best_model_pdb
            else:
                existing = None
                for model_num in prediction_models:
                    cand = os.path.join(paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                    if os.path.exists(cand):
                        existing = cand
                        break
                fail_path = existing or os.path.join(paths["MPNN/Relaxed"], f"{mpnn_design_name}_model1.pdb")
            handle_failed_design(mpnn_design_name, fail_path, filter_conditions, paths, paths["failure_csv"])
            row["accepted"] = False


def default_spawn_children(*, spec: dict, parent_result: dict, parent_row: dict, target_settings: Dict[str, Any], advanced_settings: Dict[str, Any]) -> List[Tuple[dict, Callable[[dict], dict]]]:
    """Generate MPNN redesigns using BindCraft's mpnn_gen_sequence and create child predict-only workflows.

    Returns list of (child_spec, child_build) pairs.
    """
    from BindCraft.functions import mpnn_gen_sequence
    from .predict import make_predict_only_workflow

    binder_chain = advanced_settings.get("binder_chain", "B")
    pdb_relaxed = parent_row.get("pdb_relaxed")
    interface_residues = parent_row.get("interface_residues", "")
    if pdb_relaxed is None:
        return []

    mpnn_traj = mpnn_gen_sequence(pdb_relaxed, binder_chain, interface_residues, advanced_settings)
    seqs = mpnn_traj.get("seq", [])
    scores = mpnn_traj.get("score", [])
    seqids = mpnn_traj.get("seqid", [])
    # Ensure child sequences match parent binder length
    parent_len = int(parent_row.get("spec", {}).get("binder_len", 0)) or None
    children: List[Tuple[dict, Callable[[dict], dict]]] = []
    for idx, seq in enumerate(seqs):
        seq_use = seq[-parent_len:] if parent_len else seq
        child_spec = {
            "binder_len": len(seq_use),
            "seed": spec.get("seed", 0),
            "sequence": seq_use,
            "idx": idx,
            "mpnn_score": float(scores[idx]) if idx < len(scores) else 0.0,
            "mpnn_seqid": float(seqids[idx]) if idx < len(seqids) else 0.0,
        }
        def child_build(s=child_spec):
            # Minimal no-op workflow; emit handles real predictions
            return {"phases": [], "binder_len": s["binder_len"], "seed": s.get("seed", 0)}
        children.append((child_spec, child_build))
    return children


def sample_specs_bindcraft_style(*, max_trajectories: int, target_settings: Dict[str, Any], runtime_seed: int | None, runtime_length: int | None) -> List[dict]:
    # Deterministic per-trajectory seeds and lengths in [70, 100]
    import numpy as _np
    rng = _np.random.default_rng(int(runtime_seed) if runtime_seed is not None else None)
    specs: List[dict] = []
    for i in range(int(max_trajectories)):
        seed_i = int(rng.integers(0, 2**31 - 1))
        length_i = int(rng.integers(70, 101))  # inclusive upper bound 100
        specs.append({
            "seed": seed_i,
            "binder_len": length_i,
            "idx": i,
        })
    return specs


def run_bindcraft_compat(
    *,
    repo_root: str,                  # "/Users/.../adaptyv_bindcraft"
    design_path: str,                # output directory
    build_parent: Callable[[dict], dict] | None,
    spawn_children: Callable[[dict, dict, dict], List[Tuple[dict, Callable[[dict], dict]]]] | None,
    emit_row: Callable[[str, dict, Dict[str, str]], None] | None,
    max_trajectories: int,
    target_settings: Dict[str, Any],
    advanced_settings: Dict[str, Any],
    filters: Dict[str, Any],
    runtime_seed: int | None = None,
    runtime_length: int | None = None,
    stop: Callable[[List[dict]], bool] | None = None,
) -> dict:
    """One-shot BindCraft-compatible loop using Mosaic workflows.

    - build_parent(spec) -> workflow dict (Mosaic)
    - spawn_children(spec, parent_result) -> list[(child_spec, child_build)]
    - emit_row(kind, row, paths) -> IO and CSV writes
    - Uses BindCraft CSV schemas and deterministic seeds/lengths
    """
    _add_bindcraft_paths(repo_root)
    paths = _bindcraft_dirs(design_path)
    _init_csvs(paths)

    specs = sample_specs_bindcraft_style(
        max_trajectories=max_trajectories,
        target_settings=target_settings,
        runtime_seed=runtime_seed,
        runtime_length=runtime_length,
    )

    # Provide a default AF2+MPNN BindCraft-like builder if none provided
    if build_parent is None:
        target_pdb = target_settings.get("starting_pdb")
        if not target_pdb:
            raise ValueError("build_parent is None and target_settings lacks 'starting_pdb'")
        build_parent = make_build_parent_bindcraft_af2_prior(
            target_pdb_path=str(target_pdb),
            af_params_dir=advanced_settings.get("af_params_dir", "."),
            num_recycles=int(advanced_settings.get("num_recycles_design", 1)),
            optimizer=advanced_settings.get("optimizer", "simplex_apgm"),
            use_boltz2=bool(advanced_settings.get("use_boltz2", False)),
            loss_clip_l=advanced_settings.get("loss_clip_l"),
            loss_clip_u=advanced_settings.get("loss_clip_u"),
            grad_norm_mode=advanced_settings.get("grad_norm_mode", "l2"),
        )

    def _emit(kind: str, row: dict) -> None:
        if emit_row is not None:
            emit_row(kind, row, paths)
        else:
            default_emit_row(kind=kind, row=row, paths=paths, target_settings=target_settings, advanced_settings=advanced_settings, filters=filters)

    out = run_many(
        specs=specs,
        build=build_parent,
        spawn=(lambda a,b,c: default_spawn_children(spec=a, parent_result=b, parent_row=c, target_settings=target_settings, advanced_settings=advanced_settings)) if spawn_children is None else spawn_children,
        emit=_emit,
        stop=stop,
        resume=False,
        out_dir=design_path,
    )
    return out


@lru_cache(maxsize=1)
def _load_boltz_cached():
    from mosaic.losses.boltz import load_boltz
    return load_boltz()


def make_build_parent_bindcraft_af2_prior(
    *,
    target_pdb_path: str,
    af_params_dir: str | None = None,
    num_recycles: int = 1,
    optimizer: str | None = None,
    use_boltz2: bool = False,
    loss_clip_l: float | None = None,
    loss_clip_u: float | None = None,
    grad_norm_mode: str = "l2_effL",
):
    """Return a build(spec)->workflow that runs AF2 with a BindCraft-like loss plus a ProteinMPNN sequence prior.

    - Uses binder (variable) + target (fixed) multimer with the target templated from `target_pdb_path`.
    - Structure loss follows BindCraft-style terms; adds a sequence recovery prior from ProteinMPNN.
    """
    import gemmi
    import numpy as np
    import mosaic.losses.structure_prediction as sp
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import ClippedLoss
    from mosaic.models.af2 import AlphaFold2
    from mosaic.structure_prediction import TargetChain
    from mosaic.common import LossTerm
    import mosaic.losses.boltz2 as bl2
    from mosaic.losses.boltz2 import Boltz2Loss
    from mosaic_workflows.optimizers import (
        simplex_APGM_adapter,
        sgd_logits_adapter,
        adamw_logits_adapter,
        gradient_MCMC_adapter,
        st_gumbel_adapter,
        rao_gumbel_adapter,
        zgr_adapter,
        semi_greedy_adapter,
    )
    from mosaic_workflows.transforms import (
        temperature_on_logits,
        e_soft_on_logits,
        gradient_normalizer,
        zero_disallowed,
    )

    # Load template structure and extract target chain and sequence
    st = gemmi.read_structure(str(target_pdb_path))
    st.setup_entities()
    target_chain = st[0][0]  # first chain in the provided PDB
    target_sequence = gemmi.one_letter_code([r.name for r in target_chain])

    # Preload models shared across options
    af2_model = AlphaFold2(data_dir=af_params_dir or ".")
    mpnn = ProteinMPNN.from_pretrained()
    boltz2_model = None  # lazy init when used

    def _select_optimizer(name: str):
        name = (name or "").lower()
        if name in ("", "simplex_apgm", "simplex"):  # default
            return simplex_APGM_adapter
        if name in ("sgd", "sgd_logits"):
            return sgd_logits_adapter
        if name in ("adamw", "adamw_logits"):
            return adamw_logits_adapter
        if name in ("gradient_mcmc", "mcmc"):
            return gradient_MCMC_adapter
        if name in ("st_gumbel", "st"):
            return st_gumbel_adapter
        if name in ("rao_gumbel", "rb"):
            return rao_gumbel_adapter
        if name in ("zgr",):
            return zgr_adapter
        if name in ("semi_greedy", "semi-greedy"):
            return semi_greedy_adapter
        return simplex_APGM_adapter

    def _make_boltz2_yaml(binder_length: int, target_seq: str) -> str:
        base = (
            """
version: 1
sequences:
  - protein:
      id: [A]
      sequence: {binder}
      msa: empty
  - protein:
      id: [B]
      sequence: {target}
"""
        ).format(binder=("X" * int(binder_length)), target=target_seq)
        # Add target template if provided
        return base + (f"\ntemplates:\n  - pdb: {str(target_pdb_path)}\n" if target_pdb_path else "")

    def make_build_loss(binder_len: int, *, use_boltz2: bool = False, clip_bounds: tuple[float, float] | None = None):
        # Common BindCraft-style structure objective
        struct_loss = (
            1.0 * sp.WithinBinderContact()
            + 1.0 * sp.BinderTargetContact()
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 0.025 * sp.pTMEnergy()
            + 0.1 * sp.PLDDTLoss()
            + 0.0 * sp.PLDDTPerResidueReport()
        )
        # Clip MPNN prior by default: cap only upper tail to preserve helpful gradients
        mpnn_prior = InverseFoldingSequenceRecovery(mpnn=mpnn, temp=0.01, num_samples=8, jacobi_iterations=8)
        import numpy as _np
        seq_prior = ClippedLoss(mpnn_prior, -_np.inf, 100.0, name="mpnn_clipped")
        combined = struct_loss + 5.0 * seq_prior

        if use_boltz2:
            nonlocal boltz2_model
            if boltz2_model is None:
                boltz2_model = bl2.load_boltz2()
            yaml_str = _make_boltz2_yaml(binder_len, str(target_sequence))
            features, _writer = bl2.load_features_and_structure_writer(yaml_str)
            if clip_bounds:
                return ClippedLoss(
                    Boltz2Loss(
                        joltz2=boltz2_model,
                        features=features,
                        loss=combined,
                        deterministic=True,
                        recycling_steps=0,
                        name="boltz2",
                    ),
                    clip_bounds[0],
                    clip_bounds[1],
                    name="clipped_total",
                )
            return Boltz2Loss(
                joltz2=boltz2_model,
                features=features,
                loss=combined,
                deterministic=True,
                recycling_steps=0,
                name="boltz2",
            )
        else:
            # AF2 multimer features via standardized interface: binder + target with template on target
            feats, _ = af2_model.binder_features(
                int(binder_len),
                chains=[TargetChain(sequence=str(target_sequence), use_msa=False, template_chain=target_chain)],
            )
            loss_term = af2_model.build_loss(
                loss=combined,
                features=feats,
                recycling_steps=int(num_recycles),
                name="af2",
            )
            if clip_bounds:
                return ClippedLoss(
                    loss_term,
                    clip_bounds[0],
                    clip_bounds[1],
                    name="clipped_total",
                )
            return loss_term

    def build_parent(spec: dict) -> dict:
        binder_len = int(spec["binder_len"])
        seed = int(spec.get("seed", 0))

        # Configuration (from factory args, optionally overridden in spec)
        optimizer_name = (spec.get("optimizer") or optimizer or "simplex_apgm").lower()
        use_boltz2_flag = bool(spec.get("use_boltz2", use_boltz2))
        _clip_l = spec.get("loss_clip_l", loss_clip_l)
        _clip_u = spec.get("loss_clip_u", loss_clip_u)
        clip_bounds = None
        if _clip_l is not None and _clip_u is not None:
            try:
                clip_bounds = (float(_clip_l), float(_clip_u))
            except Exception:
                clip_bounds = None
        _grad_norm_mode = (spec.get("grad_norm_mode") or grad_norm_mode or "l2").lower()

        def build_loss():
            return make_build_loss(binder_len, use_boltz2=use_boltz2_flag, clip_bounds=clip_bounds)

        # Three-phase schedule; mirror notebook (100 + 25 + 25)
        warmup_steps = 100
        soft_steps = 25
        anneal_steps = 25

        warmup = {
            "name": "warmup",
            "build_loss": build_loss,
            "optimizer": _select_optimizer(optimizer_name),
            "steps": int(warmup_steps),
            # Provide generic fields understood by multiple optimizers
            "schedule": lambda g, p, L=binder_len: {
                "lr": 0.1,
                "temperature": 1.0,
                "e_soft": 0.8,
                "stepsize": 0.1 * float(np.sqrt(L)),
                "scale": 1.0,
            },
            "transforms": {
                "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                "grad": [gradient_normalizer(mode=_grad_norm_mode, log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=None)],
            },
            "analyze_every": 1,
        }

        soft = {
            "name": "soft",
            "build_loss": build_loss,
            "optimizer": _select_optimizer(optimizer_name),
            "steps": int(soft_steps),
            "schedule": lambda g, p, L=binder_len: {
                "lr": 0.1,
                "temperature": 1.0,
                "e_soft": 0.8,
                "stepsize": 0.1 * float(np.sqrt(L)),
                "scale": 1.25,
            },
            "transforms": {
                "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                "grad": [gradient_normalizer(mode=_grad_norm_mode, log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=None)],
            },
        }

        def anneal_sched(g, p):
            total = float(anneal_steps)
            frac = (p / total) if total > 0 else 1.0
            temp = 1.0 - (1.0 - 0.05) * (frac ** 2)
            return {"lr": 0.05, "temperature": max(0.05, float(temp)), "e_soft": 1.0}

        anneal = {
            "name": "anneal",
            "build_loss": build_loss,
            "optimizer": _select_optimizer(optimizer_name),
            "steps": int(anneal_steps),
            "schedule": anneal_sched,
            "transforms": {
                "pre_logits": [temperature_on_logits()],
                "grad": [gradient_normalizer(mode=_grad_norm_mode, log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=None)],
            },
        }

        phases = [warmup, soft, anneal]
        # simple init
        rng = np.random.default_rng(seed)
        x0 = (rng.standard_normal((binder_len, 20)).astype(np.float32) * 0.1)
        # Pass config through spec for downstream phases (optim selection, clipping, model choice)
        wf = {"phases": phases, "binder_len": binder_len, "seed": seed, "initial_x": x0}
        wf.update({
            "optimizer": optimizer_name,
            "use_boltz2": use_boltz2_flag,
            "loss_clip_l": _clip_l,
            "loss_clip_u": _clip_u,
            "grad_norm_mode": _grad_norm_mode,
        })
        return wf

    return build_parent


