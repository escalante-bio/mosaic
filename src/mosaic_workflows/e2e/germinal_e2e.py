from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable
from importlib import import_module
from pathlib import Path
import sys as _sys

from .pipeline import run_pipeline
from ..germinal import make_workflow as _mk
from .utils import ensure_dirs, sample_specs_bindcraft_style, write_csv_row


def _make_build_parent_germinal(*, target_pdb_path: str, af_params_dir: str | None, vhh_config: Dict[str, Any] | None):
    def build_parent(spec: dict) -> dict:
        binder_len = int(spec["binder_len"])
        seed = int(spec.get("seed", 0))
        vhh = dict(vhh_config or {})
        cdr_lengths = list(vhh.get("cdr_lengths", [11, 8, 18]))
        fw_lengths = list(vhh.get("fw_lengths", [25, 17, 38, 14]))
        segments = [("fw1", fw_lengths[0]), ("cdr1", cdr_lengths[0]), ("fw2", fw_lengths[1]), ("cdr2", cdr_lengths[1]), ("fw3", fw_lengths[2]), ("cdr3", cdr_lengths[2]), ("fw4", fw_lengths[3])]
        pos = 0
        cdr_positions: List[int] = []
        framework_positions: List[int] = []
        for name, length in segments:
            idxs = list(range(pos, pos + int(length)))
            if name.startswith("cdr"):
                cdr_positions.extend(idxs)
            else:
                framework_positions.extend(idxs)
            pos += int(length)
        cdr_positions = [i for i in cdr_positions if 0 <= i < binder_len]
        framework_positions = [i for i in framework_positions if 0 <= i < binder_len]
        fw_seq = "G" * binder_len
        wf = _mk(
            binder_len=binder_len,
            target_pdb_path=str(target_pdb_path),
            target_chain_id="A",
            target_hotspots=tuple(vhh.get("target_hotspots", [])),
            cdr_positions=tuple(cdr_positions),
            framework_positions=tuple(framework_positions),
            framework_sequence=str(vhh.get("framework_seq", fw_seq)),
            af2_params_dir=af_params_dir or ".",
            af2_num_recycles=int(vhh.get("num_recycles_design", 3)),
            # Loss weights (mirror Germinal defaults)
            w_plddt=float(vhh.get("weights_plddt", 1.0)),
            w_iptm=float(vhh.get("weights_iptm", 0.7)),
            w_pae_bt=float(vhh.get("weights_pae_inter", 0.5)),
            w_intra_con=float(vhh.get("weights_con_intra", 0.1)),
            w_inter_con=float(vhh.get("weights_con_inter", 0.0)),
            w_rg=float(vhh.get("weights_rg", 0.1)),
            w_dgram_cce=float(vhh.get("dgram_cce", 0.01)),
            w_fw_penalty=float(vhh.get("w_fw_penalty", 0.5)),
            w_cdr_helix_suppress=float(vhh.get("weights_helix", 0.1)),
            w_cdr_beta_suppress=float(vhh.get("weights_beta", 0.1)),
            w_pae_intra=float(vhh.get("weights_pae_intra", 0.0)),
            # Schedules / phases
            steps_logits=int(vhh.get("logits_steps", 65)),
            steps_softmax=int(vhh.get("softmax_steps", 35)),
            steps_semigreedy=int(vhh.get("search_steps", 10)),
            framework_bias=float(vhh.get("bias_redesign", 10)),
            framework_contact_offset=float(vhh.get("framework_contact_offset", 1.0)),
            lr=float(vhh.get("learning_rate", 0.1)),
            plddt_thr=float(vhh.get("plddt_threshold", 0.82)),
            iptm_thr=float(vhh.get("i_ptm_threshold", 0.68)),
            ipae_thr=float(vhh.get("i_pae_threshold", 0.27)),
            seq_entropy_thr=float(vhh.get("seq_entropy_threshold", 0.10)),
            grad_merge_method=str(vhh.get("grad_merge_method", "pcgrad")),
            omit_aas=str(vhh.get("omit_AAs", "C")),
            seq_init_mode=(vhh.get("seq_init_mode", ["gumbel"]) or ["gumbel"])[0],
        )
        wf["seed"] = seed
        return wf
    return build_parent


def _spawn_children_germinal(*, spec: dict, parent_row: dict, target_settings: Dict[str, Any], vhh_config: Dict[str, Any]) -> List[Tuple[dict, Callable[[dict], dict]]]:
    traj_pdb = parent_row.get("structure_path")
    if not traj_pdb:
        return []
    if str(Path("/tmp/germinal")) not in _sys.path:
        _sys.path.insert(0, "/tmp/germinal")
    redesign_mod = import_module("germinal.filters.redesign")
    abmpnn_seqs, ok = redesign_mod.run_abmpnn_redesign_pipeline(
        trajectory_pdb_af=str(traj_pdb),
        target_chain=str(target_settings.get("chains", "A")),
        binder_chain=str(vhh_config.get("binder_chain", "B")),
        run_settings={
            "backbone_noise": float(vhh_config.get("backbone_noise", 0.0)),
            "model_path": str(vhh_config.get("model_path", "abmpnn")),
            "mpnn_weights": str(vhh_config.get("mpnn_weights", "abmpnn")),
            "mpnn_fix_interface": bool(vhh_config.get("mpnn_fix_interface", True)),
            "omit_AAs": str(vhh_config.get("omit_AAs", "C")),
            "sampling_temp": float(vhh_config.get("sampling_temp", 0.1)),
            "num_seqs": int(vhh_config.get("num_seqs", 40)),
            "max_mpnn_sequences": int(vhh_config.get("max_mpnn_sequences", 4)),
            "cdr_positions": list(vhh_config.get("cdr_positions", [])),
        },
        atom_distance_cutoff=float(vhh_config.get("atom_distance_cutoff", 3.0)),
    )
    out: List[Tuple[dict, Callable[[dict], dict]]] = []
    for idx, s in enumerate(abmpnn_seqs or []):
        child_spec = {"binder_len": len(s.get("seq", "")), "seed": int(spec.get("seed", 0)), "sequence": str(s.get("seq", "")), "idx": idx}
        def child_build(s=child_spec):
            return {"phases": [], "binder_len": s["binder_len"], "seed": s.get("seed", 0)}
        out.append((child_spec, child_build))
    return out


def _emit_row_germinal(*, kind: str, row: dict, paths: Dict[str, str], target_settings: Dict[str, Any], vhh_config: Dict[str, Any], af3_settings: Dict[str, Any]) -> None:
    seq = row.get("best_sequence") or row.get("spec", {}).get("sequence")
    if not seq:
        return
    for p in ("/tmp/germinal", "/root/germinal"):
        if p not in _sys.path and Path(p).exists():
            _sys.path.insert(0, p)
    af3 = import_module("germinal.filters.af3")
    out_dir = paths.get("Trajectory") or str(Path(paths.get("trajectory_csv", ".")).parent)
    base_name = row.get("spec", {}).get("design_name", "design")
    design_name = f"{base_name}_child_{int(row.get('spec', {}).get('idx', 0))}" if kind == "child" else base_name
    pdb_path, scores = af3.run_af3(
        binder_seq=str(seq),
        target_seq=str(target_settings.get("target_seq", "")),
        target_chains=str(target_settings.get("chains", "A")),
        output_dir=str(out_dir),
        design_name=str(design_name),
        seed=int(row.get("spec", {}).get("seed", 0)),
        run_settings=dict(af3_settings or {}),
        binder_chain=str(vhh_config.get("binder_chain", "B")),
        msa_mode=str(vhh_config.get("msa_mode", "target")),
    )
    row["structure_path"] = pdb_path
    pl = float(scores.get("plddt", 0.0))
    iptm = float((scores.get("iptm") or [0.0])[0])
    pae = float(scores.get("pae", 1e9))
    passed = (pl > float(vhh_config.get("plddt_threshold", 0.85))) and (iptm > float(vhh_config.get("i_ptm_threshold", 0.75))) and (pae <= float(vhh_config.get("pae_threshold", 7.0)))
    row.setdefault("metrics", {})
    row["metrics"].update({"af3.plddt": pl, "af3.iptm": iptm, "af3.pae": pae, "af3.passed": bool(passed)})
    write_csv_row(paths["trajectory_csv"], [design_name, pl, iptm, pae, passed])


def run_e2e(*, design_path: str, target_settings: Dict[str, Any], vhh_config: Dict[str, Any], af3_settings: Dict[str, Any], max_trajectories: int, runtime_seed: int | None = None) -> dict:
    paths = ensure_dirs(design_path)
    build_parent = _make_build_parent_germinal(target_pdb_path=str(target_settings["starting_pdb"]), af_params_dir=af3_settings.get("af_params_dir", "."), vhh_config=vhh_config)
    specs = sample_specs_bindcraft_style(max_trajectories=int(max_trajectories), runtime_seed=runtime_seed)
    out = run_pipeline(
        specs=specs,
        build_parent=build_parent,
        spawn_children=lambda a,b,c: _spawn_children_germinal(spec=a, parent_row=c, target_settings=target_settings, vhh_config=vhh_config),
        emit_row=lambda k,r: _emit_row_germinal(kind=k, row=r, paths=paths, target_settings=target_settings, vhh_config=vhh_config, af3_settings=af3_settings),
        out_dir=design_path,
    )
    return out


