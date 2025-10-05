from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from ..design import run_workflow
from ..mhetase_scaffold import make_workflow as make_mhetase_workflow
from .utils import ensure_dirs, write_csv_row, save_fasta
from importlib import import_module
from pathlib import Path
import sys as _sys


def _metric(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _extract_motif_rmsd(aux: Dict[str, Any]) -> float:
    if isinstance(aux, dict):
        for k, v in aux.items():
            if k.lower().find("motif_rmsd") >= 0 and isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                inner = _extract_motif_rmsd(v)
                if inner > 0.0:
                    return inner
    return 0.0


def run_e2e(
    *,
    design_path: str,
    binder_len: int,
    scaffold_kwargs: Dict[str, Any],
    af3_settings: Dict[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    paths = ensure_dirs(design_path)

    wf = make_mhetase_workflow(binder_len=int(binder_len), **scaffold_kwargs)
    wf["seed"] = int(seed)
    wf["initial_x"] = np.random.randn(int(binder_len), 20).astype(np.float32) * 0.1

    out = run_workflow(wf)
    seq = str(out.get("best_sequence", ""))
    save_fasta("mhetase", seq, paths)

    # Collect simple metrics
    metrics = out.get("metrics") or {}
    traj = out.get("trajectory") or []
    last = traj[-1] if traj else {}
    aux = last.get("aux", {}) if isinstance(last, dict) else {}
    pl = _metric(metrics.get("plddt", 0.0))
    iptm = _metric(metrics.get("i_ptm", 0.0))
    motif_rmsd = _extract_motif_rmsd(aux)
    write_csv_row(paths["trajectory_csv"], ["mhetase", pl, iptm, motif_rmsd])

    # Optional AF3 validation on binder-only
    if af3_settings:
        for p in ("/tmp/germinal", "/root/germinal"):
            if p not in _sys.path and Path(p).exists():
                _sys.path.insert(0, p)
        af3 = import_module("germinal.filters.af3")
        af3.run_af3(
            binder_seq=seq,
            target_seq="",
            target_chains="A",
            output_dir=str(Path(design_path) / "Trajectory"),
            design_name="mhetase",
            seed=int(seed),
            run_settings=dict(af3_settings),
            binder_chain="A",
            msa_mode="none",
        )
    return {"best_sequence": seq, "metrics": {"plddt": pl, "i_ptm": iptm, "motif_rmsd": motif_rmsd}}


