from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List
import sys as _sys
import numpy as _np


def ensure_dirs(base: str) -> Dict[str, str]:
    names = [
        "Accepted", "Trajectory", "MPNN", "Rejected", "AF", "CSV", "logs", "settings",
        "Accepted/Ranked", "Accepted/Plots",
        "Trajectory/Relaxed", "Trajectory/Plots",
        "MPNN/Sequences", "MPNN/Relaxed",
    ]
    paths: Dict[str, str] = {}
    for n in names:
        p = Path(base) / n
        p.mkdir(parents=True, exist_ok=True)
        paths[n] = str(p)
    paths["trajectory_csv"] = str(Path(base) / "trajectory_stats.csv")
    paths["mpnn_csv"] = str(Path(base) / "mpnn_design_stats.csv")
    paths["final_csv"] = str(Path(base) / "final_design_stats.csv")
    return paths


def write_csv_row(csv_path: str, row: List[Any]) -> None:
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def save_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def save_bytes(path: str, data: bytes) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def save_fasta(name: str, seq: str, base_paths: Dict[str, str]) -> None:
    fasta = Path(base_paths["Accepted"]) / f"{name}.fasta"
    save_text(str(fasta), f">{name}\n{seq}\n")


def sample_specs_bindcraft_style(*, max_trajectories: int, runtime_seed: int | None) -> List[dict]:
    rng = _np.random.default_rng(int(runtime_seed) if runtime_seed is not None else None)
    specs: List[dict] = []
    for i in range(int(max_trajectories)):
        seed_i = int(rng.integers(0, 2**31 - 1))
        length_i = int(rng.integers(70, 101))
        specs.append({"seed": seed_i, "binder_len": length_i, "idx": i})
    return specs


def make_design_name(binder_name: str, length: int, seed: int) -> str:
    return f"{binder_name}_l{int(length)}_s{int(seed)}"


def clean_sequence(seq: str) -> str:
    return "".join([c for c in (seq or "").upper() if "A" <= c <= "Z"])  # fast A–Z filter


def add_sys_path_if_exists(p: str) -> None:
    if p and (p not in _sys.path) and Path(p).exists():
        _sys.path.insert(0, p)


