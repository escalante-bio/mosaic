"""Run the same binder complexes through multiple Mosaic folding backends.

Each backend runs in a fresh subprocess and writes to its own directory. This
both releases model memory between runs and prevents one backend from appending
to another backend's ``evaluation_stats.csv``. Arbitrary filter configs and
explicit metric groups are forwarded unchanged, so PyRosetta remains optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

BACKEND_PRESETS = {
    "af2": "af2.json",
    "boltz2": "boltz2.json",
    "protenix-v2": "protenix_v2.json",
    "esmfold2": "esmfold2_fast.json",
}
METRIC_GROUPS = ("confidence", "monomer", "geometry", "dssp", "pyrosetta")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mosaic.binder_design.matrix")
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--filters", required=True, type=Path)
    parser.add_argument("--advanced-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--backend",
        action="append",
        choices=tuple(BACKEND_PRESETS),
        dest="backends",
        help="backend to run; repeat as needed (default: all)",
    )
    parser.add_argument("--input-pdb", action="append", type=Path, default=[])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--binder-chain")
    parser.add_argument("--target-chains")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metric-group",
        action="append",
        choices=METRIC_GROUPS,
        default=[],
        help="compute a metric family without adding an acceptance threshold",
    )
    parser.add_argument(
        "--device",
        action="append",
        default=[],
        metavar="BACKEND=CUDA_MASK|cpu",
        help="per-backend device assignment; repeat as needed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse backend results that already contain evaluation_stats.csv",
    )
    return parser


def _device_map(values: Sequence[str]) -> dict[str, str]:
    devices: dict[str, str] = {}
    for value in values:
        backend, separator, device = value.partition("=")
        backend, device = backend.strip(), device.strip()
        if not separator or backend not in BACKEND_PRESETS or not device:
            choices = ", ".join(BACKEND_PRESETS)
            raise ValueError(
                f"Invalid --device '{value}'; expected BACKEND=CUDA_MASK|cpu "
                f"with BACKEND in: {choices}"
            )
        devices[backend] = device
    return devices


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} not found: {resolved}")
    return resolved


def _write_combined_csv(output_root: Path, backends: Sequence[str]) -> Path:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = ["Backend"]
    for backend in backends:
        stats = output_root / backend / "evaluation_stats.csv"
        if not stats.is_file():
            continue
        with stats.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"Backend": backend, **row})
                for field in row:
                    if field not in fieldnames:
                        fieldnames.append(field)

    combined = output_root / "evaluation_matrix.csv"
    with combined.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return combined


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = _require_file(args.settings, "settings config")
        filters = _require_file(args.filters, "filter config")
        devices = _device_map(args.device)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    try:
        inputs = [_require_file(path, "input PDB") for path in args.input_pdb]
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    input_dir = None
    if args.input_dir is not None:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            print(f"input directory not found: {input_dir}", file=sys.stderr)
            return 2
    if not inputs and input_dir is None:
        print("provide --input-pdb and/or --input-dir", file=sys.stderr)
        return 2

    backends = args.backends or list(BACKEND_PRESETS)
    advanced_dir = args.advanced_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "settings": str(settings),
        "filters": str(filters),
        "metric_groups": list(dict.fromkeys(args.metric_group)),
        "backends": {},
    }
    failed = False

    for backend in backends:
        try:
            advanced = _require_file(
                advanced_dir / BACKEND_PRESETS[backend],
                f"{backend} advanced config",
            )
        except ValueError as error:
            print(error, file=sys.stderr)
            return 2
        backend_output = output_root / backend
        stats = backend_output / "evaluation_stats.csv"
        log_path = backend_output / "run.log"
        if stats.is_file():
            if not args.resume:
                print(
                    f"refusing to append duplicate results for {backend}: {stats}; "
                    "use --resume to reuse them",
                    file=sys.stderr,
                )
                return 2
            manifest["backends"][backend] = {
                "status": "reused",
                "output_dir": str(backend_output),
                "evaluation_stats": str(stats),
                "log": str(log_path),
            }
            continue

        backend_output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "mosaic.binder_design",
            "--settings",
            str(settings),
            "--advanced",
            str(advanced),
            "--filters",
            str(filters),
            "--mode",
            "evaluate",
            "--output-dir",
            str(backend_output),
            "--seed",
            str(args.seed),
        ]
        for path in inputs:
            command.extend(["--input-pdb", str(path)])
        if input_dir is not None:
            command.extend(["--input-dir", str(input_dir)])
        if args.binder_chain:
            command.extend(["--binder-chain", args.binder_chain])
        if args.target_chains:
            command.extend(["--target-chains", args.target_chains])
        for group in dict.fromkeys(args.metric_group):
            command.extend(["--evaluation-metric-group", group])

        environment = os.environ.copy()
        device = devices.get(backend)
        if device:
            mask = "" if device.lower() == "cpu" else device
            environment["CUDA_VISIBLE_DEVICES"] = mask
            environment["NVIDIA_VISIBLE_DEVICES"] = mask

        with log_path.open("w", encoding="utf-8") as log_handle:
            result = runner(
                command,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return_code = int(getattr(result, "returncode", 1))
        status = "completed" if return_code == 0 and stats.is_file() else "failed"
        failed |= status == "failed"
        manifest["backends"][backend] = {
            "status": status,
            "return_code": return_code,
            "device": device or "inherited",
            "advanced": str(advanced),
            "output_dir": str(backend_output),
            "evaluation_stats": str(stats) if stats.is_file() else None,
            "log": str(log_path),
        }

    combined = _write_combined_csv(output_root, backends)
    manifest["combined_csv"] = str(combined)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Combined evaluation matrix: {combined}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
