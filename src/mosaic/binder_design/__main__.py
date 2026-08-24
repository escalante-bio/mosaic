"""Command line entry point for the native-Mosaic binder design pipeline.

    python -m mosaic.binder_design \\
        --settings  configs/settings_target/q26.json \\
        --advanced  configs/settings_advanced/q26_advanced.json \\
        --filters   configs/settings_filters/production.json \\
        --mode      filter

Modes mirror DdCraft's plus a standalone evaluation entry point:
``trajectory`` runs stage 1 only, ``filter`` runs stages 2-4 over existing
trajectories, ``full`` runs everything, and ``evaluate`` starts from existing
complex PDBs and runs folding plus configured filters (never hallucination).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mosaic.binder_design")
    parser.add_argument("--settings", required=True, help="target settings JSON")
    parser.add_argument("--advanced", required=True, help="advanced settings JSON")
    parser.add_argument("--filters", required=True, help="filter thresholds JSON")
    parser.add_argument(
        "--mode",
        default="full",
        choices=("trajectory", "filter", "full", "evaluate"),
        help="which stages to run",
    )
    parser.add_argument(
        "--input-pdb",
        action="append",
        type=Path,
        default=[],
        help="complex PDB to evaluate; repeat for multiple inputs",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="evaluate every .pdb directly inside this directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override design_path from target settings for this invocation",
    )
    parser.add_argument(
        "--evaluation-metric-group",
        action="append",
        choices=("confidence", "monomer", "geometry", "dssp", "pyrosetta"),
        default=[],
        help=(
            "compute an evaluation metric family without making it an acceptance "
            "filter; repeat to request multiple families"
        ),
    )
    parser.add_argument(
        "--binder-chain",
        help="override the binder chain from target settings in evaluate mode",
    )
    parser.add_argument(
        "--target-chains",
        help="override comma-separated target chains in evaluate mode",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="stop after this many trajectory backbones",
    )
    parser.add_argument("--seed", type=int, default=0, help="master RNG seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    import numpy as np

    from mosaic.binder_design.pipeline import BinderDesignPipeline, PipelineConfig

    config = PipelineConfig.from_files(args.settings, args.advanced, args.filters)
    if args.output_dir is not None:
        config.target_settings["design_path"] = str(args.output_dir.expanduser().resolve())
    if args.evaluation_metric_group:
        if args.mode != "evaluate":
            print(
                "--evaluation-metric-group is only valid with --mode evaluate",
                file=sys.stderr,
            )
            return 1
        configured_groups = config.advanced_settings.get(
            "evaluation_metric_groups", []
        ) or []
        if isinstance(configured_groups, str):
            configured_groups = [configured_groups]
        config.advanced_settings["evaluation_metric_groups"] = list(
            dict.fromkeys([*configured_groups, *args.evaluation_metric_group])
        )
    if args.binder_chain:
        config.target_settings["binder_chain"] = args.binder_chain
    if args.target_chains:
        config.target_settings["chains"] = args.target_chains
    pipeline = BinderDesignPipeline(config)

    design_path = Path(config.target_settings["design_path"])
    trajectory_dir = Path(pipeline.design_paths["Trajectory"])
    mpnn_csv = design_path / "mpnn_design_stats.csv"
    final_csv = design_path / "final_design_stats.csv"
    key = jax.random.key(args.seed)

    if args.mode == "evaluate":
        inputs = list(args.input_pdb)
        if args.input_dir is not None:
            if not args.input_dir.is_dir():
                print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
                return 1
            inputs.extend(sorted(args.input_dir.glob("*.pdb")))
        # Preserve order while avoiding duplicate work when a PDB is supplied
        # both explicitly and through --input-dir.
        inputs = list(dict.fromkeys(path.resolve() for path in inputs))
        if not inputs:
            print(
                "Evaluate mode requires --input-pdb and/or --input-dir",
                file=sys.stderr,
            )
            return 1

        evaluation_csv = design_path / "evaluation_stats.csv"
        records = []
        for input_pdb in inputs:
            key, sub = jax.random.split(key)
            try:
                record = pipeline.evaluate_pdb(input_pdb, key=sub)
            except Exception:
                logger.exception("Error evaluating %s", input_pdb)
                continue
            records.append(record)
            pipeline.file_design(record)
        pipeline.write_csv(evaluation_csv, records)
        accepted = sum(record.accepted for record in records)
        print(
            f"Evaluated {len(records)}/{len(inputs)} complexes with "
            f"{pipeline.folding_selection.name}; {accepted} passed filters. "
            f"Results: {evaluation_csv}"
        )
        return 0 if records else 1

    pipeline.load_seen_sequences(mpnn_csv)

    if args.mode == "filter":
        trajectories = sorted(trajectory_dir.glob("*.pdb"))
        if args.max_trajectories:
            trajectories = trajectories[: args.max_trajectories]
        if not trajectories:
            print(f"No trajectories found in {trajectory_dir}", file=sys.stderr)
            return 1
        accepted = attempted = 0
        for trajectory in trajectories:
            key, sub = jax.random.split(key)
            try:
                records = pipeline.run_trajectory(trajectory, key=sub)
            except Exception:
                logger.exception("Error processing %s", trajectory.name)
                continue
            pipeline.write_csv(mpnn_csv, records)
            for record in records:
                pipeline.file_design(record)
            attempted += len(records)
            accepted += sum(1 for record in records if record.accepted)
        ranked = pipeline.write_final_csv(mpnn_csv, final_csv)
        print(f"Ranked {ranked} accepted designs into {final_csv}")
        print(
            f"Processed {len(trajectories)} trajectories, "
            f"{attempted} designs, {accepted} accepted"
        )
        return 0

    # trajectory / full: drive stage 1 ourselves.
    target = int(
        args.max_trajectories
        or config.advanced_settings.get("max_trajectories")
        or config.target_settings.get("number_of_final_designs", 1)
    )
    # max_trajectories bounds the work; number_of_final_designs is what the run
    # is actually for, and reaching it stops generation early.
    final_target = int(config.target_settings.get("number_of_final_designs", 0) or 0)
    master_seed = config.advanced_settings.get("trajectory_random_seed", args.seed)
    rng = np.random.default_rng(int(master_seed))

    backbones: list[Path] = []
    accepted = attempted = 0
    for index in range(target):
        seed = pipeline.sample_seed(rng)
        length = pipeline.sample_length(rng)
        logger.info("Trajectory %d/%d (length %d, seed %d)", index + 1, target, length, seed)
        backbone, _ = pipeline.generate_trajectory(seed=seed, length=length)
        if backbone is None:
            continue
        backbones.append(backbone)
        if args.mode != "full":
            continue
        key, sub = jax.random.split(key)
        try:
            records = pipeline.run_trajectory(backbone, key=sub)
        except Exception:
            logger.exception("Error processing %s", backbone.name)
            continue
        pipeline.write_csv(mpnn_csv, records)
        for record in records:
            pipeline.file_design(record)
        attempted += len(records)
        accepted += sum(1 for record in records if record.accepted)
        if final_target and pipeline.accepted_design_count() >= final_target:
            logger.info("Target number of final designs reached")
            break

    if args.mode == "full":
        ranked = pipeline.write_final_csv(mpnn_csv, final_csv)
        print(f"Ranked {ranked} accepted designs into {final_csv}")
        print(
            f"Generated {len(backbones)}/{target} usable backbones, "
            f"{attempted} designs, {accepted} accepted"
        )
    else:
        print(f"Generated {len(backbones)}/{target} usable backbones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
