"""The full binder design pipeline, running natively inside Mosaic.

Four stages, all in one Python 3.12 process:

1. **Trajectory** -- hallucinate a binder backbone against the target
   (:mod:`mosaic.binder_design.trajectory`).
2. **Sequence design** -- ProteinMPNN with fixed target/interface positions
   (:mod:`mosaic.binder_design.mpnn`).
3. **Validation** -- refold each design as a complex and on its own
   (:mod:`mosaic.binder_design.validation`).
4. **Filtering** -- PyRosetta interface scoring, DSSP secondary structure,
   clash and RMSD checks (:mod:`mosaic.binder_design.rosetta`,
   :mod:`mosaic.binder_design.geometry`, :mod:`mosaic.binder_design.filters`).

Output layout and CSV columns match DdCraft so existing configs, filter files
and analysis scripts keep working.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import jax
import numpy as np

from mosaic.binder_design import filters as filter_utils
from mosaic.binder_design import geometry
from mosaic.binder_design.components import create_structure_model, model_selection
from mosaic.binder_design.io import normalize_binder_pdb, normalize_complex_pdb
from mosaic.binder_design.labels import (
    CORE_LABELS,
    design_labels,
    final_labels,
)
from mosaic.binder_design.mpnn import DesignedSequence, MPNNDesigner
from mosaic.binder_design.validation import ComplexPredictor, MonomerPredictor

logger = logging.getLogger(__name__)

__all__ = ["BinderDesignPipeline", "DesignRecord", "PipelineConfig"]

DESIGN_DIRECTORIES = (
    "Accepted",
    "Accepted/Ranked",
    "Accepted/Animation",
    "Accepted/Plots",
    "Accepted/Pickle",
    "Trajectory",
    "Trajectory/Relaxed",
    "Trajectory/Plots",
    "Trajectory/Clashing",
    "Trajectory/LowConfidence",
    "Trajectory/Animation",
    "Trajectory/Pickle",
    "MPNN",
    "MPNN/Binder",
    "MPNN/Sequences",
    "MPNN/Relaxed",
    "MPNN/Failed",
    "Rejected",
)


def load_helicity(advanced_settings: dict[str, Any]) -> float:
    """Per-trajectory helicity bias, sampled or preset, as DdCraft does."""
    if advanced_settings.get("random_helicity") is True:
        return round(float(np.random.uniform(-3, 1)), 2)
    return float(advanced_settings.get("weights_helicity", 0) or 0)


def generate_directories(design_path: str | Path) -> dict[str, str]:
    """Create DdCraft's output tree and return the path lookup it uses."""
    design_path = Path(design_path)
    paths = {}
    for name in DESIGN_DIRECTORIES:
        directory = design_path / name
        directory.mkdir(parents=True, exist_ok=True)
        paths[name] = str(directory)
    return paths


@dataclass
class PipelineConfig:
    target_settings: dict[str, Any]
    advanced_settings: dict[str, Any]
    filters: dict[str, Any]

    @property
    def binder_chain(self) -> str:
        # Launchers use an empty value to mean a de-novo binder. Downstream
        # PDB normalization still needs a concrete one-character chain ID.
        return str(self.target_settings.get("binder_chain", "") or "").strip() or "B"

    @property
    def target_chains(self) -> str:
        return self.target_settings["chains"]

    @property
    def first_target_chain(self) -> str:
        return self.target_chains.split(",")[0].strip()

    @classmethod
    def from_files(
        cls,
        target: str | Path,
        advanced: str | Path,
        filters: str | Path,
        root: str | Path | None = None,
    ) -> PipelineConfig:
        config = cls(
            target_settings=json.loads(Path(target).read_text()),
            advanced_settings=json.loads(Path(advanced).read_text()),
            filters=json.loads(Path(filters).read_text()),
        )
        config.resolve_paths(root)
        return config

    def resolve_paths(self, root: str | Path | None = None) -> None:
        """Expand ``${DDCRAFT_DIR}`` and fill in the default asset locations.

        Mirrors DdCraft's ``perform_advanced_settings_check`` so the same config
        files work unchanged.
        """
        root = Path(root or os.environ.get("DDCRAFT_DIR", ".")).resolve()

        for key, value in list(self.target_settings.items()):
            if isinstance(value, str) and "${DDCRAFT_DIR}" in value:
                self.target_settings[key] = value.replace("${DDCRAFT_DIR}", str(root))

        defaults = {
            "af_params_dir": str(root),
            "dssp_path": str(root / "functions" / "dssp"),
            "dalphaball_path": str(root / "functions" / "DAlphaBall.gcc"),
        }
        for key, default in defaults.items():
            if not self.advanced_settings.get(key):
                self.advanced_settings[key] = default

        omit = self.advanced_settings.get("omit_AAs")
        if omit in (None, False, ""):
            self.advanced_settings["omit_AAs"] = None
        elif isinstance(omit, str):
            self.advanced_settings["omit_AAs"] = omit.strip()

    def validation_models(self) -> list[int]:
        """AF2 models used for validation.

        DdCraft keeps design and validation models disjoint when designing with
        the multimer weights, so validation never sees a model that shaped the
        backbone.
        """
        configured = self.advanced_settings.get("predict_models")
        if configured:
            return [int(m) for m in configured]
        if self.advanced_settings.get("use_multimer_design", True):
            return [3, 4]
        return [0, 1, 2, 3, 4]


@dataclass
class DesignRecord:
    """One MPNN design with every metric the filters can reference."""

    name: str
    sequence: str
    values: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    unmet: list[str] = field(default_factory=list)

    def row(self, labels: Iterable[str]) -> list[Any]:
        return [self.values.get(label) for label in labels]


def _average(values: list[Any]) -> Any:
    present = [v for v in values if v is not None]
    if not present:
        return None
    if isinstance(present[0], dict):
        keys = set().union(*(set(v) for v in present))
        return {k: float(np.mean([v.get(k, 0) for v in present])) for k in keys}
    return float(np.mean(present))


def _spread(
    values: dict[int, dict[str, Any]], model_indices: list[int]
) -> dict[str, Any]:
    """Turn per-model metric dicts into DdCraft's Average_/1_../5_ columns."""
    out: dict[str, Any] = {}
    for label in CORE_LABELS:
        per_model = []
        for model_number in range(1, 6):
            value = values.get(model_number, {}).get(label)
            out[f"{model_number}_{label}"] = value
            if model_number in [i + 1 for i in model_indices]:
                per_model.append(value)
        out[f"Average_{label}"] = _average(per_model)
    return out


class BinderDesignPipeline:
    """Runs stages 2-4 for trajectories produced by stage 1."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        design_paths: dict[str, str] | None = None,
        structure_model=None,
    ):
        self.config = config
        self.target_settings = config.target_settings
        self.advanced_settings = config.advanced_settings
        self.filters = config.filters
        self.design_paths = design_paths or generate_directories(
            self.target_settings["design_path"]
        )

        self.hallucination_selection = model_selection(
            self.advanced_settings, "hallucination"
        )
        self.folding_selection = model_selection(self.advanced_settings, "folding")
        self._designer = None
        self._rosetta = None
        self._structure_model = structure_model
        self._monomer_model = None
        self.design_labels = design_labels()
        self.final_labels = final_labels()
        self._seen_sequences: set[str] = set()

    @property
    def designer(self) -> MPNNDesigner:
        """Load ProteinMPNN only when sequence design is actually requested."""
        if self._designer is None:
            self._designer = MPNNDesigner(
                weights=self.advanced_settings.get("mpnn_weights", "soluble"),
                backbone_noise=float(self.advanced_settings.get("backbone_noise", 0.0)),
                sampling_temp=float(self.advanced_settings.get("sampling_temp", 0.1)),
                num_seqs=int(self.advanced_settings.get("num_seqs", 20)),
                omit_AAs=self.advanced_settings.get("omit_AAs", "C"),
            )
        return self._designer

    def _load_rosetta(self):
        """Import and initialize PyRosetta only for filters that need it."""
        if self._rosetta is None:
            from mosaic.binder_design import rosetta

            rosetta.init_pyrosetta(self.advanced_settings.get("dalphaball_path", ""))
            self._rosetta = rosetta
        return self._rosetta

    def load_seen_sequences(self, mpnn_csv: str | Path) -> int:
        """Remember sequences already in the CSV so a resumed run skips them."""
        mpnn_csv = Path(mpnn_csv)
        if not mpnn_csv.exists():
            return 0
        with mpnn_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                sequence = (row.get("Sequence") or "").strip()
                if sequence:
                    self._seen_sequences.add(sequence)
        return len(self._seen_sequences)

    @property
    def structure_model(self):
        """The folding model used to validate the binder/target complex."""
        if self._structure_model is None:
            self._structure_model = create_structure_model(
                self.folding_selection,
                self.advanced_settings,
            )
        return self._structure_model

    @property
    def monomer_model(self):
        """The folding model used to predict the binder on its own.

        Deliberately *not* the complex model. DdCraft builds the binder-alone
        predictor with ``use_multimer=False``, and the multimer network scores a
        lone chain systematically lower: reusing it here failed
        ``5_Binder_pLDDT`` on 38% of designs where DdCraft failed none.
        """
        if self._monomer_model is None:
            if self.folding_selection.name == "af2":
                self._monomer_model = create_structure_model(
                    self.folding_selection,
                    self.advanced_settings,
                    multimer=False,
                    model_numbers=tuple(i + 1 for i in self._model_indices()),
                )
            else:
                self._monomer_model = self.structure_model
        return self._monomer_model

    # -- stage 2 ---------------------------------------------------------

    def _interface_spec(self, trajectory_pdb: str) -> tuple[str, list[int]]:
        interface = geometry.hotspot_residues(
            trajectory_pdb,
            binder_chain=self.config.binder_chain,
            atom_distance_cutoff=4.0,
        )
        residues = list(interface.keys())
        spec = ",".join(f"{self.config.binder_chain}{r}" for r in residues)
        return spec, residues

    def design_sequences(
        self, trajectory_pdb: str, *, key
    ) -> tuple[list[DesignedSequence], str]:
        interface_spec, _ = (
            self._interface_spec(trajectory_pdb)
            if self.advanced_settings.get("mpnn_fix_interface", True)
            else ("", [])
        )
        sequences = self.designer.design(
            trajectory_pdb,
            key=key,
            binder_chain=self.config.binder_chain,
            target_chain=self.config.first_target_chain,
            interface_residues=interface_spec,
            fix_pos=self.target_settings.get("fix_pos", ""),
        )
        return self._rank_sequences(sequences), interface_spec

    def _rank_sequences(
        self, sequences: list[DesignedSequence]
    ) -> list[DesignedSequence]:
        """Deduplicate, drop restricted sequences and sort by MPNN score.

        Every survivor is returned. ``max_mpnn_sequences`` is *not* applied
        here: DdCraft treats it as a cap on how many designs may be **accepted**
        per trajectory, and keeps evaluating candidates until it is reached.
        """
        restricted = self._restricted_amino_acids()
        best: dict[str, DesignedSequence] = {}
        for sequence in sequences:
            if sequence.sequence in self._seen_sequences:
                continue
            if restricted and any(aa in sequence.sequence.upper() for aa in restricted):
                continue
            existing = best.get(sequence.sequence)
            if existing is None or sequence.score < existing.score:
                best[sequence.sequence] = sequence
        return sorted(best.values(), key=lambda s: s.score)

    def _restricted_amino_acids(self) -> set[str]:
        if not self.advanced_settings.get("force_reject_AA", False):
            return set()
        omit = str(self.advanced_settings.get("omit_AAs", ""))
        return {aa.strip().upper() for aa in omit.split(",") if aa.strip()}

    # -- stage 1 ---------------------------------------------------------

    def design_models(self) -> list[int]:
        """Configured members/samples used for the trajectory stage."""
        if self.hallucination_selection.members is not None:
            return list(self.hallucination_selection.members)
        if self.hallucination_selection.name != "af2":
            return [0]
        configured_numbers = self.hallucination_selection.options.get("model_numbers")
        if configured_numbers is not None:
            return [int(number) - 1 for number in configured_numbers]
        if self.advanced_settings.get("use_multimer_design", True):
            return [0, 1, 2]
        return [0, 1]

    def sample_seed(self, rng) -> int:
        return int(rng.integers(0, 999999))

    def sample_length(self, rng) -> int:
        min_length, max_length = self.target_settings["lengths"]
        return int(rng.integers(min_length, max_length + 1))

    def generate_trajectory(self, *, seed: int, length: int):
        """Hallucinate one binder backbone.

        Returns ``(pdb, metrics)`` where ``pdb`` is ``None`` when the backbone
        was rejected as clashing or low confidence, matching DdCraft, which
        files those trajectories away instead of designing sequences for them.
        """
        from mosaic.binder_design.trajectory import run_job

        design_name = f"{self.target_settings['binder_name']}_l{length}_s{seed}"
        raw_dir = Path(self.target_settings["design_path"]) / "Mosaic" / "Raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_pdb = raw_dir / f"{design_name}.pdb"

        target_chains = [
            chain.strip()
            for chain in str(self.target_settings["chains"]).split(",")
            if chain.strip()
        ]
        binder_chain = str(self.target_settings.get("binder_chain", "")).strip()

        result = run_job(
            {
                "schema_version": 1,
                "design_name": design_name,
                "seed": seed,
                "length": length,
                "helicity_value": load_helicity(self.advanced_settings),
                "starting_pdb": self.target_settings["starting_pdb"],
                "target_chains": target_chains,
                "binder_chain": binder_chain,
                "target_settings": self.target_settings,
                "advanced_settings": self.advanced_settings,
                "design_models": self.design_models(),
                "af_params_dir": self.advanced_settings["af_params_dir"],
                "output_pdb": str(raw_pdb),
            }
        )

        trajectory_pdb = Path(self.design_paths["Trajectory"]) / f"{design_name}.pdb"
        normalize_complex_pdb(
            raw_pdb=raw_pdb,
            output_pdb=trajectory_pdb,
            starting_pdb=Path(self.target_settings["starting_pdb"]),
            target_chain_ids=target_chains,
            binder_chain_id=binder_chain or self.config.binder_chain,
            binder_length=length,
            structure_conditioned=bool(binder_chain),
        )
        raw_pdb.unlink(missing_ok=True)

        metrics = result.get("metrics", {})
        rejection = self._trajectory_rejection(trajectory_pdb, result)
        if rejection is not None:
            (Path(self.design_paths[rejection]) / trajectory_pdb.name).write_bytes(
                trajectory_pdb.read_bytes()
            )
            trajectory_pdb.unlink()
            logger.warning("Trajectory %s filed under %s", design_name, rejection)
            return None, metrics
        return trajectory_pdb, metrics

    def _trajectory_rejection(self, trajectory_pdb: Path, result: dict) -> str | None:
        """Where a backbone belongs, or ``None`` when it is usable."""
        if result.get("status") == "low_confidence":
            return "Trajectory/LowConfidence"

        plddt = result.get("metrics", {}).get("plddt")
        threshold = self.advanced_settings.get("start_monitoring_plddt")
        if (
            plddt is not None
            and threshold is not None
            and float(plddt) < float(threshold)
        ):
            return "Trajectory/LowConfidence"

        clashes = geometry.calculate_clash_score(str(trajectory_pdb), 2.4)
        limit = self.advanced_settings.get("max_trajectory_clashes")
        if limit is not None and clashes > float(limit):
            return "Trajectory/Clashing"
        return None

    # -- stages 3 and 4 --------------------------------------------------

    def _model_indices(self) -> list[int]:
        selection = getattr(self, "folding_selection", None)
        if selection is None:
            validation_models = getattr(self.config, "validation_models", None)
            if validation_models is not None:
                return validation_models()
            return [
                int(member)
                for member in self.advanced_settings.get("predict_models", [3, 4])
            ]
        if selection.members is not None:
            return list(selection.members)
        if selection.name == "af2":
            configured_numbers = selection.options.get("model_numbers")
            if configured_numbers is not None:
                return [int(number) - 1 for number in configured_numbers]
            return self.config.validation_models()
        return list(self.structure_model.ensemble_members())

    def _folding_sampling_steps(self) -> int | None:
        selection = getattr(self, "folding_selection", None)
        return None if selection is None else selection.sampling_steps

    #: Metrics the complex prediction yields directly. The gate below must not
    #: reference anything computed later (binder-alone confidences, PyRosetta
    #: scores), otherwise every design fails on a missing value.
    BASE_METRICS = ("pLDDT", "pTM", "i_pTM", "pAE", "i_pAE", "IPSAE")

    MONOMER_METRICS = frozenset(
        {
            "Binder_pLDDT",
            "Binder_pTM",
            "Binder_pAE",
            "Binder_RMSD",
        }
    )
    PYROSETTA_METRICS = frozenset(
        {
            "Relaxed_Clashes",
            "Binder_Energy_Score",
            "Surface_Hydrophobicity",
            "ShapeComplementarity",
            "PackStat",
            "dG",
            "dSASA",
            "dG/dSASA",
            "Interface_SASA_%",
            "Interface_Hydrophobicity",
            "n_InterfaceResidues",
            "n_InterfaceHbonds",
            "InterfaceHbondsPercentage",
            "n_InterfaceUnsatHbonds",
            "InterfaceUnsatHbondsPercentage",
        }
    )
    SECONDARY_STRUCTURE_METRICS = frozenset(
        {
            "Binder_Helix%",
            "Binder_BetaSheet%",
            "Binder_Loop%",
            "Interface_Helix%",
            "Interface_BetaSheet%",
            "Interface_Loop%",
            "i_pLDDT",
            "ss_pLDDT",
        }
    )
    STRUCTURE_METRICS = frozenset(
        set(CORE_LABELS) - set(BASE_METRICS) - MONOMER_METRICS
    )
    EVALUATION_METRIC_GROUPS: ClassVar[dict[str, frozenset[str]]] = {
        # Complex confidence is always returned by the folding model. Keeping
        # it as an explicit group makes batch manifests self-describing.
        "confidence": frozenset(BASE_METRICS),
        # Binder_RMSD uses PyRosetta alignment, so the weight-free monomer group
        # contains only confidence values from the binder-alone fold.
        "monomer": frozenset(
            {"Binder_pLDDT", "Binder_pTM", "Binder_pAE"}
        ),
        "geometry": frozenset(
            {"Unrelaxed_Clashes", "Hotspot_RMSD", "Target_RMSD", "InterfaceAAs"}
        ),
        "dssp": SECONDARY_STRUCTURE_METRICS,
        "pyrosetta": frozenset({*PYROSETTA_METRICS, "Binder_RMSD"}),
    }

    @staticmethod
    def _base_metric_name(label: str) -> str:
        if label.startswith("Average_"):
            return label[len("Average_") :]
        prefix, separator, remainder = label.partition("_")
        if separator and prefix in {"1", "2", "3", "4", "5"}:
            return remainder
        return label

    def _active_metrics(self) -> set[str]:
        """Metrics requested explicitly or by a non-null filter threshold.

        ``evaluation_metric_groups`` requests computation only. Acceptance is
        still controlled exclusively by ``filters``. This separation lets an
        evaluator collect PyRosetta or DSSP distributions before choosing any
        thresholds and keeps those dependencies lazy when no group/filter asks
        for them.
        """
        active: set[str] = set()
        configured_groups = getattr(self, "advanced_settings", {}).get(
            "evaluation_metric_groups", []
        ) or []
        if isinstance(configured_groups, str):
            configured_groups = [configured_groups]
        if not isinstance(configured_groups, (list, tuple, set)):
            raise TypeError("evaluation_metric_groups must be a JSON array or string")
        for group in configured_groups:
            name = str(group).strip().lower()
            try:
                active.update(self.EVALUATION_METRIC_GROUPS[name])
            except KeyError:
                choices = ", ".join(self.EVALUATION_METRIC_GROUPS)
                raise ValueError(
                    f"Unknown evaluation metric group '{group}'. Choose one of: {choices}"
                ) from None
        for label, conditions in self.filters.items():
            if self._base_metric_name(label) == "InterfaceAAs":
                if any(
                    condition.get("threshold") is not None
                    for condition in conditions.values()
                ):
                    active.add("InterfaceAAs")
            elif conditions.get("threshold") is not None:
                active.add(self._base_metric_name(label))
        return active

    def _base_confidence_filters_pass(self, values: dict[str, Any]) -> list[str]:
        """Cheap confidence gate, run before any PyRosetta work.

        Only the *per-model* thresholds of the models actually predicted apply,
        exactly as in ``predict_binder_complex``. The ``Average_*`` thresholds
        deliberately play no part: they are enforced later, on the full record.

        The distinction is not academic. Validation runs models 4 and 5, and
        production filter sets leave ``4_*``/``5_*`` at ``null`` while setting
        ``Average_*``, which makes this gate a no-op there. Including the
        averages rejected 45% of designs before they were ever scored.
        """
        labels = {
            f"{index}_{metric}"
            for index in (i + 1 for i in self._model_indices())
            for metric in self.BASE_METRICS
        }
        cheap = {
            label: conditions
            for label, conditions in self.filters.items()
            if label in labels
        }
        return filter_utils.unmet_filters(values, cheap)

    # Backward-compatible private name for callers/tests from the AF2-only era.
    _base_af2_filters_pass = _base_confidence_filters_pass

    def evaluate_design(
        self,
        design: DesignedSequence,
        *,
        design_name: str,
        trajectory_pdb: str,
        complex_predictor: ComplexPredictor,
        monomer_predictor: MonomerPredictor | None,
        key,
        reference_pdb: str | None = None,
        configured_only: bool = False,
    ) -> DesignRecord:
        model_indices = self._model_indices()
        reference_pdb = reference_pdb or trajectory_pdb
        requested_metrics = self._active_metrics() if configured_only else None
        record = DesignRecord(name=design_name, sequence=design.sequence)
        record.values.update(
            {
                "Design": design_name,
                "Protocol": f"folding:{self.folding_selection.name}",
                "Sequence": design.sequence,
                "Length": len(design.sequence),
                "MPNN_score": design.score,
                "MPNN_seq_recovery": design.seqid,
            }
        )

        key, complex_key, monomer_key = jax.random.split(key, 3)
        predictions = complex_predictor.predict(
            design.sequence, key=complex_key, model_indices=model_indices
        )

        per_model: dict[int, dict[str, Any]] = {}
        complex_pdbs: dict[int, str] = {}
        for prediction in predictions:
            model_number = prediction.model_index + 1
            metrics = prediction.metrics
            per_model[model_number] = {
                "pLDDT": metrics.plddt,
                "pTM": metrics.ptm,
                "i_pTM": metrics.i_ptm,
                "pAE": metrics.pae,
                "i_pAE": metrics.i_pae,
                "IPSAE": metrics.ipsae,
            }
            raw = prediction.write_pdb(
                Path(self.design_paths["MPNN"])
                / f"{design_name}_model{model_number}.raw.pdb"
            )
            complex_pdbs[model_number] = str(
                normalize_complex_pdb(
                    raw,
                    Path(self.design_paths["MPNN"])
                    / f"{design_name}_model{model_number}.pdb",
                    reference_pdb,
                    [
                        c.strip()
                        for c in self.config.target_chains.split(",")
                        if c.strip()
                    ],
                    self.config.binder_chain,
                    len(design.sequence),
                    structure_conditioned=configured_only,
                )
            )
            raw.unlink()

        record.values.update(_spread(per_model, model_indices))

        unmet = self._base_confidence_filters_pass(record.values)
        if unmet:
            record.unmet = unmet
            return record

        logger.info(
            "%s passed confidence filters, proceeding with requested calculations",
            design_name,
        )

        needs_monomer = requested_metrics is None or bool(
            requested_metrics & self.MONOMER_METRICS
        )
        if needs_monomer:
            if monomer_predictor is None:
                raise RuntimeError(
                    "monomer predictor is required by the active filters"
                )
            monomer = monomer_predictor.predict(
                design.sequence, key=monomer_key, model_indices=model_indices
            )
            for prediction in monomer:
                model_number = prediction.model_index + 1
                raw = prediction.write_pdb(
                    Path(self.design_paths["MPNN/Binder"])
                    / f"{design_name}_model{model_number}.raw.pdb"
                )
                binder_pdb = normalize_binder_pdb(
                    raw,
                    Path(self.design_paths["MPNN/Binder"])
                    / f"{design_name}_model{model_number}.pdb",
                    len(design.sequence),
                )
                raw.unlink()
                metrics = {
                    "Binder_pLDDT": prediction.metrics.plddt,
                    "Binder_pTM": prediction.metrics.ptm,
                    "Binder_pAE": prediction.metrics.pae,
                }
                if requested_metrics is None or "Binder_RMSD" in requested_metrics:
                    rosetta = self._load_rosetta()
                    rosetta.align_pdbs(
                        reference_pdb,
                        str(binder_pdb),
                        self.config.binder_chain,
                        "A",
                    )
                    metrics["Binder_RMSD"] = rosetta.unaligned_rmsd(
                        reference_pdb,
                        str(binder_pdb),
                        self.config.binder_chain,
                        "A",
                    )
                per_model[model_number].update(metrics)
                if self.advanced_settings.get("remove_binder_monomer", True):
                    binder_pdb.unlink(missing_ok=True)

        needs_structure_scores = requested_metrics is None or bool(
            requested_metrics & self.STRUCTURE_METRICS
        )
        if needs_structure_scores:
            for model_number, complex_pdb in complex_pdbs.items():
                per_model[model_number].update(
                    self._score_structure(
                        complex_pdb,
                        design_name,
                        model_number,
                        reference_pdb,
                        requested_metrics=requested_metrics,
                    )
                )

        record.values.update(_spread(per_model, model_indices))
        unmet = filter_utils.unmet_filters(record.values, self.filters)
        record.unmet = unmet
        record.accepted = not unmet
        return record

    def _score_structure(
        self,
        complex_pdb: str,
        design_name: str,
        model_number: int,
        trajectory_pdb: str,
        *,
        requested_metrics: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run only the requested structure-filter groups for one complex."""
        binder_chain = self.config.binder_chain

        def wants(metric: str) -> bool:
            return requested_metrics is None or metric in requested_metrics

        values: dict[str, Any] = {}
        if wants("Unrelaxed_Clashes"):
            values["Unrelaxed_Clashes"] = geometry.calculate_clash_score(
                complex_pdb, 2.4
            )

        relaxed_pdb = str(
            Path(self.design_paths["MPNN/Relaxed"])
            / f"{design_name}_model{model_number}.pdb"
        )
        needs_pyrosetta = requested_metrics is None or bool(
            requested_metrics & self.PYROSETTA_METRICS
        )
        scored_pdb = complex_pdb
        interface_aa = None
        if needs_pyrosetta:
            rosetta = self._load_rosetta()
            rosetta.pr_relax(complex_pdb, relaxed_pdb, binder_chain=binder_chain)
            scores, interface_aa, _ = rosetta.score_interface(
                relaxed_pdb, binder_chain=binder_chain
            )
            scored_pdb = relaxed_pdb
            score_map = {
                "Binder_Energy_Score": "binder_score",
                "Surface_Hydrophobicity": "surface_hydrophobicity",
                "ShapeComplementarity": "interface_sc",
                "PackStat": "interface_packstat",
                "dG": "interface_dG",
                "dSASA": "interface_dSASA",
                "dG/dSASA": "interface_dG_SASA_ratio",
                "Interface_SASA_%": "interface_fraction",
                "Interface_Hydrophobicity": "interface_hydrophobicity",
                "n_InterfaceResidues": "interface_nres",
                "n_InterfaceHbonds": "interface_interface_hbonds",
                "InterfaceHbondsPercentage": "interface_hbond_percentage",
                "n_InterfaceUnsatHbonds": "interface_delta_unsat_hbonds",
                "InterfaceUnsatHbondsPercentage": (
                    "interface_delta_unsat_hbonds_percentage"
                ),
            }
            for metric, score_key in score_map.items():
                if wants(metric):
                    values[metric] = scores.get(score_key)
            if wants("Relaxed_Clashes"):
                values["Relaxed_Clashes"] = geometry.calculate_clash_score(
                    relaxed_pdb, 2.4
                )

        if wants("InterfaceAAs"):
            if interface_aa is None:
                interface_aa = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
                for amino_acid in geometry.hotspot_residues(
                    scored_pdb, binder_chain=binder_chain
                ).values():
                    if amino_acid in interface_aa:
                        interface_aa[amino_acid] += 1
            values["InterfaceAAs"] = interface_aa

        if requested_metrics is None or bool(
            requested_metrics & self.SECONDARY_STRUCTURE_METRICS
        ):
            # (binder helix/sheet/loop, interface helix/sheet/loop,
            #  interface pLDDT, secondary-structure pLDDT)
            ss = geometry.calc_ss_percentage(
                scored_pdb, self.advanced_settings, binder_chain
            )
            for metric, value in zip(
                (
                    "Binder_Helix%",
                    "Binder_BetaSheet%",
                    "Binder_Loop%",
                    "Interface_Helix%",
                    "Interface_BetaSheet%",
                    "Interface_Loop%",
                    "i_pLDDT",
                    "ss_pLDDT",
                ),
                ss,
            ):
                if wants(metric):
                    values[metric] = value

        if wants("Hotspot_RMSD"):
            values["Hotspot_RMSD"] = geometry.hotspot_rmsd(
                trajectory_pdb,
                scored_pdb,
                self.config.first_target_chain,
                binder_chain,
            )
        if wants("Target_RMSD"):
            values["Target_RMSD"] = geometry.chain_rmsd(
                trajectory_pdb, scored_pdb, self.config.first_target_chain
            )
        return values

    # -- drivers ---------------------------------------------------------

    def evaluate_pdb(
        self,
        complex_pdb: str | Path,
        *,
        key,
        design_name: str | None = None,
    ) -> DesignRecord:
        """Evaluate an existing binder/target complex without hallucination or MPNN.

        The binder sequence is read from ``config.binder_chain`` in the input
        PDB. The configured folding model refolds the complex, then only filter
        groups with active thresholds are run. This makes confidence-only,
        model+DSSP/geometry, and model+PyRosetta evaluation independent entry
        points into the same result schema.
        """
        import gemmi

        complex_pdb = Path(complex_pdb)
        if not complex_pdb.is_file():
            raise FileNotFoundError(f"Evaluation PDB not found: {complex_pdb}")
        structure = gemmi.read_structure(str(complex_pdb))
        if not len(structure):
            raise ValueError(f"No model found in evaluation PDB: {complex_pdb}")
        chains = {chain.name: chain for chain in structure[0]}
        required = {
            self.config.binder_chain,
            *(
                chain.strip()
                for chain in self.config.target_chains.split(",")
                if chain.strip()
            ),
        }
        missing = sorted(required - set(chains))
        if missing:
            raise ValueError(
                f"{complex_pdb}: required chains {', '.join(missing)} not found"
            )
        sequence = gemmi.one_letter_code(
            [residue.name for residue in chains[self.config.binder_chain]]
        ).upper()
        if not sequence or "X" in sequence:
            raise ValueError(
                f"{complex_pdb}: binder chain {self.config.binder_chain} has "
                "missing or unsupported protein residues"
            )

        started = time.time()
        model_indices = self._model_indices()
        complex_predictor = ComplexPredictor(
            self.structure_model,
            target_pdb=complex_pdb,
            target_chains=self.config.target_chains,
            binder_length=len(sequence),
            recycling_steps=self._recycling_steps(),
            model_indices=model_indices,
            sampling_steps=self._folding_sampling_steps(),
            ipsae_cutoff=self._ipsae_cutoff(),
            advanced_settings=self.advanced_settings,
        )
        active_metrics = self._active_metrics()
        monomer_predictor = None
        if active_metrics & self.MONOMER_METRICS:
            monomer_predictor = MonomerPredictor(
                self.monomer_model,
                binder_length=len(sequence),
                recycling_steps=self._recycling_steps(),
                model_indices=model_indices,
                sampling_steps=self._folding_sampling_steps(),
            )

        name = design_name or complex_pdb.stem
        design = DesignedSequence(
            sequence=sequence,
            score=None,
            seqid=None,
            full_sequence=sequence,
        )
        record = self.evaluate_design(
            design,
            design_name=name,
            trajectory_pdb=str(complex_pdb),
            reference_pdb=str(complex_pdb),
            complex_predictor=complex_predictor,
            monomer_predictor=monomer_predictor,
            key=key,
            configured_only=True,
        )
        record.values["Protocol"] = f"evaluation:{self.folding_selection.name}"
        record.values["DesignTime"] = round(time.time() - started, 2)
        return record

    def run_trajectory(self, trajectory_pdb: str | Path, *, key) -> list[DesignRecord]:
        """Stages 2-4 for one trajectory backbone."""
        trajectory_pdb = str(trajectory_pdb)
        stem = Path(trajectory_pdb).stem
        started = time.time()

        length = geometry.get_chain_length(trajectory_pdb, self.config.binder_chain)
        key, design_key = jax.random.split(key)
        sequences, interface_spec = self.design_sequences(
            trajectory_pdb, key=design_key
        )
        if not sequences:
            logger.warning("No valid MPNN sequences generated for %s", stem)
            return []

        complex_predictor = ComplexPredictor(
            self.structure_model,
            target_pdb=self.target_settings["starting_pdb"],
            target_chains=self.config.target_chains,
            binder_length=length,
            recycling_steps=self._recycling_steps(),
            model_indices=self._model_indices(),
            sampling_steps=self._folding_sampling_steps(),
            ipsae_cutoff=self._ipsae_cutoff(),
            advanced_settings=self.advanced_settings,
        )
        monomer_predictor = MonomerPredictor(
            self.monomer_model,
            binder_length=length,
            recycling_steps=self._recycling_steps(),
            model_indices=self._model_indices(),
            sampling_steps=self._folding_sampling_steps(),
        )

        records = []
        accepted_target = int(self.advanced_settings.get("max_mpnn_sequences", 2) or 0)
        accepted_count = 0
        for number, design in enumerate(sequences, start=1):
            if accepted_target and accepted_count >= accepted_target:
                logger.debug(
                    "%s reached %d accepted designs, skipping the remaining %d "
                    "candidates",
                    stem,
                    accepted_count,
                    len(sequences) - number + 1,
                )
                break
            key, design_key = jax.random.split(key)
            try:
                record = self.evaluate_design(
                    design,
                    design_name=f"{stem}_mpnn{number}",
                    trajectory_pdb=trajectory_pdb,
                    complex_predictor=complex_predictor,
                    monomer_predictor=monomer_predictor,
                    key=design_key,
                )
            except Exception:
                # PyRosetta occasionally fails on a single structure. DdCraft
                # logs and moves to the next candidate rather than abandoning
                # every remaining trajectory, so a transient error costs one
                # design instead of the whole run.
                logger.exception("Error processing %s_mpnn%d", stem, number)
                self._seen_sequences.add(design.sequence)
                continue
            record.values.setdefault("InterfaceResidues", interface_spec)
            record.values.setdefault("DesignTime", round(time.time() - started, 2))
            records.append(record)
            self._seen_sequences.add(design.sequence)
            accepted_count += bool(record.accepted)
            logger.info(
                "%s %s", record.name, "ACCEPTED" if record.accepted else "FILTER_FAIL"
            )

        gc.collect()
        return records

    def _recycling_steps(self) -> int:
        """Forward passes for validation.

        ColabDesign runs ``num_recycles + 1`` passes, while Mosaic's recycling
        scan length *is* the number of passes. The same off-by-one bit stage 1
        and showed up there as systematically lower pLDDT.
        """
        return int(self.advanced_settings.get("num_recycles_validation", 3)) + 1

    def _ipsae_cutoff(self) -> float | None:
        config = self.advanced_settings.get("ipsae", {})
        if isinstance(config, dict) and config.get("enabled", False):
            return float(config.get("cutoff", 10.0))
        return None

    def accepted_design_count(self) -> int:
        """How many accepted designs are on disk.

        Counted from the Accepted directory rather than from this run's tally so
        that a resumed run stops at the same total, which is how DdCraft decides
        it has enough.
        """
        accepted = Path(self.design_paths["Accepted"])
        if not accepted.exists():
            return 0
        return sum(1 for f in accepted.iterdir() if f.suffix == ".pdb")

    def file_design(self, record: DesignRecord) -> Path | None:
        """Copy a design's best model into ``Accepted`` or ``Rejected``.

        DdCraft keeps one representative structure per design, picked by mean CA
        B-factor (the relaxed PDBs carry pLDDT there), so downstream tooling can
        glob a single directory instead of five per-model files.
        """
        best = self._best_model_pdb(record.name)
        if best is None:
            logger.warning("No folded model found for %s", record.name)
            return None
        folder = "Accepted" if record.accepted else "Rejected"
        destination = Path(self.design_paths[folder]) / f"{record.name}.pdb"
        destination.write_bytes(best.read_bytes())
        return destination

    def _best_model_pdb(self, design_name: str) -> Path | None:
        best: tuple[float, Path] | None = None
        # Prefer relaxed structures when PyRosetta ran. Confidence-only and
        # geometry-only evaluation intentionally produce only the folded PDBs.
        for directory in ("MPNN/Relaxed", "MPNN"):
            for model in range(1, 6):
                path = Path(self.design_paths[directory]) / (
                    f"{design_name}_model{model}.pdb"
                )
                if not path.exists():
                    continue
                try:
                    score = geometry.mean_ca_bfactor(str(path))
                except Exception as error:  # noqa: BLE001 - one bad PDB is skippable
                    logger.warning("Cannot rank folded model %s: %s", path, error)
                    continue
                if score is None:
                    continue
                if best is None or score > best[0]:
                    best = (score, path)
            if best is not None:
                break
        return best[1] if best else None

    def write_final_csv(self, mpnn_csv: str | Path, final_csv: str | Path) -> int:
        """Rank the accepted designs by ``Average_i_pTM`` and record them.

        Mirrors DdCraft's ``check_accepted_designs``: ranked copies land in
        ``Accepted/Ranked`` with a rank prefix, and the CSV is rewritten from
        scratch so re-running after more designs land renumbers everything.
        """
        mpnn_csv, final_csv = Path(mpnn_csv), Path(final_csv)
        accepted_dir = Path(self.design_paths["Accepted"])
        ranked_dir = Path(self.design_paths["Accepted/Ranked"])
        accepted = {path.stem: path for path in accepted_dir.glob("*.pdb")}
        if not accepted or not mpnn_csv.exists():
            return 0

        for stale in ranked_dir.glob("*.pdb"):
            stale.unlink()

        with mpnn_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        def sort_key(row: dict[str, str]) -> float:
            try:
                return -float(row.get("Average_i_pTM") or "nan")
            except ValueError:
                return float("inf")

        ranked_rows = []
        rank = 1
        for row in sorted(rows, key=sort_key):
            source = accepted.get(row.get("Design", ""))
            if source is None:
                continue
            (ranked_dir / f"{rank}_{source.name}").write_bytes(source.read_bytes())
            ranked_rows.append(
                {"Rank": rank, **{k: row.get(k, "") for k in self.design_labels}}
            )
            rank += 1

        with final_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.final_labels)
            writer.writeheader()
            writer.writerows(ranked_rows)
        return len(ranked_rows)

    def write_csv(self, path: str | Path, records: list[DesignRecord]) -> Path:
        path = Path(path)
        exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(self.design_labels)
            for record in records:
                writer.writerow(record.row(self.design_labels))
        return path
