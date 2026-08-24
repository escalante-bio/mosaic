"""Tests for the native-Mosaic binder design pipeline.

These deliberately avoid loading model weights or PyRosetta so they stay fast;
the numerical parity with DdCraft/ColabDesign is established separately by the
comparison runs recorded in the repository README.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mosaic.binder_design import filters as filter_utils
from mosaic.binder_design.labels import design_labels, final_labels, trajectory_labels
from mosaic.binder_design.validation import ComplexPredictor, MonomerPredictor
from mosaic.models.af2 import AlphaFold2
from mosaic.structure_prediction import StructurePredictionModel

# --- CSV schema ---------------------------------------------------------


def test_design_labels_have_average_and_per_model_columns():
    labels = design_labels()
    assert labels[:3] == ["Design", "Protocol", "Length"]
    for metric in ("pLDDT", "ShapeComplementarity", "Binder_RMSD"):
        assert f"Average_{metric}" in labels
        for model in range(1, 6):
            assert f"{model}_{metric}" in labels
    assert labels[-1] == "AdvancedSettings"


def test_label_counts_match_ddcraft():
    """DdCraft's generate_dataframe_labels produced exactly these widths."""
    assert len(trajectory_labels()) == 45
    assert len(design_labels()) == 237
    assert len(final_labels()) == 238
    assert final_labels()[0] == "Rank"
    assert final_labels()[1:] == design_labels()


# --- filters ------------------------------------------------------------


def test_filter_without_threshold_is_inactive():
    filters = {"Average_pLDDT": {"threshold": None, "higher": True}}
    assert filter_utils.check_filters({"Average_pLDDT": 0.1}, None, filters) is True


def test_higher_and_lower_filters():
    filters = {
        "Average_pLDDT": {"threshold": 0.8, "higher": True},
        "Average_pAE": {"threshold": 0.3, "higher": False},
    }
    assert filter_utils.check_filters({"Average_pLDDT": 0.9, "Average_pAE": 0.2}, None, filters) is True
    assert filter_utils.check_filters({"Average_pLDDT": 0.7, "Average_pAE": 0.2}, None, filters) == [
        "Average_pLDDT"
    ]
    assert filter_utils.check_filters({"Average_pLDDT": 0.9, "Average_pAE": 0.4}, None, filters) == [
        "Average_pAE"
    ]


def test_missing_per_model_value_passes_but_missing_average_fails():
    """DdCraft's asymmetry: not every model is necessarily predicted, but an
    averaged metric that was never computed must not silently accept."""
    filters = {
        "1_pLDDT": {"threshold": 0.8, "higher": True},
        "Average_pLDDT": {"threshold": 0.8, "higher": True},
    }
    assert filter_utils.unmet_filters({"1_pLDDT": None, "Average_pLDDT": 0.9}, filters) == []
    assert filter_utils.unmet_filters({"1_pLDDT": 0.9, "Average_pLDDT": None}, filters) == [
        "Average_pLDDT"
    ]


def test_interface_aa_filters_are_nested_by_amino_acid():
    filters = {
        "Average_InterfaceAAs": {
            "C": {"threshold": 0, "higher": False},
            "W": {"threshold": 2, "higher": False},
        }
    }
    values = {"Average_InterfaceAAs": {"C": 1, "W": 1}}
    assert filter_utils.unmet_filters(values, filters) == ["Average_InterfaceAAs_C"]
    assert filter_utils.unmet_filters({"Average_InterfaceAAs": None}, filters) == []


def test_check_filters_accepts_labels_and_values_in_parallel():
    filters = {"Average_pLDDT": {"threshold": 0.8, "higher": True}}
    assert filter_utils.check_filters([0.9], ["Average_pLDDT"], filters) is True
    assert filter_utils.check_filters([0.5], ["Average_pLDDT"], filters) == ["Average_pLDDT"]


def test_non_numeric_value_is_treated_as_a_failure():
    filters = {"Average_pLDDT": {"threshold": 0.8, "higher": True}}
    assert filter_utils.unmet_filters({"Average_pLDDT": "n/a"}, filters) == ["Average_pLDDT"]


# --- the pre-PyRosetta gate --------------------------------------------


def _gate_pipeline(tmp_path, filters, model_indices=(3, 4)):
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    pipeline = object.__new__(BinderDesignPipeline)
    pipeline.filters = filters
    pipeline._model_indices = lambda: list(model_indices)
    return pipeline


def test_base_gate_only_uses_the_predicted_models_thresholds(tmp_path):
    """The gate mirrors predict_binder_complex: per-model thresholds only.

    Validation runs models 4 and 5, so a threshold on model 1 must be ignored
    even though the value is present and would fail.
    """
    pipeline = _gate_pipeline(
        tmp_path,
        {
            "1_pLDDT": {"threshold": 0.75, "higher": True},
            "4_pLDDT": {"threshold": 0.60, "higher": True},
        },
    )
    unmet = pipeline._base_af2_filters_pass({"1_pLDDT": 0.10, "4_pLDDT": 0.70})
    assert unmet == []


def test_base_gate_ignores_average_thresholds(tmp_path):
    """Regression: including Average_* rejected 45% of designs before scoring.

    Production filter sets leave 4_*/5_* at null while setting Average_*, which
    makes DdCraft's gate a deliberate no-op. Honouring the averages here killed
    designs that native would have scored and accepted.
    """
    pipeline = _gate_pipeline(
        tmp_path,
        {
            "Average_pLDDT": {"threshold": 0.75, "higher": True},
            "Average_i_pTM": {"threshold": 0.45, "higher": True},
            "4_pLDDT": {"threshold": None, "higher": True},
            "5_pLDDT": {"threshold": None, "higher": True},
        },
    )
    unmet = pipeline._base_af2_filters_pass(
        {"Average_pLDDT": 0.40, "Average_i_pTM": 0.20}
    )
    assert unmet == []


def test_base_gate_still_rejects_on_a_set_per_model_threshold(tmp_path):
    pipeline = _gate_pipeline(
        tmp_path, {"5_i_pTM": {"threshold": 0.45, "higher": True}}
    )
    assert pipeline._base_af2_filters_pass({"5_i_pTM": 0.10}) == ["5_i_pTM"]


def test_base_gate_never_references_metrics_computed_after_it(tmp_path):
    """Binder_* and PyRosetta metrics only exist once the gate has passed."""
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    labels = {
        f"{i}_{metric}"
        for i in range(1, 6)
        for metric in BinderDesignPipeline.BASE_METRICS
    }
    for late in (
        "4_Binder_pLDDT",
        "4_Binder_pTM",
        "4_Binder_pAE",
        "4_Binder_RMSD",
        "4_dG",
        "4_ShapeComplementarity",
    ):
        assert late not in labels, f"{late} is not available at gate time"


# --- MPNN input handling ------------------------------------------------


def _write_two_chain_pdb(path: Path) -> None:
    lines = []
    serial = 1
    for chain, length, start in (("A", 3, 1), ("B", 4, 1)):
        for i in range(length):
            for atom, element in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} ALA {chain}{start + i:4d}    "
                    f"{serial:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00          {element:>2s}"
                )
                serial += 1
    lines.append("END")
    path.write_text("\n".join(lines))


def test_inputs_from_pdb_reads_chains_in_the_requested_order(tmp_path):
    from mosaic.proteinmpnn.inputs import inputs_from_pdb

    pdb = tmp_path / "complex.pdb"
    _write_two_chain_pdb(pdb)

    inputs = inputs_from_pdb(pdb, "A,B")
    assert inputs.chains == ["A", "B"]
    assert inputs.lengths == [3, 4]
    assert inputs.X.shape == (7, 4, 3)
    assert inputs.sequence == "AAAAAAA"
    assert inputs.chain_slice("B") == slice(3, 7)

    # per-chain residue indexing, offset by chain, as ProteinMPNN expects
    assert list(inputs.residue_idx) == [0, 1, 2, 100, 101, 102, 103]


def test_parse_fixed_positions_handles_chains_residues_and_ranges(tmp_path):
    from mosaic.proteinmpnn.inputs import inputs_from_pdb, parse_fixed_positions

    pdb = tmp_path / "complex.pdb"
    _write_two_chain_pdb(pdb)
    inputs = inputs_from_pdb(pdb, "A,B")

    assert list(parse_fixed_positions("A", inputs)) == [0, 1, 2]
    assert list(parse_fixed_positions("B2", inputs)) == [4]
    assert list(parse_fixed_positions("B2-4", inputs)) == [4, 5, 6]
    assert list(parse_fixed_positions("A,B1", inputs)) == [0, 1, 2, 3]

    with pytest.raises(ValueError):
        parse_fixed_positions("Z1", inputs)


def test_make_bias_omits_requested_amino_acids_and_the_unknown_token():
    from mosaic.proteinmpnn.sampling import MPNN_ALPHABET, make_bias

    bias = make_bias(5, omit_AAs="C,M")
    for aa in ("C", "M", "X"):
        assert np.all(bias[:, MPNN_ALPHABET.index(aa)] < -1e5)
    assert bias[0, MPNN_ALPHABET.index("A")] == 0.0

    # both spellings DdCraft configs use
    assert np.array_equal(make_bias(3, omit_AAs="CM"), make_bias(3, omit_AAs="C,M"))


def test_sequence_round_trips_through_the_mpnn_alphabet():
    from mosaic.proteinmpnn.sampling import decode_sequence, encode_sequence

    sequence = "ACDEFGHIKLMNPQRSTVWY"
    assert decode_sequence(encode_sequence(sequence)) == sequence
    # unknown residues collapse to X rather than raising
    assert decode_sequence(encode_sequence("AZB")) == "AXX"


# --- structure output conventions ---------------------------------------


def test_pipeline_config_defaults_blank_binder_chain_to_b():
    from mosaic.binder_design.pipeline import PipelineConfig

    config = PipelineConfig(
        target_settings={"chains": "A", "binder_chain": ""},
        advanced_settings={},
        filters={},
    )

    assert config.binder_chain == "B"


def test_normalize_complex_pdb_puts_target_chains_first(tmp_path):
    from Bio.PDB import PDBParser

    from mosaic.binder_design.io import normalize_complex_pdb

    # a prediction as Mosaic emits it: binder first, then the target
    raw = tmp_path / "raw.pdb"
    lines = []
    serial = 1
    for chain, length in (("B", 4), ("A", 3)):
        for i in range(length):
            for atom, element in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} ALA {chain}{i + 1:4d}    "
                    f"{serial:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00          {element:>2s}"
                )
                serial += 1
    lines.append("END")
    raw.write_text("\n".join(lines))

    starting = tmp_path / "start.pdb"
    _write_two_chain_pdb(starting)

    out = normalize_complex_pdb(
        raw, tmp_path / "out.pdb", starting, ["A"], "B", binder_length=4
    )

    model = PDBParser(QUIET=True).get_structure("x", str(out))[0]
    assert [chain.id for chain in model] == ["A", "B"]
    assert len(list(model["A"])) == 3
    assert len(list(model["B"])) == 4


def test_normalize_binder_pdb_renames_the_single_chain(tmp_path):
    from Bio.PDB import PDBParser

    from mosaic.binder_design.io import ChainLayoutError, normalize_binder_pdb

    raw = tmp_path / "binder.pdb"
    lines = []
    serial = 1
    for i in range(4):
        for atom, element in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
            lines.append(
                f"ATOM  {serial:5d}  {atom:<3s} ALA B{i + 7:4d}    "
                f"{serial:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00          {element:>2s}"
            )
            serial += 1
    lines.append("END")
    raw.write_text("\n".join(lines))

    out = normalize_binder_pdb(raw, tmp_path / "out.pdb", binder_length=4)
    model = PDBParser(QUIET=True).get_structure("x", str(out))[0]
    assert [chain.id for chain in model] == ["A"]
    assert [residue.id[1] for residue in model["A"]] == [1, 2, 3, 4]

    with pytest.raises(ChainLayoutError):
        normalize_binder_pdb(raw, tmp_path / "bad.pdb", binder_length=99)


# --- configuration ------------------------------------------------------


def test_model_slots_are_independent_and_accept_all_registered_names():
    from mosaic.binder_design.components import AVAILABLE_MODELS, model_selection

    assert AVAILABLE_MODELS == ("af2", "boltz2", "protenix-v2", "esmfold2")
    settings = {
        "hallucination_model": "boltz-2",
        "folding_model": {"name": "protenix_v2", "sampling_steps": 12},
        "folding_model_options": {"num_samples": 3},
    }
    hallucination = model_selection(settings, "hallucination")
    folding = model_selection(settings, "folding")

    assert hallucination.name == "boltz2"
    assert folding.name == "protenix-v2"
    assert folding.sampling_steps == 12
    assert folding.members == (0, 1, 2)


def test_model_slot_defaults_to_af2_and_rejects_unknown_names():
    from mosaic.binder_design.components import model_selection

    assert model_selection({}, "hallucination").name == "af2"
    assert model_selection({}, "folding").name == "af2"
    with pytest.raises(ValueError, match="Unknown structure model"):
        model_selection({"folding_model": "not-a-model"}, "folding")


def test_af2_model_numbers_drive_stage_member_indices():
    from mosaic.binder_design.components import model_selection
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    pipeline = object.__new__(BinderDesignPipeline)
    pipeline.advanced_settings = {}
    pipeline.config = SimpleNamespace(validation_models=lambda: [3, 4])
    pipeline.hallucination_selection = model_selection(
        {"hallucination_model_options": {"model_numbers": [1, 3]}},
        "hallucination",
    )
    pipeline.folding_selection = model_selection(
        {"folding_model_options": {"model_numbers": [2, 5]}}, "folding"
    )

    assert pipeline.design_models() == [0, 2]
    assert pipeline._model_indices() == [1, 4]


def test_model_members_use_bounded_zero_based_csv_slots():
    from mosaic.binder_design.components import model_selection

    selection = model_selection(
        {"folding_model": "esmfold2", "folding_model_options": {"members": [0, 4]}},
        "folding",
    )
    assert selection.members == (0, 4)
    with pytest.raises(ValueError, match="zero-based"):
        _ = model_selection(
            {"folding_model_options": {"members": [5]}}, "folding"
        ).members


def test_active_filter_metrics_select_standalone_filter_groups(tmp_path):
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    pipeline = object.__new__(BinderDesignPipeline)
    pipeline.filters = {
        "Average_pLDDT": {"threshold": 0.8, "higher": True},
        "4_dG": {"threshold": -10, "higher": False},
        "Average_Binder_pTM": {"threshold": None, "higher": True},
        "Average_InterfaceAAs": {
            "C": {"threshold": 0, "higher": False},
            "W": {"threshold": None, "higher": False},
        },
    }

    assert pipeline._active_metrics() == {"pLDDT", "dG", "InterfaceAAs"}


def test_geometry_filter_group_does_not_load_pyrosetta(tmp_path):
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    pdb = tmp_path / "complex.pdb"
    _write_two_chain_pdb(pdb)
    pipeline = _bare_pipeline(tmp_path / "results", design_labels())
    pipeline.config = SimpleNamespace(
        binder_chain="B", first_target_chain="A", target_chains="A"
    )
    pipeline.advanced_settings = {}
    pipeline._load_rosetta = lambda: pytest.fail(
        "unrelaxed geometry must not load PyRosetta"
    )

    values = BinderDesignPipeline._score_structure(
        pipeline,
        str(pdb),
        "geometry_only",
        1,
        str(pdb),
        requested_metrics={"Unrelaxed_Clashes"},
    )
    assert set(values) == {"Unrelaxed_Clashes"}


def test_pyrosetta_filter_group_is_loaded_lazily(tmp_path):
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    pdb = tmp_path / "complex.pdb"
    _write_two_chain_pdb(pdb)
    pipeline = _bare_pipeline(tmp_path / "results", design_labels())
    pipeline.config = SimpleNamespace(
        binder_chain="B", first_target_chain="A", target_chains="A"
    )
    pipeline.advanced_settings = {}
    calls = []

    class FakeRosetta:
        @staticmethod
        def pr_relax(source, destination, binder_chain):
            calls.append(("relax", binder_chain))
            Path(destination).write_bytes(Path(source).read_bytes())

        @staticmethod
        def score_interface(path, binder_chain):
            calls.append(("score", binder_chain))
            return {"interface_dG": -12.5}, {}, ""

    pipeline._load_rosetta = lambda: FakeRosetta
    values = BinderDesignPipeline._score_structure(
        pipeline,
        str(pdb),
        "pyrosetta_only",
        1,
        str(pdb),
        requested_metrics={"dG"},
    )
    assert values == {"dG": -12.5}
    assert calls == [("relax", "B"), ("score", "B")]


def test_config_expands_paths_and_fills_defaults(tmp_path):
    from mosaic.binder_design.pipeline import PipelineConfig

    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target.json"
    advanced = tmp_path / "advanced.json"
    filters = tmp_path / "filters.json"
    target.write_text(
        json.dumps(
            {
                "design_path": "${DDCRAFT_DIR}/results/run/",
                "starting_pdb": "${DDCRAFT_DIR}/example/t.pdb",
                "chains": "A",
                "binder_chain": "B",
            }
        )
    )
    advanced.write_text(
        json.dumps({"af_params_dir": "", "dssp_path": "", "dalphaball_path": "", "omit_AAs": " C,M "})
    )
    filters.write_text("{}")

    config = PipelineConfig.from_files(target, advanced, filters, root=root)

    assert config.target_settings["design_path"] == f"{root}/results/run/"
    assert config.target_settings["starting_pdb"] == f"{root}/example/t.pdb"
    assert config.advanced_settings["af_params_dir"] == str(root)
    assert config.advanced_settings["dssp_path"].endswith("functions/dssp")
    assert config.advanced_settings["dalphaball_path"].endswith("functions/DAlphaBall.gcc")
    assert config.advanced_settings["omit_AAs"] == "C,M"


def test_validation_models_are_disjoint_from_design_models(tmp_path):
    """DdCraft never validates with a model that shaped the backbone."""
    from mosaic.binder_design.pipeline import PipelineConfig

    config = PipelineConfig(
        target_settings={"chains": "A"},
        advanced_settings={"use_multimer_design": True},
        filters={},
    )
    assert config.validation_models() == [3, 4]

    config.advanced_settings["use_multimer_design"] = False
    assert config.validation_models() == [0, 1, 2, 3, 4]

    config.advanced_settings["predict_models"] = [1]
    assert config.validation_models() == [1]


def _bare_pipeline(tmp_path, design_labels_list):
    """A pipeline shell without PyRosetta or model loading.

    ``file_design`` and ``write_final_csv`` only touch the filesystem, so the
    expensive constructor is skipped rather than mocked out piece by piece.
    """
    from mosaic.binder_design.pipeline import BinderDesignPipeline, generate_directories

    pipeline = object.__new__(BinderDesignPipeline)
    pipeline.design_paths = generate_directories(tmp_path)
    pipeline.design_labels = design_labels_list
    pipeline.final_labels = ["Rank"] + design_labels_list
    return pipeline


def _write_relaxed(pipeline, name, model, bfactor):
    path = Path(pipeline.design_paths["MPNN/Relaxed"]) / f"{name}_model{model}.pdb"
    path.write_text(
        f"ATOM      1  CA  ALA B   1      "
        f"0.000   0.000   0.000  1.00{bfactor:6.2f}           C\nEND\n"
    )
    return path


def test_file_design_keeps_the_highest_confidence_model(tmp_path):
    from mosaic.binder_design.pipeline import DesignRecord

    pipeline = _bare_pipeline(tmp_path, design_labels())
    _write_relaxed(pipeline, "d1", 1, 50.0)
    _write_relaxed(pipeline, "d1", 2, 90.0)
    _write_relaxed(pipeline, "d1", 3, 70.0)

    record = DesignRecord(name="d1", sequence="A")
    record.accepted = True
    destination = pipeline.file_design(record)

    assert destination == Path(pipeline.design_paths["Accepted"]) / "d1.pdb"
    assert "90.00" in destination.read_text()


def test_rejected_designs_go_to_the_rejected_folder(tmp_path):
    from mosaic.binder_design.pipeline import DesignRecord

    pipeline = _bare_pipeline(tmp_path, design_labels())
    _write_relaxed(pipeline, "d2", 1, 42.0)

    record = DesignRecord(name="d2", sequence="A")
    record.accepted = False
    destination = pipeline.file_design(record)

    assert destination.parent.name == "Rejected"
    assert not (Path(pipeline.design_paths["Accepted"]) / "d2.pdb").exists()


def test_file_design_skips_an_unreadable_fold_without_crashing(tmp_path):
    from mosaic.binder_design.pipeline import DesignRecord

    pipeline = _bare_pipeline(tmp_path, design_labels())
    malformed = Path(pipeline.design_paths["MPNN"]) / "bad_model1.pdb"
    malformed.write_text(
        "ATOM      1  CA  ALA B   1    -1017.150  -4.005-2090.420  1.00 0.00 C\n"
    )
    record = DesignRecord(name="bad", sequence="A")

    assert pipeline.file_design(record) is None


def test_final_csv_ranks_accepted_designs_by_i_ptm(tmp_path):
    import csv

    labels = design_labels()
    pipeline = _bare_pipeline(tmp_path, labels)
    accepted = Path(pipeline.design_paths["Accepted"])
    for name in ("low", "high"):
        (accepted / f"{name}.pdb").write_text("END\n")

    mpnn_csv = tmp_path / "mpnn_design_stats.csv"
    with mpnn_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=labels)
        writer.writeheader()
        for name, iptm in (("low", 0.4), ("high", 0.9), ("unfiled", 0.99)):
            writer.writerow({"Design": name, "Average_i_pTM": iptm})

    final_csv = tmp_path / "final_design_stats.csv"
    assert pipeline.write_final_csv(mpnn_csv, final_csv) == 2

    rows = list(csv.DictReader(final_csv.open()))
    assert [row["Design"] for row in rows] == ["high", "low"]
    assert [row["Rank"] for row in rows] == ["1", "2"]
    ranked = Path(pipeline.design_paths["Accepted/Ranked"])
    assert {p.name for p in ranked.glob("*.pdb")} == {"1_high.pdb", "2_low.pdb"}


def test_final_csv_renumbers_when_rerun(tmp_path):
    import csv

    labels = design_labels()
    pipeline = _bare_pipeline(tmp_path, labels)
    accepted = Path(pipeline.design_paths["Accepted"])
    (accepted / "a.pdb").write_text("END\n")

    mpnn_csv = tmp_path / "mpnn.csv"
    final_csv = tmp_path / "final.csv"
    with mpnn_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=labels)
        writer.writeheader()
        writer.writerow({"Design": "a", "Average_i_pTM": 0.5})
    pipeline.write_final_csv(mpnn_csv, final_csv)

    # A better design lands later; it must take rank 1 and displace "a".
    (accepted / "b.pdb").write_text("END\n")
    with mpnn_csv.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=labels).writerow(
            {"Design": "b", "Average_i_pTM": 0.8}
        )
    assert pipeline.write_final_csv(mpnn_csv, final_csv) == 2

    ranked = Path(pipeline.design_paths["Accepted/Ranked"])
    assert {p.name for p in ranked.glob("*.pdb")} == {"1_b.pdb", "2_a.pdb"}


def _designed(sequence, score):
    from mosaic.binder_design.mpnn import DesignedSequence

    return DesignedSequence(
        sequence=sequence, score=score, seqid=0.0, full_sequence=sequence
    )


def test_ranking_keeps_every_candidate_not_just_max_mpnn_sequences(tmp_path):
    """``max_mpnn_sequences`` caps *accepted* designs, not candidates.

    Truncating the candidate list here meant only two of twenty sequences were
    ever validated, so a trajectory whose best sequences failed the filters was
    abandoned instead of falling through to the next candidate.
    """
    pipeline = _bare_pipeline(tmp_path, design_labels())
    pipeline.advanced_settings = {"max_mpnn_sequences": 2}
    pipeline._seen_sequences = set()

    candidates = [_designed(f"AAA{i}", score=1.0 - i * 0.01) for i in range(20)]
    ranked = pipeline._rank_sequences(candidates)

    assert len(ranked) == 20
    assert [s.score for s in ranked] == sorted(s.score for s in candidates)


def test_ranking_deduplicates_and_skips_sequences_already_evaluated(tmp_path):
    pipeline = _bare_pipeline(tmp_path, design_labels())
    pipeline.advanced_settings = {}
    pipeline._seen_sequences = {"SEEN"}

    ranked = pipeline._rank_sequences(
        [
            _designed("SEEN", 0.1),
            _designed("DUP", 0.9),
            _designed("DUP", 0.5),
            _designed("NEW", 0.7),
        ]
    )

    assert [s.sequence for s in ranked] == ["DUP", "NEW"]
    assert ranked[0].score == 0.5


def test_force_reject_aa_drops_sequences_containing_omitted_residues(tmp_path):
    pipeline = _bare_pipeline(tmp_path, design_labels())
    pipeline.advanced_settings = {"force_reject_AA": True, "omit_AAs": "C,M"}
    pipeline._seen_sequences = set()

    ranked = pipeline._rank_sequences(
        [_designed("AAC", 0.1), _designed("AAM", 0.2), _designed("AAK", 0.3)]
    )

    assert [s.sequence for s in ranked] == ["AAK"]


def test_load_seen_sequences_reads_the_existing_csv(tmp_path):
    import csv

    labels = design_labels()
    pipeline = _bare_pipeline(tmp_path, labels)
    pipeline.advanced_settings = {}
    pipeline._seen_sequences = set()

    mpnn_csv = tmp_path / "mpnn.csv"
    with mpnn_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=labels)
        writer.writeheader()
        writer.writerow({"Design": "d1", "Sequence": "ABC"})
        writer.writerow({"Design": "d2", "Sequence": ""})

    assert pipeline.load_seen_sequences(mpnn_csv) == 1
    assert pipeline._rank_sequences([_designed("ABC", 0.1)]) == []


def test_a_failing_design_does_not_abandon_the_remaining_candidates(
    tmp_path, monkeypatch
):
    """One bad structure costs one design, not the rest of the run.

    PyRosetta occasionally fails to read a structure it has just written. That
    killed a nine-backbone run on its third design, so evaluation is wrapped
    per candidate exactly as DdCraft wraps it.
    """
    import jax

    from mosaic.binder_design import pipeline as pipeline_module
    from mosaic.binder_design.pipeline import BinderDesignPipeline, DesignRecord

    trajectory = tmp_path / "traj.pdb"
    _write_two_chain_pdb(trajectory)

    pipeline = _bare_pipeline(tmp_path, design_labels())
    pipeline.advanced_settings = {"max_mpnn_sequences": 99}
    pipeline.target_settings = {"starting_pdb": str(trajectory)}
    pipeline._seen_sequences = set()
    pipeline.config = SimpleNamespace(binder_chain="B", target_chains="A")
    pipeline._structure_model = object()
    pipeline._monomer_model = object()
    pipeline._recycling_steps = lambda: 1
    pipeline._ipsae_cutoff = lambda: None

    monkeypatch.setattr(pipeline_module, "ComplexPredictor", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_module, "MonomerPredictor", lambda *a, **k: None)

    candidates = [_designed(f"SEQ{i}", score=0.1 * i) for i in range(4)]
    evaluated = []

    def fake_evaluate(design, *, design_name, **kwargs):
        evaluated.append(design_name)
        if design.sequence == "SEQ1":
            raise RuntimeError("Cannot open file")
        return DesignRecord(name=design_name, sequence=design.sequence)

    pipeline.evaluate_design = fake_evaluate
    pipeline.design_sequences = lambda pdb, key: (candidates, "")

    records = BinderDesignPipeline.run_trajectory(
        pipeline, trajectory, key=jax.random.key(0)
    )

    assert len(evaluated) == 4, "every candidate should still be attempted"
    assert [r.name for r in records] == ["traj_mpnn1", "traj_mpnn3", "traj_mpnn4"]


def test_af2_member_kwargs_map_global_model_numbers_to_stack_positions():
    """A model holding only a subset still reports global model numbers.

    Monomer models 3-5 are trained without templates and cannot be stacked with
    1-2, so the binder-alone predictor loads just 4 and 5. Callers keep speaking
    in global indices so the CSV labels stay 4_* and 5_*.
    """
    model = object.__new__(AlphaFold2)
    object.__setattr__(model, "model_numbers", (4, 5))

    assert model.ensemble_members() == (3, 4)
    assert model.member_kwargs(3) == {"model_idx": 0}
    assert model.member_kwargs(4) == {"model_idx": 1}


def test_af2_member_kwargs_reject_a_model_that_was_not_loaded():
    model = object.__new__(AlphaFold2)
    object.__setattr__(model, "model_numbers", (4, 5))

    with pytest.raises(ValueError, match="model 1 is not loaded"):
        model.member_kwargs(0)


def test_a_model_without_an_ensemble_has_one_member():
    """Models whose members differ only by RNG key need no override."""

    class SingleNetwork(StructurePredictionModel):
        def target_only_features(self, chains):
            raise NotImplementedError

        def binder_features(self, binder_length, chains):
            raise NotImplementedError

        def predict(self, **kwargs):
            raise NotImplementedError

        def model_output(self, **kwargs):
            raise NotImplementedError

        def build_loss(self, **kwargs):
            raise NotImplementedError

    model = object.__new__(SingleNetwork)
    assert model.ensemble_members() == (0,)
    assert model.member_kwargs(0) == {}


def test_complex_predictor_takes_its_members_from_the_model():
    """The default member list is the model's, not an AF2-shaped assumption."""
    predictor = object.__new__(ComplexPredictor)
    model = SimpleNamespace(ensemble_members=lambda: (0, 1, 2))
    predictor.model_indices = list(model.ensemble_members())

    assert predictor.model_indices == [0, 1, 2]


def test_sequence_dependent_backend_is_refeaturized_for_each_complex(tmp_path):
    pdb = tmp_path / "complex.pdb"
    _write_two_chain_pdb(pdb)

    class SequenceDependentModel:
        def __init__(self):
            self.calls = []

        def supports_template_chains(self):
            return False

        def prediction_features_depend_on_sequence(self):
            return True

        def ensemble_members(self):
            return (0,)

        def target_only_features(self, chains):
            self.calls.append(chains)
            return tuple(chain.sequence for chain in chains), "writer"

        def binder_features(self, **kwargs):
            raise AssertionError("native finished-sequence scoring must not use a design pack")

    model = SequenceDependentModel()
    predictor = ComplexPredictor(
        model,
        target_pdb=pdb,
        target_chains="A",
        binder_length=4,
    )
    pssm, features, writer = predictor._prediction_inputs("CCCC")

    assert pssm is None
    assert features == ("CCCC", "AAA")
    assert writer == "writer"
    assert model.calls[0][0].template_chain is None
    assert model.calls[0][1].template_chain is None


def test_monomer_predictor_defaults_to_backend_members():
    class ReusableModel:
        def prediction_features_depend_on_sequence(self):
            return False

        def ensemble_members(self):
            return (0, 2)

        def target_only_features(self, chains):
            return "features", "writer"

    predictor = MonomerPredictor(ReusableModel(), binder_length=4)
    assert predictor.model_indices == [0, 2]


def test_evaluate_pdb_bypasses_hallucination_mpnn_and_pyrosetta(
    tmp_path, monkeypatch
):
    import jax

    from mosaic.binder_design import pipeline as pipeline_module
    from mosaic.binder_design.pipeline import DesignRecord

    pdb = tmp_path / "input.pdb"
    _write_two_chain_pdb(pdb)
    pipeline = _bare_pipeline(tmp_path / "results", design_labels())
    pipeline.config = SimpleNamespace(binder_chain="B", target_chains="A")
    pipeline.target_settings = {"starting_pdb": str(pdb)}
    pipeline.advanced_settings = {}
    pipeline.filters = {}
    pipeline.folding_selection = SimpleNamespace(name="boltz2", sampling_steps=5)
    pipeline._structure_model = object()
    pipeline._model_indices = lambda: [0]
    pipeline._recycling_steps = lambda: 1
    pipeline._ipsae_cutoff = lambda: None
    pipeline.generate_trajectory = lambda **kwargs: pytest.fail(
        "evaluation must not hallucinate"
    )
    pipeline.design_sequences = lambda *args, **kwargs: pytest.fail(
        "evaluation must not run ProteinMPNN"
    )
    pipeline._load_rosetta = lambda: pytest.fail(
        "confidence-only evaluation must not load PyRosetta"
    )

    captured = {}

    class FakeComplexPredictor:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(pipeline_module, "ComplexPredictor", FakeComplexPredictor)

    def fake_evaluate(design, **kwargs):
        captured["sequence"] = design.sequence
        captured.update(kwargs)
        return DesignRecord(name=kwargs["design_name"], sequence=design.sequence)

    pipeline.evaluate_design = fake_evaluate
    record = pipeline.evaluate_pdb(pdb, key=jax.random.key(0))

    assert captured["sequence"] == "AAAA"
    assert captured["configured_only"] is True
    assert captured["reference_pdb"] == str(pdb)
    assert captured["sampling_steps"] == 5
    assert captured["monomer_predictor"] is None
    assert record.values["Protocol"] == "evaluation:boltz2"


def test_cli_exposes_evaluation_inputs_and_chain_overrides():
    from mosaic.binder_design.__main__ import build_parser

    args = build_parser().parse_args(
        [
            "--settings",
            "target.json",
            "--advanced",
            "advanced.json",
            "--filters",
            "filters.json",
            "--mode",
            "evaluate",
            "--input-pdb",
            "one.pdb",
            "--input-pdb",
            "two.pdb",
            "--binder-chain",
            "H",
            "--target-chains",
            "A,L",
            "--output-dir",
            "separate-results",
            "--evaluation-metric-group",
            "pyrosetta",
        ]
    )
    assert args.mode == "evaluate"
    assert args.input_pdb == [Path("one.pdb"), Path("two.pdb")]
    assert args.binder_chain == "H"
    assert args.target_chains == "A,L"
    assert args.output_dir == Path("separate-results")
    assert args.evaluation_metric_group == ["pyrosetta"]


def test_evaluation_metric_groups_request_work_without_becoming_filters(tmp_path):
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    pipeline = object.__new__(BinderDesignPipeline)
    pipeline.advanced_settings = {
        "evaluation_metric_groups": ["monomer", "pyrosetta"]
    }
    pipeline.filters = {
        "Average_pLDDT": {"threshold": None, "higher": True}
    }

    active = pipeline._active_metrics()

    assert {"Binder_pLDDT", "Binder_pTM", "Binder_pAE"} <= active
    assert BinderDesignPipeline.PYROSETTA_METRICS <= active
    assert "Binder_RMSD" in active
    assert filter_utils.unmet_filters({}, pipeline.filters) == []


def test_unknown_evaluation_metric_group_is_rejected():
    from mosaic.binder_design.pipeline import BinderDesignPipeline

    pipeline = object.__new__(BinderDesignPipeline)
    pipeline.advanced_settings = {"evaluation_metric_groups": ["mystery"]}
    pipeline.filters = {}

    with pytest.raises(ValueError, match="Unknown evaluation metric group"):
        pipeline._active_metrics()


def test_evaluation_matrix_isolates_backends_and_forwards_optional_metrics(
    tmp_path, monkeypatch
):
    from mosaic.binder_design import matrix

    settings = tmp_path / "settings.json"
    filters = tmp_path / "filters.json"
    input_pdb = tmp_path / "complex.pdb"
    advanced_dir = tmp_path / "advanced"
    output_root = tmp_path / "matrix"
    settings.write_text("{}\n")
    filters.write_text("{}\n")
    input_pdb.write_text("MODEL\nEND\n")
    advanced_dir.mkdir()
    for filename in matrix.BACKEND_PRESETS.values():
        (advanced_dir / filename).write_text("{}\n")

    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs["env"]))
        output = Path(command[command.index("--output-dir") + 1])
        backend = output.name
        with (output / "evaluation_stats.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Design", "Average_pLDDT"])
            writer.writeheader()
            writer.writerow({"Design": f"{backend}_design", "Average_pLDDT": "0.9"})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.sys, "executable", "/runtime/python")
    result = matrix.main(
        [
            "--settings",
            str(settings),
            "--filters",
            str(filters),
            "--advanced-dir",
            str(advanced_dir),
            "--output-root",
            str(output_root),
            "--backend",
            "boltz2",
            "--backend",
            "protenix-v2",
            "--input-pdb",
            str(input_pdb),
            "--metric-group",
            "pyrosetta",
            "--device",
            "boltz2=0",
            "--device",
            "protenix-v2=2",
        ],
        runner=fake_runner,
    )

    assert result == 0
    assert len(calls) == 2
    assert calls[0][0][:3] == [
        "/runtime/python",
        "-m",
        "mosaic.binder_design",
    ]
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[1][1]["CUDA_VISIBLE_DEVICES"] == "2"
    assert all("--evaluation-metric-group" in command for command, _ in calls)
    with (output_root / "evaluation_matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Backend"] for row in rows] == ["boltz2", "protenix-v2"]
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["metric_groups"] == ["pyrosetta"]
    assert manifest["backends"]["boltz2"]["status"] == "completed"


def test_shipped_evaluation_presets_cover_every_backend_without_extra_filters():
    from mosaic.binder_design.components import model_selection
    from mosaic.binder_design.pipeline import BinderDesignPipeline, PipelineConfig

    repository = Path(__file__).parents[2]
    config_root = repository / "ddcraft" / "configs"
    if not config_root.is_dir():
        pytest.skip("evaluation presets live in kite-binder-design")
    target = config_root / "settings_target" / "mosaic_evaluation_example.json"
    filters = config_root / "settings_filters" / "confidence_only.json"
    presets = {
        "af2.json": ("af2", (3,), None),
        "boltz2.json": ("boltz2", (0,), 5),
        "protenix_v2.json": ("protenix-v2", (0,), 5),
        "esmfold2_fast.json": ("esmfold2", (0,), 5),
    }

    filter_settings = json.loads(filters.read_text())
    assert set(filter_settings) == {
        f"Average_{metric}" for metric in BinderDesignPipeline.BASE_METRICS
    }
    assert all(
        conditions["threshold"] is None
        for conditions in filter_settings.values()
    )

    for filename, expected in presets.items():
        config = PipelineConfig.from_files(
            target,
            config_root / "settings_advanced" / "mosaic_evaluation" / filename,
            filters,
            root=repository / "ddcraft",
        )
        selection = model_selection(config.advanced_settings, "folding")
        assert (selection.name, selection.members, selection.sampling_steps) == expected
        assert config.advanced_settings["num_recycles_validation"] == 0
        assert config.target_settings["design_path"].startswith(
            str(repository / "ddcraft")
        )


def test_accepted_design_count_reads_the_accepted_directory(tmp_path):
    """Generation stops on accepted designs on disk, so a resume stops at the same total."""
    pipeline = _bare_pipeline(tmp_path, design_labels())
    accepted = Path(pipeline.design_paths["Accepted"])
    accepted.mkdir(parents=True, exist_ok=True)

    assert pipeline.accepted_design_count() == 0

    (accepted / "a_model4.pdb").write_text("")
    (accepted / "b_model5.pdb").write_text("")
    (accepted / "notes.txt").write_text("ignored")

    assert pipeline.accepted_design_count() == 2
