"""Focused contract tests for the OpenDDE adapter."""

from pathlib import Path

import gemmi
import numpy as np
import pytest

from mosaic.models.opendde import (
    _binder_extents,
    _build_spec,
    _featurize,
    _target_template_features,
)
from mosaic.structure_prediction import PolymerType, TargetChain


def test_build_spec_for_msa_free_protein():
    spec = _build_spec([TargetChain("ACDE", use_msa=False)])

    assert spec == [
        {
            "name": "design",
            "modelSeeds": [0],
            "sequences": [{"proteinChain": {"sequence": "ACDE", "count": 1}}],
        }
    ]


@pytest.mark.parametrize("polymer_type", [PolymerType.RNA, PolymerType.DNA])
def test_build_spec_rejects_nonprotein_chains(polymer_type):
    chain = TargetChain("ACGU", polymer_type=polymer_type, use_msa=False)

    with pytest.raises(NotImplementedError, match="protein chains only"):
        _build_spec([chain])


def test_build_spec_accepts_target_templates():
    chain = TargetChain("ACDE", use_msa=False, template_chain=gemmi.Chain("A"))

    assert _build_spec([chain])[0]["sequences"] == [
        {"proteinChain": {"sequence": "ACDE", "count": 1}}
    ]


@pytest.mark.slow
@pytest.mark.opendde_smoke
def test_featurizes_1ubq_as_structural_template():
    structure = gemmi.read_structure(str(Path(__file__).parent / "data" / "1ubq.cif"))
    template = structure[0]["A"]
    sequence = gemmi.one_letter_code([res.name for res in template.get_polymer()])

    features, _writer = _featurize(
        [TargetChain(sequence, use_msa=False, template_chain=template)]
    )

    assert features.template is not None
    assert features.template.aatype.shape == (4, len(sequence))
    assert np.count_nonzero(np.asarray(features.template.pseudo_beta_mask[0])) > 0
    assert np.count_nonzero(np.asarray(features.template.backbone_frame_mask[0])) > 0


def test_target_template_ignores_waters_after_binder_offset():
    structure = gemmi.read_structure(str(Path(__file__).parent / "data" / "1ubq.cif"))
    template = structure[0]["A"]
    sequence = gemmi.one_letter_code([res.name for res in template.get_polymer()])

    features = _target_template_features(
        [
            TargetChain("WWWW", use_msa=False),
            TargetChain(sequence, use_msa=False, template_chain=template),
        ]
    )

    assert features is not None
    assert features["template_aatype"].shape == (4, 4 + len(sequence))
    assert np.all(features["template_aatype"][1:] == 0)
    assert not np.any(features["template_pseudo_beta_mask"][0, :4])
    assert np.any(features["template_pseudo_beta_mask"][0, 4:, 4:])


@pytest.mark.slow
@pytest.mark.opendde_smoke
def test_featurizes_single_protein_without_model_weights():
    sequence = "ACDEFGHIK"

    features, writer = _featurize([TargetChain(sequence, use_msa=False)])

    assert features.restype.shape[0] == len(sequence)
    assert features.asym_id.shape == (len(sequence),)
    assert np.all(np.asarray(features.asym_id) == 0)
    assert features.ref_pos.shape[-1] == 3
    assert writer is not None


@pytest.mark.slow
@pytest.mark.opendde_smoke
def test_binder_extent_matches_first_chain():
    binder_length = 5
    features, _writer = _featurize(
        [
            TargetChain("W" * binder_length, use_msa=False),
            TargetChain("ACDE", use_msa=False),
        ]
    )

    observed_length, atom_allocation = _binder_extents(features)

    assert observed_length == binder_length
    assert atom_allocation > binder_length


def test_model_build_loads_predictor_without_process_cache(monkeypatch, tmp_path):
    import jopendde.inference as inference
    import mosaic.models.opendde as adapter

    calls = []
    model = object()

    class SummaryParams:
        pae_bins = (0.0, 32.0, 64)
        plddt_bins = (0.0, 1.0, 50)

    class LoadedPredictor:
        summary_params = SummaryParams()

    class FakePredictor:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            calls.append((args, kwargs))
            predictor = LoadedPredictor()
            predictor.model = model
            return predictor

    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(inference, "Predictor", FakePredictor)
    monkeypatch.setattr(
        adapter,
        "_build_dense_atom_to_atom37",
        lambda: np.zeros((1, 1)),
    )
    monkeypatch.setattr(adapter, "_get_atom_templates", lambda: None)
    monkeypatch.setattr(adapter, "OpenDDEModel", lambda **kwargs: kwargs)

    first = adapter._build_model("opendde_v1", "opendde.pt")
    second = adapter._build_model("opendde_v1", "opendde.pt")

    expected_call = (
        ("opendde_v1",),
        {
            "checkpoint_file": "opendde.pt",
            "asset_cache_dir": tmp_path / "opendde",
        },
    )
    assert calls == [expected_call, expected_call]
    assert first["model"] is model
    assert second["model"] is model
