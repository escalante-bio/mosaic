from pathlib import Path

import numpy as np

from mosaic.models.af2 import make_af_features
from mosaic.models.af2_msa import (
    create_features_from_a3m,
    create_features_from_raw_msa,
    load_a3m_file,
    merge_unpaired_msas,
    raw_msa_from_sequence,
)
from mosaic.structure_prediction import TargetChain


def test_query_only_fasta_is_valid_a3m(tmp_path: Path):
    path = tmp_path / "query.fasta"
    path.write_text(">query\nACD\n")

    features = create_features_from_a3m(path, sequence="ACD")
    feature_dict = features.as_dict(multimer=True)

    assert feature_dict["msa_feat"].shape == (1, 3, 49)
    assert feature_dict["msa_mask"].shape == (1, 3)
    assert feature_dict["extra_msa"].shape == (1, 3)
    assert feature_dict["extra_msa"].dtype == np.int32
    assert not feature_dict["extra_msa_mask"].any()
    # Query-only msa_feat contains one residue and one profile one-hot per site.
    assert feature_dict["msa_feat"].sum() == 6


def test_a3m_insertions_deduplication_and_clustering(tmp_path: Path):
    path = tmp_path / "alignment.a3m"
    path.write_text(
        ">query\nACD\n"
        ">insertion\nAcCE\n"
        ">gap\nA-D\n"
        ">duplicate\nA-D\n"
    )

    raw_msa = load_a3m_file(path, sequence="ACD")
    assert raw_msa.tokens.shape == (3, 3)
    assert raw_msa.deletion_matrix.max() == 1

    features = create_features_from_raw_msa(
        raw_msa, max_msa_clusters=2, max_extra_msa=8, seed=0
    )
    assert features.msa_feat.shape == (2, 3, 49)
    assert features.extra_msa.shape == (1, 3)
    assert features.msa_feat.dtype == np.float32


def test_unpaired_chain_merge_gap_pads_homolog_rows(tmp_path: Path):
    path = tmp_path / "target.a3m"
    path.write_text(">query\nACD\n>homolog\nA-D\n")

    binder = raw_msa_from_sequence("GG")
    target = load_a3m_file(path, sequence="ACD")
    merged = merge_unpaired_msas([binder, target])

    assert merged.tokens.shape == (2, 5)
    np.testing.assert_array_equal(
        merged.tokens[0],
        np.concatenate([binder.tokens[0], target.tokens[0]]),
    )
    np.testing.assert_array_equal(merged.tokens[1, :2], np.asarray([21, 21]))


def test_make_af_features_integrates_target_msa_with_binder(tmp_path: Path):
    path = tmp_path / "target.a3m"
    path.write_text(
        ">query\nACD\n"
        ">homolog_one\nA-D\n"
        ">homolog_two\nAC-\n"
    )

    features = make_af_features(
        [
            TargetChain(sequence="GG", use_msa=False),
            TargetChain(sequence="ACD", use_msa=True, msa_path=str(path)),
        ],
        max_msa_clusters=1,
        max_extra_msa=8,
        msa_seed=0,
    )

    assert features["msa_feat"].shape == (1, 5, 49)
    assert features["msa_mask"].shape == (1, 5)
    assert features["extra_msa"].shape == (2, 5)
    assert features["extra_msa_mask"].shape == (2, 5)
    np.testing.assert_array_equal(features["asym_id"], [1, 1, 2, 2, 2])
    np.testing.assert_array_equal(
        np.argmax(features["msa_feat"][0, :, :23], axis=-1),
        features["aatype"],
    )
    np.testing.assert_array_equal(
        features["extra_msa"][:, :2],
        np.full((2, 2), 21),
    )
