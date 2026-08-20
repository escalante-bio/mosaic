"""Inference-only AlphaFold 2 MSA feature construction.

This module deliberately stops at the tensors consumed by the vendored AF2
model.  MSA search, target/template features, and differentiable binder
sequence updates belong elsewhere in the Mosaic wrapper.

The algorithm for the feature extraction has been implemented following the original paper (https://www.nature.com/articles/s41586-021-03819-2). 

Importantly, this does not compute the raw MSA but it expects one to compute the features from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mosaic.alphafold.common import residue_constants
from mosaic.alphafold.data import parsers


_GAP_TOKEN = residue_constants.restype_num + 1
_NUM_MSA_TOKENS = residue_constants.restype_num + 3


@dataclass(frozen=True)
class RawMSA:
    """An aligned MSA before clustering.

    Row zero is always the query. Tokens use AF2's internal residue ordering,
    with X=20 and gap=21.
    """

    tokens: np.ndarray  # int32 [N, L]
    deletion_matrix: np.ndarray  # float32 [N, L], raw insertion counts

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ValueError(f"MSA tokens must have rank 2, got {self.tokens.shape}")
        if self.deletion_matrix.shape != self.tokens.shape:
            raise ValueError(
                "MSA tokens and deletion matrix must have the same shape, got "
                f"{self.tokens.shape} and {self.deletion_matrix.shape}"
            )
        if self.tokens.shape[0] == 0:
            raise ValueError("MSA must contain at least the query row")


@dataclass(frozen=True)
class AF2MSAFeatures:
    """Post-clustering AF2 MSA features.

    Raw extra-MSA deletion counts are retained here because this vendored
    multimer implementation transforms them inside the model, whereas the
    monomer implementation expects the transformed values as input.
    """

    msa_feat: np.ndarray  # float32 [N_cluster, L, 49]
    msa_mask: np.ndarray  # float32 [N_cluster, L]
    extra_msa: np.ndarray  # int32 [N_extra, L]
    extra_deletion_matrix: np.ndarray  # float32 [N_extra, L]
    extra_msa_mask: np.ndarray  # float32 [N_extra, L]

    def as_dict(self, *, multimer: bool) -> dict[str, np.ndarray]:
        deletion_matrix = self.extra_deletion_matrix.astype(np.float32)
        deletion_value = (
            deletion_matrix
            if multimer
            else _deletion_value(deletion_matrix)
        )
        return {
            "msa_feat": self.msa_feat.astype(np.float32),
            "msa_mask": self.msa_mask.astype(np.float32),
            "msa_row_mask": np.any(self.msa_mask != 0, axis=-1).astype(np.float32),
            "extra_msa": self.extra_msa.astype(np.int32),
            "extra_deletion_value": deletion_value.astype(np.float32),
            "extra_has_deletion": np.clip(
                deletion_matrix, 0.0, 1.0
            ).astype(np.float32),
            "extra_msa_mask": self.extra_msa_mask.astype(np.float32),
            "extra_msa_row_mask": np.any(
                self.extra_msa_mask != 0, axis=-1
            ).astype(np.float32),
        }


def _deletion_value(deletion_matrix: np.ndarray) -> np.ndarray:
    return np.arctan(deletion_matrix / 3.0) * (2.0 / np.pi)


def _encode_aligned_sequence(sequence: str) -> np.ndarray:
    """Encode an aligned sequence using AF2's internal residue ordering."""

    try:
        hhblits_tokens = np.asarray(
            [residue_constants.HHBLITS_AA_TO_ID[residue] for residue in sequence],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError(
            f"Unsupported residue {error.args[0]!r} in aligned MSA sequence"
        ) from error

    return np.take(
        np.asarray(
            residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE,
            dtype=np.int32,
        ),
        hhblits_tokens,
    )


def raw_msa_from_sequence(sequence: str) -> RawMSA:
    """Construct a query-only MSA for a chain with ``use_msa=False``."""

    if not sequence:
        raise ValueError("Protein sequence must not be empty")
    tokens = _encode_aligned_sequence(sequence)[None]
    return RawMSA(
        tokens=tokens,
        deletion_matrix=np.zeros(tokens.shape, dtype=np.float32),
    )


def load_a3m_file(file_name: str | Path, *, sequence: str) -> RawMSA:
    """Parse and validate an A3M whose first row is ``sequence``."""

    path = Path(file_name).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MSA not found: {path}")

    parsed = parsers.parse_a3m(path.read_text())
    if not parsed.sequences:
        raise ValueError(f"MSA is empty: {path}")
    if parsed.sequences[0] != sequence:
        raise ValueError(
            f"A3M query in {path} does not match the target sequence: "
            f"{parsed.sequences[0]!r} != {sequence!r}"
        )

    unique_sequences: list[str] = []
    unique_deletions: list[Sequence[int]] = []
    seen_sequences: set[str] = set()
    for aligned_sequence, deletion_row in zip(
        parsed.sequences, parsed.deletion_matrix
    ):
        if len(aligned_sequence) != len(sequence):
            raise ValueError(
                f"Aligned row in {path} has length {len(aligned_sequence)}; "
                f"expected {len(sequence)}"
            )
        if len(deletion_row) != len(sequence):
            raise ValueError(
                f"Deletion row in {path} has length {len(deletion_row)}; "
                f"expected {len(sequence)}"
            )
        if aligned_sequence in seen_sequences:
            continue
        seen_sequences.add(aligned_sequence)
        unique_sequences.append(aligned_sequence)
        unique_deletions.append(deletion_row)

    return RawMSA(
        tokens=np.stack(
            [_encode_aligned_sequence(row) for row in unique_sequences]
        ).astype(np.int32),
        deletion_matrix=np.asarray(unique_deletions, dtype=np.float32),
    )


def merge_unpaired_msas(msas: Sequence[RawMSA]) -> RawMSA:
    """Merge per-chain MSAs into a full-complex, unpaired alignment.

    Row zero is the concatenated query. Each non-query homolog occupies only
    its chain's residue columns and is gap-padded across all other chains.
    """

    if not msas:
        raise ValueError("At least one chain MSA is required")

    lengths = [msa.tokens.shape[1] for msa in msas]
    total_length = sum(lengths)
    query = np.concatenate([msa.tokens[0] for msa in msas], axis=0)
    token_rows = [query]
    deletion_rows = [
        np.concatenate([msa.deletion_matrix[0] for msa in msas], axis=0).astype(
            np.float32
        )
    ]

    offset = 0
    for msa, length in zip(msas, lengths):
        for tokens, deletion_matrix in zip(
            msa.tokens[1:], msa.deletion_matrix[1:]
        ):
            merged_tokens = np.full(total_length, _GAP_TOKEN, dtype=np.int32)
            merged_deletions = np.zeros(total_length, dtype=np.float32)
            merged_tokens[offset : offset + length] = tokens
            merged_deletions[offset : offset + length] = deletion_matrix
            token_rows.append(merged_tokens)
            deletion_rows.append(merged_deletions)
        offset += length

    return RawMSA(
        tokens=np.stack(token_rows).astype(np.int32),
        deletion_matrix=np.stack(deletion_rows).astype(np.float32),
    )


def create_features_from_raw_msa(
    msa: RawMSA,
    *,
    max_msa_clusters: int = 512,
    max_extra_msa: int = 2048,
    seed: int = 0,
) -> AF2MSAFeatures:
    """Cluster a raw MSA and construct the tensors consumed by AF2."""

    if max_msa_clusters < 1:
        raise ValueError("max_msa_clusters must be at least 1")
    if max_extra_msa < 1:
        raise ValueError("max_extra_msa must be at least 1")

    num_sequences, num_residues = msa.tokens.shape
    rng = np.random.default_rng(seed)

    non_query_indices = rng.permutation(np.arange(1, num_sequences))
    index_order = np.concatenate(
        [np.asarray([0], dtype=np.int64), non_query_indices]
    )
    num_clusters = min(max_msa_clusters, num_sequences)
    cluster_indices = index_order[:num_clusters]
    extra_indices = index_order[num_clusters:]

    cluster_tokens = msa.tokens[cluster_indices]
    cluster_deletions = msa.deletion_matrix[cluster_indices]
    extra_tokens_all = msa.tokens[extra_indices]
    extra_deletions_all = msa.deletion_matrix[extra_indices]

    token_eye = np.eye(_NUM_MSA_TOKENS, dtype=np.float32)
    cluster_one_hot = token_eye[cluster_tokens]
    extra_one_hot_all = token_eye[extra_tokens_all]

    cluster_profile_sum = cluster_one_hot.copy()
    cluster_deletion_sum = cluster_deletions.copy()
    cluster_counts = np.ones(num_clusters, dtype=np.float32)

    if extra_tokens_all.shape[0]:
        agreement_weights = np.concatenate(
            [
                np.ones(residue_constants.restype_num + 1, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            ]
        )
        agreement = np.einsum(
            "xra,kra,a->xk",
            extra_one_hot_all,
            cluster_one_hot,
            agreement_weights,
            optimize=True,
        )
        assignment = np.argmax(agreement, axis=-1)
        np.add.at(cluster_profile_sum, assignment, extra_one_hot_all)
        np.add.at(cluster_deletion_sum, assignment, extra_deletions_all)
        np.add.at(cluster_counts, assignment, 1.0)

    cluster_profile = cluster_profile_sum / cluster_counts[:, None, None]
    cluster_deletion_mean = cluster_deletion_sum / cluster_counts[:, None]

    msa_feat = np.concatenate(
        [
            cluster_one_hot,
            np.clip(cluster_deletions, 0.0, 1.0)[..., None],
            _deletion_value(cluster_deletions)[..., None],
            cluster_profile,
            _deletion_value(cluster_deletion_mean)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)
    msa_mask = np.ones((num_clusters, num_residues), dtype=np.float32)

    if extra_tokens_all.shape[0]:
        crop_order = rng.permutation(extra_tokens_all.shape[0])[:max_extra_msa]
        extra_msa = extra_tokens_all[crop_order].astype(np.int32)
        extra_deletion_matrix = extra_deletions_all[crop_order].astype(np.float32)
        extra_msa_mask = np.ones(extra_msa.shape, dtype=np.float32)
    else:
        # The extra-MSA stack is not robust to a zero-sized sequence dimension.
        extra_msa = np.zeros((1, num_residues), dtype=np.int32)
        extra_deletion_matrix = np.zeros((1, num_residues), dtype=np.float32)
        extra_msa_mask = np.zeros((1, num_residues), dtype=np.float32)

    return AF2MSAFeatures(
        msa_feat=msa_feat,
        msa_mask=msa_mask,
        extra_msa=extra_msa,
        extra_deletion_matrix=extra_deletion_matrix,
        extra_msa_mask=extra_msa_mask,
    )


def create_features_from_a3m(
    file_name: str | Path,
    *,
    sequence: str,
    max_msa_clusters: int = 512,
    max_extra_msa: int = 2048,
    seed: int = 0,
) -> AF2MSAFeatures:
    """Convenience wrapper for the common single-chain A3M case."""

    return create_features_from_raw_msa(
        load_a3m_file(file_name, sequence=sequence),
        max_msa_clusters=max_msa_clusters,
        max_extra_msa=max_extra_msa,
        seed=seed,
    )
