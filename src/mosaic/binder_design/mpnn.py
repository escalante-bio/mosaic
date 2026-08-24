"""Stage 2: ProteinMPNN sequence design on a trajectory backbone.

Wraps :mod:`mosaic.proteinmpnn.sampling` with the position-fixing policy DdCraft
uses: the target chain is always fixed, and optionally the interface residues
found on the trajectory plus any user-specified positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mosaic.proteinmpnn import sampling
from mosaic.proteinmpnn.inputs import MPNNInputs, inputs_from_pdb, parse_fixed_positions
from mosaic.proteinmpnn.mpnn import load_abmpnn, load_mpnn, load_mpnn_sol

logger = logging.getLogger(__name__)

__all__ = ["DesignedSequence", "MPNNDesigner", "load_mpnn_weights"]

_WEIGHTS = {
    "original": load_mpnn,
    "soluble": load_mpnn_sol,
    "abmpnn": load_abmpnn,
}


def load_mpnn_weights(weights: str = "soluble", backbone_noise: float = 0.0):
    """Load ProteinMPNN weights by DdCraft's ``mpnn_weights`` name."""
    try:
        loader = _WEIGHTS[weights]
    except KeyError:
        raise ValueError(
            f"unknown mpnn weights {weights!r}, expected one of {sorted(_WEIGHTS)}"
        ) from None
    return loader(backbone_noise=backbone_noise)


@dataclass
class DesignedSequence:
    """One MPNN design: the binder sequence plus its MPNN statistics."""

    sequence: str
    score: float | None
    seqid: float | None
    full_sequence: str

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "Sequence": self.sequence,
            "MPNN_score": self.score,
            "MPNN_seq_recovery": self.seqid,
        }


class MPNNDesigner:
    """Designs binder sequences for trajectory backbones."""

    def __init__(
        self,
        *,
        weights: str = "soluble",
        backbone_noise: float = 0.0,
        sampling_temp: float = 0.1,
        num_seqs: int = 20,
        omit_AAs: str = "C",
        batch_size: int | None = None,
    ):
        self.model = load_mpnn_weights(weights, backbone_noise)
        self.sampling_temp = sampling_temp
        self.num_seqs = num_seqs
        self.omit_AAs = omit_AAs
        self.batch_size = batch_size

    def fixed_positions(
        self,
        inputs: MPNNInputs,
        *,
        target_chain: str = "A",
        interface_residues: str = "",
        fix_pos: str = "",
    ) -> np.ndarray:
        """Resolve the positions MPNN must not redesign.

        The target chain is always fixed; interface and user-specified residues
        are added on top.  Mirrors DdCraft's ``mpnn_gen_sequence``.
        """
        spec = target_chain
        if interface_residues:
            spec += "," + interface_residues
            logger.info("Fixing interface residues: %s", interface_residues)
        if fix_pos:
            spec += "," + fix_pos
            logger.info("Fixing user-specified positions: %s", fix_pos)
        spec = spec.rstrip(",")
        logger.info("Total fixed positions for MPNN: %s", spec)
        return parse_fixed_positions(spec, inputs)

    def design(
        self,
        trajectory_pdb: str | Path,
        *,
        key,
        binder_chain: str = "B",
        target_chain: str = "A",
        interface_residues: str = "",
        fix_pos: str = "",
        num_seqs: int | None = None,
    ) -> list[DesignedSequence]:
        inputs = inputs_from_pdb(trajectory_pdb, f"{target_chain},{binder_chain}")
        fixed = self.fixed_positions(
            inputs,
            target_chain=target_chain,
            interface_residues=interface_residues,
            fix_pos=fix_pos,
        )

        samples = sampling.sample_batch(
            self.model,
            X=inputs.X,
            S_native=inputs.S,
            residue_idx=inputs.residue_idx,
            chain_encoding_all=inputs.chain_index,
            mask=inputs.mask,
            key=key,
            num_seqs=self.num_seqs if num_seqs is None else num_seqs,
            batch_size=self.batch_size,
            temperature=self.sampling_temp,
            fix_pos=fixed,
            omit_AAs=self.omit_AAs,
        )

        binder = inputs.chain_slice(binder_chain)
        return [
            DesignedSequence(
                sequence=s.sequence[binder],
                score=s.score,
                seqid=s.seqid,
                full_sequence=s.sequence,
            )
            for s in samples
        ]
