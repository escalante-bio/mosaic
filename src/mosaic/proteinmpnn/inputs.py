"""Build ProteinMPNN inputs from a PDB file.

Mirrors ColabDesign's ``mk_mpnn_model().prep_inputs`` so that mosaic can design
sequences on the same backbones without going through ColabDesign.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

from mosaic.proteinmpnn.sampling import encode_sequence

__all__ = ["MPNNInputs", "inputs_from_pdb", "parse_fixed_positions"]

_BACKBONE = ("N", "CA", "C", "O")

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class MPNNInputs:
    X: np.ndarray  # (L, 4, 3) N, CA, C, O
    S: np.ndarray  # (L,) indices into MPNN_ALPHABET
    mask: np.ndarray  # (L,) bool
    residue_idx: np.ndarray  # (L,) per-chain, offset by chain
    chain_index: np.ndarray  # (L,) 0-based chain number
    chains: list[str]
    lengths: list[int]
    residue_numbers: np.ndarray  # (L,) author numbering, for fix_pos parsing

    @property
    def sequence(self) -> str:
        from mosaic.proteinmpnn.sampling import decode_sequence

        return decode_sequence(self.S)

    def chain_slice(self, chain: str) -> slice:
        i = self.chains.index(chain)
        start = int(sum(self.lengths[:i]))
        return slice(start, start + self.lengths[i])


def inputs_from_pdb(
    pdb_path: str | Path,
    chains: str | list[str] | None = None,
) -> MPNNInputs:
    """Parse ``pdb_path`` into ProteinMPNN inputs.

    ``chains`` selects and orders the chains (``"A,B"`` or ``["A", "B"]``).
    """
    structure = gemmi.read_structure(str(pdb_path))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    model = structure[0]

    available = [chain.name for chain in model]
    if chains is None:
        selected = available
    else:
        if isinstance(chains, str):
            chains = [c.strip() for c in chains.split(",") if c.strip()]
        missing = [c for c in chains if c not in available]
        if missing:
            raise ValueError(f"{pdb_path}: chains {missing} not found in {available}")
        selected = list(chains)

    X, S, mask, residue_idx, chain_index, residue_numbers = [], [], [], [], [], []
    lengths = []

    for chain_number, chain_name in enumerate(selected):
        chain = model[chain_name]
        n_residues = 0
        for position, residue in enumerate(chain):
            one_letter = _THREE_TO_ONE.get(residue.name, "X")
            coords = np.full((4, 3), np.nan, dtype=np.float32)
            found = 0
            for atom_i, atom_name in enumerate(_BACKBONE):
                atom = residue.find_atom(atom_name, "*")
                if atom is not None:
                    coords[atom_i] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    found += 1
            if found == 0:
                continue

            X.append(np.nan_to_num(coords))
            S.append(one_letter)
            mask.append(found == len(_BACKBONE))
            residue_idx.append(position + 100 * chain_number)
            chain_index.append(chain_number)
            residue_numbers.append(residue.seqid.num)
            n_residues += 1
        lengths.append(n_residues)

    if not X:
        raise ValueError(f"{pdb_path}: no residues with backbone atoms found")

    return MPNNInputs(
        X=np.stack(X).astype(np.float32),
        S=encode_sequence("".join(S)),
        mask=np.array(mask, dtype=bool),
        residue_idx=np.array(residue_idx, dtype=np.int32),
        chain_index=np.array(chain_index, dtype=np.int32),
        chains=selected,
        lengths=lengths,
        residue_numbers=np.array(residue_numbers, dtype=np.int32),
    )


def parse_fixed_positions(spec: str, inputs: MPNNInputs) -> np.ndarray:
    """Resolve a ColabDesign-style position spec to flat indices.

    Accepts chain-wide selections (``"A"``), single residues (``"B12"``) and
    ranges (``"B12-20"``), comma separated.  Residue numbers refer to the
    author numbering in the source PDB.
    """
    fixed: set[int] = set()
    offsets = {c: int(sum(inputs.lengths[:i])) for i, c in enumerate(inputs.chains)}

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        if token in offsets:
            start = offsets[token]
            fixed.update(range(start, start + inputs.lengths[inputs.chains.index(token)]))
            continue

        chain = token[0]
        if chain not in offsets:
            raise ValueError(f"unknown chain in fix_pos token {token!r}")
        body = token[1:]
        lo, _, hi = body.partition("-")
        lo_num = int(lo)
        hi_num = int(hi) if hi else lo_num

        chain_mask = inputs.chain_index == inputs.chains.index(chain)
        in_range = (
            chain_mask
            & (inputs.residue_numbers >= lo_num)
            & (inputs.residue_numbers <= hi_num)
        )
        found = np.flatnonzero(in_range)
        if found.size == 0:
            raise ValueError(f"fix_pos token {token!r} matched no residues")
        fixed.update(found.tolist())

    return np.array(sorted(fixed), dtype=np.int32)
