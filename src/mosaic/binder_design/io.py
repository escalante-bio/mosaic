"""PDB conventions shared by the design stages.

Mosaic's AF2 representation places the designed binder first, but DdCraft's
filters, RMSD helpers and downstream analysis all expect target chains first
with the binder last, using the residue numbering of the starting structure.
Everything that writes a structure for the filtering stages goes through here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from Bio.PDB import Chain, Model, PDBIO, PDBParser, Structure

logger = logging.getLogger(__name__)

__all__ = [
    "ChainLayoutError",
    "normalize_binder_pdb",
    "normalize_complex_pdb",
    "reference_residue_ids",
]


class ChainLayoutError(RuntimeError):
    """Raised when a predicted structure does not have the expected chains."""


def _protein_residues(chain: Any) -> list[Any]:
    return [residue for residue in chain if residue.id[0] == " "]


def reference_residue_ids(starting_pdb: str | Path, chain_id: str) -> list[tuple]:
    """Residue ids of ``chain_id`` in ``starting_pdb``, in order."""
    model = PDBParser(QUIET=True).get_structure("reference", str(starting_pdb))[0]
    if chain_id not in model:
        raise ChainLayoutError(
            f"Chain '{chain_id}' is missing from starting PDB {starting_pdb}"
        )
    return [residue.id for residue in _protein_residues(model[chain_id])]


def _clone_and_rename_chain(
    source_chain: Any, destination_id: str, residue_ids: Iterable[tuple]
) -> Any:
    protein_residues = _protein_residues(source_chain)
    residue_ids = list(residue_ids)
    if len(protein_residues) != len(residue_ids):
        raise ChainLayoutError(
            f"Predicted chain has {len(protein_residues)} residues but "
            f"{destination_id} expects {len(residue_ids)}"
        )
    chain = Chain.Chain(destination_id)
    for source_residue, residue_id in zip(protein_residues, residue_ids):
        residue = source_residue.copy()
        residue.id = residue_id
        chain.add(residue)
    return chain


def _write(chains: Sequence[Any], output_pdb: str | Path) -> Path:
    structure = Structure.Structure("mosaic")
    model = Model.Model(0)
    structure.add(model)
    for chain in chains:
        model.add(chain)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))
    return output_pdb


def normalize_complex_pdb(
    raw_pdb: str | Path,
    output_pdb: str | Path,
    starting_pdb: str | Path,
    target_chain_ids: list[str],
    binder_chain_id: str,
    binder_length: int,
    structure_conditioned: bool = False,
) -> Path:
    """Reorder a predicted complex into target-chains-then-binder order."""
    raw_model = PDBParser(QUIET=True).get_structure("mosaic", str(raw_pdb))[0]
    raw_chains = list(raw_model.get_chains())
    expected = 1 + len(target_chain_ids)
    if len(raw_chains) != expected:
        raise ChainLayoutError(
            f"Predicted structure contains {len(raw_chains)} chains; expected {expected}"
        )

    target_chains = [
        _clone_and_rename_chain(
            raw_chain, chain_id, reference_residue_ids(starting_pdb, chain_id)
        )
        for raw_chain, chain_id in zip(raw_chains[1:], target_chain_ids)
    ]

    binder_residue_ids = (
        reference_residue_ids(starting_pdb, binder_chain_id)
        if structure_conditioned
        else [(" ", index, " ") for index in range(1, binder_length + 1)]
    )
    binder_chain = _clone_and_rename_chain(
        raw_chains[0], binder_chain_id, binder_residue_ids
    )

    return _write([*target_chains, binder_chain], output_pdb)


def normalize_binder_pdb(
    raw_pdb: str | Path,
    output_pdb: str | Path,
    binder_length: int,
    chain_id: str = "A",
) -> Path:
    """Rename a binder-alone prediction to a single chain numbered from 1.

    DdCraft predicts the binder on its own as chain ``A`` and aligns it against
    the trajectory, so the monomer output has to use that chain id regardless of
    what the folding model emitted.
    """
    raw_model = PDBParser(QUIET=True).get_structure("mosaic", str(raw_pdb))[0]
    raw_chains = list(raw_model.get_chains())
    if len(raw_chains) != 1:
        raise ChainLayoutError(
            f"Binder-alone prediction has {len(raw_chains)} chains; expected 1"
        )
    chain = _clone_and_rename_chain(
        raw_chains[0],
        chain_id,
        [(" ", index, " ") for index in range(1, binder_length + 1)],
    )
    return _write([chain], output_pdb)
