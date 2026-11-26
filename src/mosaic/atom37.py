"""Canonical protein atom coordinates and Gemmi serialization."""

import equinox as eqx
import gemmi
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from mosaic.common import TOKENS

# Canonical atom37 ordering (matches AlphaFold residue_constants.atom_types).
ATOM37_NAMES: tuple[str, ...] = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)
assert len(ATOM37_NAMES) == 37

ATOM37_INDEX: dict[str, int] = {name: i for i, name in enumerate(ATOM37_NAMES)}


def scatter_atom37(
    atom_coords: Float[Array, "N_atom 3"],
    atom_to_token: Int[Array, " N_atom"],
    atom37_idx: Int[Array, " N_atom"],
    n_token: int,
) -> tuple[Float[Array, "N_token 37 3"], Float[Array, "N_token 37"]]:
    """Scatter all-atom coordinates into a `[n_token, 37, 3]` array and its mask.

    Atoms with `atom37_idx == -1`, such as padding, are dropped. They are moved to
    slot 37 first, because `mode='drop'` ignores only indices past the end and a
    raw -1 would wrap to slot 36 (OXT).
    """
    slot = jnp.where(atom37_idx < 0, 37, atom37_idx)
    coords = (
        jnp.zeros((n_token, 37, 3), dtype=jnp.float32)
        .at[atom_to_token, slot]
        .set(atom_coords, mode="drop")
    )
    mask = (
        jnp.zeros((n_token, 37), dtype=jnp.float32)
        .at[atom_to_token, slot]
        .set(jnp.ones(atom_coords.shape[:-1], dtype=jnp.float32), mode="drop")
    )
    return coords, mask


# Chain names in `asym_id` order (AlphaFold's `PDB_CHAIN_IDS`).
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _aatype(sequence: Float[Array, "N 20"]) -> np.ndarray:
    """Return each token's most likely amino acid, or 20 if its row is all zero, as it is
    for unknown residues."""
    sequence = np.asarray(sequence)
    return np.where(sequence.max(-1) > 0, sequence.argmax(-1), 20)


def atom37_structure(
    coords: Float[Array, "N 37 3"],
    mask: Float[Array, "N 37"],
    aatype: Int[Array, " N"],
    chain_index: Int[Array, " N"],
    residue_number: Int[Array, " N"],
    b_factor: Float[Array, " N"] | None = None,
    chain_names: str = CHAIN_IDS,
) -> gemmi.Structure:
    """Build a protein gemmi.Structure from atom37 coordinates.

    `aatype` indexes `common.TOKENS`, and 20 is an unknown residue, written as UNK.
    Residues with the same `chain_index` form one
    chain named `chain_names[chain_index]`, and chains are written in order of first
    appearance. Residues without atoms, such as padding, ligands and nucleic acids, are
    left out. `b_factor` is per residue. If it is None, gemmi's default is kept.
    """
    coords = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(mask)
    aatype = np.asarray(aatype).astype(np.int32)
    chain_index = np.asarray(chain_index).astype(np.int32)
    residue_number = np.asarray(residue_number).astype(np.int32)
    names = [gemmi.expand_one_letter(c, gemmi.ResidueKind.AA) for c in TOKENS] + ["UNK"]

    model = gemmi.Model("1")
    present = (mask >= 0.5).any(-1)
    for c in dict.fromkeys(chain_index[present].tolist()):
        chain = gemmi.Chain(chain_names[c])
        for i in np.flatnonzero(present & (chain_index == c)):
            residue = gemmi.Residue()
            residue.name = names[aatype[i]]
            residue.seqid = gemmi.SeqId(int(residue_number[i]), " ")
            residue.entity_type = gemmi.EntityType.Polymer
            for slot in np.flatnonzero(mask[i] >= 0.5):
                atom = gemmi.Atom()
                atom.name = ATOM37_NAMES[slot]
                atom.element = gemmi.Element(ATOM37_NAMES[slot][0])
                atom.pos = gemmi.Position(*coords[i, slot])
                atom.occ = 1.0
                if b_factor is not None:
                    atom.b_iso = float(b_factor[i])
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure = gemmi.Structure()
    structure.add_model(model)
    structure.setup_entities()
    return structure


class Atom37(eqx.Module):
    """Coordinates in AlphaFold's atom37 layout."""

    coords: Float[Array, "T 37 3"]
    mask: Float[Array, "T 37"]
