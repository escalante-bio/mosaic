"""Helpers for using RFD4-Proteina, through jrfd4-proteina, with mosaic.

Sampling, partial diffusion, inverse folding and beam search are implemented in
`jrfd4_proteina`, and design problems are featurized with `jrfd4_proteina.convert.featurize`.
This module loads the model, uses mosaic losses as search rewards, and converts designs to
mosaic sequences and gemmi structures.
"""

import dataclasses
import os

import equinox as eqx
import gemmi
import jax
import numpy as np
from jaxtyping import Array, Bool, Float
from jrfd4_proteina import Batch, Design, RFD4Proteina, SamplingConfig, io

from ..cache import resolve_cache
from ..common import TOKENS, LossTerm
from ..losses.atom37 import ATOM37_NAMES

BUNDLE = "nvidia/RFD4-Proteina"
REVISION = "48377e45eb25a4fce8d2aff56c1a659b726d2106"

# Unified atom37 vocabulary: index 0 is the mask token and 1-20 are the standard amino acids in mosaic's
# `TOKENS` order, so a mosaic token index is the unified index minus one.
_THREE_LETTER = dict(zip(TOKENS, ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                  "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")))


def load_rfd4_proteina(cache: str | os.PathLike | None = None, *, revision: str = REVISION) -> RFD4Proteina:
    """Load the released RFD4-Proteina model.

    The first call converts the PyTorch checkpoint (Hugging Face `nvidia/RFD4-Proteina` at `revision`) and
    saves it under the mosaic cache; later calls load the saved arrays without torch. A cached model saved by
    an older jrfd4-proteina that lacks one of `RFD4Proteina`'s fields (e.g. the AE encoder) is reconverted.
    """
    path = resolve_cache(cache, "rfd4_proteina", revision[:7], create=True) / "rfd4_proteina"
    if path.with_suffix(".eqx").exists():
        model = io.load(path)
        if all(hasattr(model, field.name) for field in dataclasses.fields(RFD4Proteina)):
            return model
    from jrfd4_proteina import convert

    io.save(convert.load(BUNDLE, revision), path)
    return io.load(path)


def inverse_folding_config(config: SamplingConfig) -> SamplingConfig:
    """Return `config` with 3 trunk passes at every step.

    This is the reference's recommended `recycle_strategy=medium/flat` for inverse folding."""
    return dataclasses.replace(config, recycle_schedule=((config.nsteps, 3),))


def centered_ca(coords: Float[Array, "N 37 3"], atom_mask: Bool[Array, "N 37"]) -> Float[Array, "N 3"]:
    """Return CA coordinates in Å, centered on the heavy-atom centroid of the whole structure.

    `inverse_fold` expects its CA trace in this frame, because the reference pipeline centers
    the input structure this way when there is no absolute coordinate conditioning.
    """
    mask = atom_mask.astype(coords.dtype)[..., None]
    centroid = (coords * mask).sum((0, 1)) / mask.sum()
    return coords[:, 1] - centroid


def design_sequence(design: Design, length: int) -> Float[Array, "N 20"]:
    """Return the one-hot residue types of the first `length` tokens in mosaic token order."""
    return jax.nn.one_hot(design.residue_type[:length] - 1, 20)


class LossReward(eqx.Module):
    """Use a mosaic `LossTerm` as a `jrfd4_proteina.search` reward equal to minus the loss.

    The binder must be the first `binder_length` tokens of the batch, for example with the
    contig `"L, /0, <target>"`, as mosaic's structure prediction losses expect.
    """

    loss: LossTerm
    binder_length: int = eqx.field(static=True)

    def __call__(self, design: Design, *, key):
        value, aux = self.loss(design_sequence(design, self.binder_length), key=key)
        return -value, aux


def design_structure(design: Design, batch: Batch, chain_ids: str = "ABCDEFGHIJ") -> gemmi.Structure:
    """Convert an unbatched decoded design to a gemmi structure with one chain per molecule."""
    residue_type = np.asarray(design.residue_type)
    coords, atom_mask = np.asarray(design.coords), np.asarray(design.atom_mask)
    molecule = np.asarray(batch.molecule_iid)
    keep = np.asarray(batch.padding_mask) & (residue_type >= 1) & (residue_type <= 20)

    model = gemmi.Model(1)
    for chain_index, mol in enumerate(dict.fromkeys(molecule[keep])):
        chain = gemmi.Chain(chain_ids[chain_index])
        for number, i in enumerate(np.flatnonzero(keep & (molecule == mol)), start=1):
            residue = gemmi.Residue()
            residue.name = _THREE_LETTER[TOKENS[residue_type[i] - 1]]
            residue.seqid = gemmi.SeqId(number, " ")
            residue.entity_type = gemmi.EntityType.Polymer
            for slot in np.flatnonzero(atom_mask[i]):
                atom = gemmi.Atom()
                atom.name = ATOM37_NAMES[slot]
                atom.element = gemmi.Element(ATOM37_NAMES[slot][0])
                atom.pos = gemmi.Position(*coords[i, slot])
                atom.occ = 1.0
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)

    structure = gemmi.Structure()
    structure.add_model(model)
    structure.setup_entities()
    return structure
