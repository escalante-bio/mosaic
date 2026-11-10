"""What the jfold backends' tests share: comparing arrays, and the structures the writers
write."""

import dataclasses

import gemmi
import jax
import numpy as np
from jfold.blocks.msa import MSAModule


def rel_err(ours, ref) -> float:
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    return float(np.linalg.norm(ours - ref) / np.linalg.norm(ref))


def atom_labels(st: gemmi.Structure) -> list[tuple[str, str, int, str]]:
    """Each atom's chain, residue name and number, and name, in order."""
    return [(c.name, r.name, r.seqid.num, a.name) for c in st[0] for r in c for a in r]


def atom_positions(st: gemmi.Structure) -> np.ndarray:
    return np.array([a.pos.tolist() for c in st[0] for r in c for a in r])


def reading_every_msa_row(model):
    """`model` with its MSA modules reading every row on every pass, as the old JAX ports
    do (upstream draws rows at random each pass), for comparisons with them; cut the MSA to
    the rows the port reads."""
    is_msa = lambda x: isinstance(x, MSAModule)
    every_row = lambda x: dataclasses.replace(x, max_rows=None, min_rows=None)
    return jax.tree.map(
        lambda x: every_row(x) if is_msa(x) else x, model, is_leaf=is_msa
    )
