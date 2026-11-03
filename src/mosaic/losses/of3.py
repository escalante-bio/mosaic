"""Mosaic loss integration for OpenFold3.

Provides:
- Token vocabulary mapping between mosaic's 20-AA and OF3's 32-dim restype
- Binder sequence injection into OF3 Batch
- the atom37 view of OF3 coordinates
The forward itself is `mosaic.structure_model` over `mosaic.models.of3.OF3`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from jopenfold3._vendor.openfold3.core.data.resources.token_atom_constants import (
    STANDARD_RESIDUES_WITH_GAP_3,
    TOKEN_NAME_TO_ATOM_NAMES,
)
from jopenfold3.batch import Batch

from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37

# OF3's first 20 indices are standard AAs in identical order to mosaic's TOKENS:
#   A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9, L=10, K=11,
#   M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19
# Indices 20-31 are UNK, RNA(5), DNA(5), GAP.
_MOSAIC_TO_OF3 = np.zeros((20, 32), dtype=np.float32)
_MOSAIC_TO_OF3[:20, :20] = np.eye(20)

# OF3 vocab: 0-19 protein, 20 UNK, 21-25 RNA, 26-30 DNA, 31 GAP
GAP_IDX = 31


def _build_of3_atom37_table() -> np.ndarray:
    """`[restype, tokatom_idx] -> atom37_idx` lookup, -1 where invalid.

    OF3's `STANDARD_RESIDUES_WITH_GAP_3` is the 32-letter alphabet of token
    names; `TOKEN_NAME_TO_ATOM_NAMES[token]` gives the per-token atom-name
    list in OF3's all-atom layout order. We map each name to its atom37
    slot (if any).
    """
    max_tokatom = max(
        len(TOKEN_NAME_TO_ATOM_NAMES.get(n, [])) for n in STANDARD_RESIDUES_WITH_GAP_3
    )
    table = -np.ones((len(STANDARD_RESIDUES_WITH_GAP_3), max_tokatom), dtype=np.int32)
    for rt, token_name in enumerate(STANDARD_RESIDUES_WITH_GAP_3):
        for tokatom_idx, atom_name in enumerate(TOKEN_NAME_TO_ATOM_NAMES.get(token_name, [])):
            if atom_name in ATOM37_INDEX:
                table[rt, tokatom_idx] = ATOM37_INDEX[atom_name]
    return table


_OF3_TOKATOM_TO_ATOM37 = _build_of3_atom37_table()




def set_binder_sequence(new_sequence: Float[Array, "N 20"], batch: Batch,
                        lut=None) -> Batch:
    """Inject a soft sequence (PSSM) into the binder positions of an OF3 Batch.

    Assumes the binder occupies the first ``new_sequence.shape[0]`` tokens.
    Always updates ``restype``, ``profile``, ``msa`` (soft, differentiable).

    """

    binder_len = new_sequence.shape[0]
    of3_sequence = new_sequence @ jnp.array(_MOSAIC_TO_OF3)  # [N, 32]

    new_restype = batch.restype.at[0, :binder_len, :].set(of3_sequence)

    new_profile = batch.profile.at[0, :binder_len].set(of3_sequence)
    # Set query row in MSA
    new_msa = batch.msa.at[0, 0, :binder_len].set(of3_sequence)
    # Set the rest of MSA for binder to GAP (no hits)
    new_msa = new_msa.at[0, 1:, :binder_len,].set(jax.nn.one_hot(GAP_IDX, 32))

    return eqx.tree_at(
        lambda b: (b.restype, b.profile, b.msa),
        batch,
        (new_restype, new_profile, new_msa),
    )




def _bin_centers(bin_min: float, bin_max: float, num_bins: int) -> np.ndarray:
    width = (bin_max - bin_min) / num_bins
    return np.linspace(bin_min + 0.5 * width, bin_max - 0.5 * width, num_bins)


# Distogram: 64 bins, 2.0–22.0 Å
DISTOGRAM_BINS = _bin_centers(2.0, 22.0, 64)
# pLDDT: 50 bins, 0.0–1.0
PLDDT_BINS = _bin_centers(0.0, 1.0, 50)




def of3_atom37(batch: Batch, coords: Float[Array, "A 3"]):
    """Atom37 view: atoms within a token are contiguous from `start_atom_index`, so
    each atom's slot follows from its residue type and offset; padding atoms
    (`batch.atom_mask` 0) have none."""
    start = batch.start_atom_index[0].astype(jnp.int32)
    n_tokens = batch.restype.shape[1]
    n_atoms = coords.shape[0]
    restype_idx = batch.restype[0].argmax(-1)                          # [N_token] in OF3 32-letter
    atom_to_token = jnp.clip(
        jnp.searchsorted(start, jnp.arange(n_atoms), side="right") - 1,
        0, n_tokens - 1,
    )
    tokatom_idx = jnp.arange(n_atoms) - start[atom_to_token]
    atom37_idx = jnp.asarray(_OF3_TOKATOM_TO_ATOM37)[restype_idx[atom_to_token], tokatom_idx]
    atom37_idx = jnp.where(batch.atom_mask[0] > 0.5, atom37_idx, jnp.int32(-1))
    return scatter_atom37(coords, atom_to_token, atom37_idx, n_tokens)
