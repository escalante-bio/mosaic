"""Atom feature lookup tables for soft-sequence binder design.

Pre-computes reference atom features for all 20 standard amino acids,
padded to 14 atoms (TRP = max), enabling differentiable blending under
JIT via PSSM weights.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

MAX_ATOMS_PER_TOKEN = 14  # TRP has the most atoms

# Standard AA 1-letter codes in OF3 index order (= mosaic TOKENS order)
_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


class AtomLookup(eqx.Module):
    """Padded atom features for all 20 standard amino acids.

    Each array has shape [20, 14, ...] where 14 is the max atoms per
    token (TRP). Unused atom slots are zero-padded with mask=0.

    This is an eqx.Module (JAX pytree) so its arrays are traced (not
    hashed) when the enclosing loss term is JIT-compiled by mosaic.
    """

    ref_pos: Float[Array, "20 14 3"]
    ref_element: Float[Array, "20 14 119"]
    ref_charge: Float[Array, "20 14"]
    ref_mask: Float[Array, "20 14"]
    ref_atom_name_chars: Float[Array, "20 14 4 64"]
    atom_mask: Float[Array, "20 14"]


@functools.lru_cache(maxsize=1)
def build_atom_lookup() -> AtomLookup:
    """Build lookup tables by featurizing a 20-residue chain (one per AA).

    Uses the existing PyTorch featurization pipeline, then extracts and
    pads atom features per residue. Cached so it's built once per process.
    """
    from mosaic.structure_prediction import TargetChain

    from of3_jax.mosaic.model import _build_query_set, _compute_msas, _featurize

    chain = TargetChain(sequence=_AA_ORDER, use_msa=False)
    query_set = _build_query_set([chain])
    tmp_dir = Path(tempfile.mkdtemp(prefix="of3_atom_lut_"))
    query_set = _compute_msas(query_set, chains=[chain], msa_dir=tmp_dir / "msas")
    features, _ = _featurize(query_set)

    def to_np(key):
        v = features[key]
        return v.numpy() if hasattr(v, "numpy") else np.asarray(v)

    start = to_np("start_atom_index")
    n_atoms = to_np("num_atoms_per_token")
    ref_pos = to_np("ref_pos").astype(np.float32)
    ref_element = to_np("ref_element").astype(np.float32)
    ref_charge = to_np("ref_charge").astype(np.float32)
    ref_mask = to_np("ref_mask").astype(np.float32)
    ref_atom_name = to_np("ref_atom_name_chars").astype(np.float32)
    atom_mask_arr = to_np("atom_mask").astype(np.float32)

    # Pad each residue's atoms to MAX_ATOMS_PER_TOKEN
    lut = {
        "ref_pos": np.zeros((20, MAX_ATOMS_PER_TOKEN, 3), np.float32),
        "ref_element": np.zeros((20, MAX_ATOMS_PER_TOKEN, 119), np.float32),
        "ref_charge": np.zeros((20, MAX_ATOMS_PER_TOKEN), np.float32),
        "ref_mask": np.zeros((20, MAX_ATOMS_PER_TOKEN), np.float32),
        "ref_atom_name_chars": np.zeros((20, MAX_ATOMS_PER_TOKEN, 4, 64), np.float32),
        "atom_mask": np.zeros((20, MAX_ATOMS_PER_TOKEN), np.float32),
    }

    for i in range(20):
        s = int(start[i])
        n = int(n_atoms[i])
        lut["ref_pos"][i, :n] = ref_pos[s : s + n]
        lut["ref_element"][i, :n] = ref_element[s : s + n]
        lut["ref_charge"][i, :n] = ref_charge[s : s + n]
        lut["ref_mask"][i, :n] = ref_mask[s : s + n]
        lut["ref_atom_name_chars"][i, :n] = ref_atom_name[s : s + n]
        lut["atom_mask"][i, :n] = atom_mask_arr[s : s + n]

    return AtomLookup(**{k: jnp.array(v) for k, v in lut.items()})
