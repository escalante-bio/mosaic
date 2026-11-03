"""BioEmu2 (``biojemu2``) plumbing: the feature pack and binder sequence injection.
The forward itself is `mosaic.structure_model` over `mosaic.models.bioemu2.BioEmu2`.

BioEmu2 has no confidence heads: `plddt`, `pae` and `pae_logits` are NaN, so any loss
built on them (pLDDT, PAE, iPTM, ...) is NaN too.

Convention (as for every backend): the binder is chain 0, at token positions
``[0, L)``. As for Boltz, a design pack featurizes the binder as unknown residues
(backbone + CB atoms) and the soft sequence enters through the sequence channels --
``res_type``, ``profile`` and the MSA query row (see `set_binder_sequence`).
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from biojemu2 import Features
from jaxtyping import Array, Float, Int

from mosaic.losses.atom37 import ATOM37_INDEX

# Boltz's token alphabet: <pad>, gap, then the 20 amino acids in mosaic order, UNK, ...
NUM_TOKENS = 33
FIRST_AMINO_ACID = 2
PROTEIN = 0  # `mol_type`
# BioEmu2 keeps Boltz-1's distogram head (64 bins, 2-22 Å).
DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)


class BioEmu2FeaturePack(eqx.Module):
    """Model inputs plus the atom37 slot of every atom (-1: none, e.g. non-protein)."""

    features: Features
    atom37_idx: Int[Array, " N"]


def build_pack(feats: dict[str, np.ndarray]) -> BioEmu2FeaturePack:
    """From the reference feature dict (numpy, batch size 1). Atom names are decoded
    from ``name_encoding`` (Boltz: ``ord(c) - 32`` per character, 0 = end)."""
    names = ["".join(chr(c + 32) for c in enc if c) for enc in feats["name_encoding"][0]]
    protein = (feats["mol_type"][0] == PROTEIN) & feats["atom_pad_mask"][0].astype(bool)
    atom37_idx = [ATOM37_INDEX.get(n, -1) if p else -1 for n, p in zip(names, protein)]
    return BioEmu2FeaturePack(
        features=Features.from_reference_dict(feats),
        atom37_idx=jnp.array(atom37_idx, dtype=jnp.int32),
    )


def set_binder_sequence(pssm: Float[Array, "L 20"], features: Features) -> Features:
    """Splice the soft sequence into ``res_type``, ``profile`` and the MSA query row of
    the first ``L`` tokens.

    ``profile`` is the mean over all MSA rows, so swapping the query row is a rank-one
    update by 1 / depth that keeps the rest of the MSA's statistics."""
    L = pssm.shape[0]
    # Padding, not a matmul: GPU default (TF32) matmuls would round the PSSM.
    soft = jnp.pad(pssm, ((0, 0), (FIRST_AMINO_ACID, NUM_TOKENS - 20 - FIRST_AMINO_ACID)))
    tokens, msa = features.tokens, features.trunk.msa
    profile = tokens.profile[:L] + (soft - msa[0, :L]) / msa.shape[0]
    return eqx.tree_at(
        lambda f: (f.tokens.res_type, f.tokens.profile, f.trunk.msa),
        features,
        (tokens.res_type.at[:L].set(soft), tokens.profile.at[:L].set(profile), msa.at[0, :L].set(soft)),
    )
