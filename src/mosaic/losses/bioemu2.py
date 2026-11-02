"""BioEmu2 (``biojemu2``) plumbing: binder sequence injection, the trunk + sampler
forward as a `StructureModelOutput`, and the multi-sample loss.

BioEmu2 has no confidence heads: `plddt`, `pae` and `pae_logits` are NaN, so any loss
built on them (pLDDT, PAE, iPTM, ...) is NaN too.

Convention (as for every backend): the binder is chain 0, at token positions
``[0, L)``. As for Boltz, a design pack featurizes the binder as unknown residues
(backbone + CB atoms) and the soft sequence enters through the sequence channels --
``res_type``, ``profile`` and the MSA query row (see `set_binder_sequence`).
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from biojemu2 import BioEmu2, Features, TrunkEmbeddings
from jaxtyping import Array, Float, Int

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import (
    PAE_BINS,
    StructureModelOutput,
    reduce_samples,
)
from mosaic.sampling import Sampler, sample
from mosaic.sampling.bioemu2 import bioemu2_denoiser

# Boltz's token alphabet: <pad>, gap, then the 20 amino acids in mosaic order, UNK, ...
NUM_TOKENS = 33
FIRST_AMINO_ACID = 2
PROTEIN = 0  # `mol_type`
# BioEmu2 keeps Boltz-1's distogram head (64 bins, 2-22 Å).
DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
BACKBONE_ATOM37 = np.array([ATOM37_INDEX[name] for name in ("N", "CA", "C", "O")])


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


def prepare_features(
    pack: BioEmu2FeaturePack, pssm: Float[Array, "L 20"] | None
) -> BioEmu2FeaturePack:
    if pssm is None:
        return pack
    return eqx.tree_at(lambda p: p.features, pack, set_binder_sequence(pssm, pack.features))


def bioemu2_output(
    pack: BioEmu2FeaturePack, trunk: TrunkEmbeddings, coords: Float[Array, "N 3"]
) -> StructureModelOutput:
    """One sample (`mosaic.sampling.sample` on `bioemu2_denoiser`) and the distogram
    head as a `StructureModelOutput`; the confidence fields are NaN."""
    features = pack.features
    T = features.tokens.res_type.shape[0]
    atom37_coords, atom37_mask = scatter_atom37(
        atom_coords=coords,
        atom_to_token=features.atoms.atom_to_token,
        atom37_idx=pack.atom37_idx,
        n_token=T,
    )
    return StructureModelOutput(
        distogram_logits=trunk.distogram,
        distogram_bins=jnp.array(DISTOGRAM_BINS, dtype=jnp.float32),
        plddt=jnp.full((T,), jnp.nan),
        pae=jnp.full((T, T), jnp.nan),
        pae_logits=jnp.full((T, T, len(PAE_BINS)), jnp.nan),
        pae_bins=jnp.array(PAE_BINS, dtype=jnp.float32),
        structure_coordinates=coords[None],
        backbone_coordinates=atom37_coords[:, BACKBONE_ATOM37],
        full_sequence=features.tokens.res_type[:, FIRST_AMINO_ACID : FIRST_AMINO_ACID + 20],
        asym_id=features.trunk.asym_id,
        residue_idx=features.trunk.residue_index,
        atom37_coords=atom37_coords,
        atom37_mask=atom37_mask,
    )


class MultiSampleBioEmu2Loss(LossTerm):
    """Run the trunk once and the sampler ``num_samples`` times.

    With ``reduction=jnp.mean`` this is the expected loss over samples."""

    model: BioEmu2
    pack: BioEmu2FeaturePack
    loss: LossTerm | LinearCombination
    sampler: Sampler
    recycling_steps: int = eqx.field(static=True, default=1)
    num_samples: int = eqx.field(static=True, default=1)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)
    name: str = "bioemu2"

    def __call__(self, sequence: Float[Array, "N 20"], *, key):
        pack = prepare_features(self.pack, sequence)
        trunk = self.model.embed(pack.features, num_recycles=self.recycling_steps - 1)
        denoise, atom_mask = bioemu2_denoiser(self.model, pack.features, trunk)

        def single_sample(key):
            coords = sample(denoise, atom_mask, self.sampler, key=key)
            return self.loss(sequence, bioemu2_output(pack, trunk, coords), key=key)

        values, auxiliary = jax.vmap(single_sample)(jax.random.split(key, self.num_samples))
        return reduce_samples(values, auxiliary, self.reduction, self.num_samples)
