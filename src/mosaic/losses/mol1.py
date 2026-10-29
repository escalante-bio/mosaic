"""Mol-1 (``jol1``) plumbing: binder sequence injection, in-loop binder repacking, the
trunk/heads forward as a `StructureModelOutput`, and the multi-sample loss.

Convention (as for every backend): the binder is chain 0, at token positions
``[0, L)``. The soft sequence enters through every sequence channel Mol-1 reads --
``restype``, ``profile`` and the MSA query row (see `set_binder_sequence`). In a
`binder_features` pack its atoms -- the reference conformers Mol-1 embeds per atom --
are repacked from ``argmax(PSSM)`` every step (see `refresh_binder_atoms`).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from jol1 import Features, InputEmbedding, Mol1, TrunkState
from jol1.transformer import AtomFeatures

from mosaic.common import TOKENS, LinearCombination, LossTerm
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import (
    PAE_BINS,
    StructureModelOutput,
    reduce_samples,
)
from mosaic.sampling import Sampler, sample
from mosaic.sampling.core import random_rotation
from mosaic.sampling.mol1 import mol1_denoiser

# Mol-1's residue alphabet (Protenix's) starts with the 20 amino acids in mosaic order.
NUM_RESTYPES = 32
MSA_GAP = 31  # "-"


def bin_centers(min_bin: float, max_bin: float, num_bins: int) -> np.ndarray:
    """Mol-1's `get_bin_centers`."""
    width = (max_bin - min_bin) / num_bins
    return (min_bin + width * (np.arange(num_bins) + 0.5)).astype(np.float32)


DISTOGRAM_BINS = bin_centers(2.3125, 21.6875, 64)
PLDDT_BINS = bin_centers(0.0, 1.0, 50)
BACKBONE_ATOM37 = np.array([ATOM37_INDEX[name] for name in ("N", "CA", "C", "O")])


MAX_ATOMS_PER_RESIDUE = 15  # Trp + OXT


def binder_atom_budget(binder_length: int) -> int:
    """Atoms of the all-Trp placeholder binder of `binder_features` (14 per residue,
    plus OXT on the C-terminal one): enough for any sequence."""
    return (MAX_ATOMS_PER_RESIDUE - 1) * binder_length + 1


class ResidueAtomTemplates(eqx.Module):
    """Per-residue (mosaic-20 order) reference atoms, mid-chain (variant 0) and
    C-terminal (variant 1, with OXT), as ``[2, 20, MAX_ATOMS_PER_RESIDUE, ...]`` tables.
    Real atoms fill the leading slots in Mol-1's order (N, CA, C, O, side chain[, OXT])
    and are marked by ``atoms.atom_mask``."""

    atoms: AtomFeatures
    atom37_idx: Int[Array, "2 20 A"]
    rep_offset: Int[Array, "2 20"]  # distogram representative atom (CB; CA for Gly)


class Mol1FeaturePack(eqx.Module):
    """Model inputs plus the per-atom / per-token indices `StructureModelOutput` needs."""

    features: Features
    atom37_idx: Int[Array, " N"]  # atom37 slot of each atom, -1 for none (non-protein)
    plddt_atom_idx: Int[Array, " T"]  # CA for protein tokens, else the representative atom
    # Set by `binder_features`: the binder (chain 0, allocated at `binder_atom_budget`)
    # is repacked from argmax(PSSM) every step (`refresh_binder_atoms`).
    templates: ResidueAtomTemplates | None = None
    binder_length: int = eqx.field(static=True, default=0)


def build_pack(feats: dict, atom_array) -> Mol1FeaturePack:
    """From the reference feature dict and ``AtomArray`` (`jol1.reference`)."""
    features = Features.from_reference_dict(feats)
    names = np.asarray(atom_array.atom_name).astype(str)
    is_protein = np.asarray(feats["is_protein"]).astype(bool)
    atom37_idx = np.array([ATOM37_INDEX.get(n, -1) for n in names], np.int32)
    ca = np.flatnonzero(is_protein & (names == "CA"))
    plddt_atom = np.asarray(features.tokens.rep_atom).copy()
    plddt_atom[np.asarray(feats["atom_to_token_idx"])[ca]] = ca
    return Mol1FeaturePack(
        features=features,
        atom37_idx=jnp.asarray(np.where(is_protein, atom37_idx, -1)),
        plddt_atom_idx=jnp.asarray(plddt_atom, jnp.int32),
    )


# ---------------------------------------------------------------------------
# Binder sequence
# ---------------------------------------------------------------------------


def set_binder_sequence(pssm: Float[Array, "L 20"], features: Features) -> Features:
    """Splice the soft sequence into ``restype``, ``profile`` and the MSA query row of
    the first ``L`` tokens.

    Mol-1 takes ``profile`` per chain, as the mean over that chain's own MSA rows: the
    rows with a residue in its columns. Swapping the query row is then a rank-one update
    by 1 / that depth, which keeps the rest of a binder MSA's statistics; a binder
    without an MSA (`binder_features`) has depth 1, so its profile becomes the sequence.
    Exact unless the featurizer cropped the binder's MSA rows."""
    L = pssm.shape[0]
    # Padding, not a matmul: GPU default (TF32) matmuls would round the PSSM.
    soft = jnp.pad(pssm, ((0, 0), (0, NUM_RESTYPES - 20)))
    tokens, msa = features.tokens, features.msa.msa
    depth = (msa[:, :L, MSA_GAP] < 0.5).any(-1).sum()
    old_query = tokens.restype[:L]  # one-hot of the MSA query row at the binder positions
    # Subtract first: exact (``soft``) for depth 1.
    profile = tokens.profile[:L] - old_query / depth + soft / depth
    return eqx.tree_at(
        lambda f: (f.tokens.restype, f.tokens.profile, f.msa.msa),
        features,
        (tokens.restype.at[:L].set(soft), tokens.profile.at[:L].set(profile), msa.at[0, :L].set(soft)),
    )


# ---------------------------------------------------------------------------
# In-loop binder repack
# ---------------------------------------------------------------------------
#
# Mol-1 embeds every atom's reference conformer (position, element, name, charge), so a
# binder featurized as one fixed sequence would carry that sequence's side chains
# whatever the PSSM says. As for ESMFold2 and Protenix, the binder is allocated at the
# per-residue maximum (`binder_atom_budget`), and every step the argmax residues' atoms
# are gathered from per-residue templates and packed tightly at the front of the atom
# axis, the target atoms shifted to follow and the freed tail masked as padding (jol1
# ignores padding atoms). Static shapes, no recompiles. The argmax has no gradient: the
# PSSM gradient flows through the sequence channels. Each binder residue's reference
# conformer gets a fresh random pose every step, as the featurizer's augmentation.


def residue_atom_templates(data_dir) -> ResidueAtomTemplates:
    """Read off one featurization: residues 1-20 of ``G + TOKENS + G`` (mid-chain) and
    the last residue of each ``G + aa`` (C-terminal). ``ref_pos`` is centred over the
    residue (`refresh_binder_atoms` applies the random pose)."""
    from jol1.reference import reference_feature_dict

    sequences = ["G" + TOKENS + "G", *("G" + aa for aa in TOKENS)]
    entry = {
        "name": "templates",
        "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in sequences],
    }
    feats, atom_array = reference_feature_dict(entry, use_msa=False, data_dir=data_dir)
    pack = build_pack(feats, atom_array)
    atoms, atom37_idx, rep_atom = jax.tree.map(
        np.asarray, (pack.features.atoms, pack.atom37_idx, pack.features.tokens.rep_atom)
    )
    restype = np.asarray(feats["restype"]).argmax(-1)

    def residue(token: int, aa: int, c_terminal: bool) -> ResidueAtomTemplates:
        idx = np.flatnonzero(atoms.atom_to_token == token)
        names = [str(n) for n in atom_array.atom_name[idx]]
        assert restype[token] == aa, (TOKENS[aa], restype[token])
        assert names[:4] == ["N", "CA", "C", "O"], names
        assert (names[-1] == "OXT") == c_terminal, names

        def pad(x):
            return np.pad(x[idx], [(0, MAX_ATOMS_PER_RESIDUE - len(idx))] + [(0, 0)] * (x.ndim - 1))

        centred = eqx.tree_at(lambda a: a.ref_pos, atoms, atoms.ref_pos - atoms.ref_pos[idx].mean(0))
        return ResidueAtomTemplates(
            atoms=jax.tree.map(pad, centred),
            atom37_idx=pad(atom37_idx),
            rep_offset=np.int32(rep_atom[token] - idx[0]),
        )

    mid_chain = [residue(1 + aa, aa, False) for aa in range(20)]
    c_terminal = [residue(len(sequences[0]) + 2 * aa + 1, aa, True) for aa in range(20)]
    return jax.tree.map(
        lambda *xs: jnp.asarray(np.stack(xs).reshape(2, 20, *np.shape(xs[0]))),
        *mid_chain,
        *c_terminal,
    )


def refresh_binder_atoms(
    pack: Mol1FeaturePack, pssm: Float[Array, "L 20"], key: jax.Array
) -> Mol1FeaturePack:
    """Repack the binder's atoms as ``argmax(pssm)`` (see the comment above): every
    atom-axis feature and atom index of the pack, rebuilt for the new layout.

    ``ref_pos`` gets the featurizer's augmentation: per residue, a random translation in
    [-1, 1]^3 and then a uniform random rotation (``key``)."""
    L, tmpl, atoms = pack.binder_length, pack.templates, pack.features.atoms
    assert pssm.shape == (L, 20), (pssm.shape, L)
    N = atoms.atom_to_token.shape[0]
    allocated = binder_atom_budget(L)
    ids = jnp.argmax(pssm, axis=-1)
    variant = (jnp.arange(L) == L - 1).astype(jnp.int32)  # C-terminal residue has OXT

    def gather(table):  # [2, 20, ...] -> [L, ...]
        return table[variant, ids]

    binder = jax.tree.map(gather, tmpl.atoms)  # [L, A, ...]
    k = binder.atom_mask.sum(-1)
    offset = jnp.cumsum(k) - k  # first atom of each binder residue
    shift = jnp.sum(k) - allocated  # target atoms move by this much
    slot = jnp.arange(MAX_ATOMS_PER_RESIDUE)
    dest = jnp.where(slot < k[:, None], offset[:, None] + slot, N).reshape(-1)  # N: dropped
    target_dest = jnp.arange(allocated, N) + shift

    def repack(binder, field, fill=0):
        """Binder atoms packed tightly, then the target atoms; the freed tail is ``fill``."""
        binder = binder.reshape(-1, *field.shape[1:]).astype(field.dtype)
        packed = jnp.full_like(field, fill).at[dest].set(binder, mode="drop")
        return packed.at[target_dest].set(field[allocated:])

    def token_atoms(binder, field):
        """Per-token atom indices: binder from the new offsets, target shifted."""
        return jnp.concatenate([binder, field[L:] + shift]).astype(field.dtype)

    kt, kr = jax.random.split(key)
    translation = jax.random.uniform(kt, (L, 1, 3), minval=-1.0, maxval=1.0)
    rotation = jax.vmap(random_rotation)(jax.random.split(kr, L))
    ref_pos = jnp.einsum("lac,ldc->lad", binder.ref_pos + translation, rotation)
    # Binder residues own tokens and reference-space uids 0..L-1. The zero-filled tail
    # of `atom_mask` marks the padding.
    token = jnp.broadcast_to(jnp.arange(L)[:, None], (L, MAX_ATOMS_PER_RESIDUE))
    binder = eqx.tree_at(
        lambda a: (a.ref_pos, a.ref_space_uid, a.atom_to_token), binder, (ref_pos, token, token)
    )
    features = eqx.tree_at(
        lambda f: (f.atoms, f.tokens.rep_atom),
        pack.features,
        (
            jax.tree.map(repack, binder, atoms),
            token_atoms(offset + gather(tmpl.rep_offset), pack.features.tokens.rep_atom),
        ),
    )
    return eqx.tree_at(
        lambda p: (p.features, p.atom37_idx, p.plddt_atom_idx),
        pack,
        (
            features,
            repack(gather(tmpl.atom37_idx), pack.atom37_idx, fill=-1),
            token_atoms(offset + 1, pack.plddt_atom_idx),  # CA
        ),
    )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def prepare_features(
    pack: Mol1FeaturePack, pssm: Float[Array, "L 20"] | None, key: jax.Array
) -> Mol1FeaturePack:
    """Apply the sequence and, for a `binder_features` pack, the repack (``key`` drives
    the conformer augmentation)."""
    if pssm is None:
        return pack
    if pack.templates is not None:
        # A separate stream for the augmentation, leaving ``key`` to the fold.
        pack = refresh_binder_atoms(pack, pssm, jax.random.fold_in(key, 0x70A7))
    return eqx.tree_at(lambda p: p.features, pack, set_binder_sequence(pssm, pack.features))


def mol1_forward_from_trunk(
    model: Mol1,
    pack: Mol1FeaturePack,
    inputs: InputEmbedding,
    state: TrunkState,
    *,
    sampler: Sampler,
    key,
) -> StructureModelOutput:
    """Diffusion sample, confidence and distogram heads as a `StructureModelOutput`."""
    features = pack.features
    denoise, atom_mask = mol1_denoiser(model, features, inputs, state)
    coords = sample(denoise, atom_mask, sampler, key=key)
    confidence = model.confidence(features, inputs, state, coords)

    plddt = (jax.nn.softmax(confidence.plddt_logits[pack.plddt_atom_idx]) * PLDDT_BINS).sum(-1)
    pae = (jax.nn.softmax(confidence.pae_logits) * PAE_BINS).sum(-1)
    atom37_coords, atom37_mask = scatter_atom37(
        atom_coords=coords,
        atom_to_token=features.atoms.atom_to_token,
        atom37_idx=pack.atom37_idx,
        n_token=features.num_tokens,
    )
    return StructureModelOutput(
        distogram_logits=model.distogram_logits(state),
        distogram_bins=jnp.asarray(DISTOGRAM_BINS),
        plddt=plddt,
        pae=pae,
        pae_logits=confidence.pae_logits,
        pae_bins=jnp.asarray(PAE_BINS, jnp.float32),
        structure_coordinates=coords[None],
        backbone_coordinates=atom37_coords[:, BACKBONE_ATOM37],
        full_sequence=features.tokens.restype[:, :20],
        asym_id=features.tokens.asym_id,
        residue_idx=features.tokens.residue_index,
        atom37_coords=atom37_coords,
        atom37_mask=atom37_mask,
    )


class MultiSampleMol1Loss(LossTerm):
    """Run the trunk once and the diffusion + confidence heads ``num_samples`` times.

    With ``reduction=jnp.mean`` this is the expected loss over diffusion samples."""

    model: Mol1
    pack: Mol1FeaturePack
    loss: LossTerm | LinearCombination
    sampler: Sampler
    recycling_steps: int = eqx.field(static=True, default=1)
    msa_rows: int | None = eqx.field(static=True, default=None)  # see `Mol1.trunk`
    num_samples: int = eqx.field(static=True, default=1)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)
    name: str = "mol1"

    def __call__(self, sequence: Float[Array, "N 20"], *, key):
        pack = prepare_features(self.pack, sequence, key)
        inputs = self.model.embed(pack.features)
        state = self.model.trunk(
            pack.features, inputs, num_cycles=self.recycling_steps, msa_rows=self.msa_rows
        )

        def single_sample(key):
            output = mol1_forward_from_trunk(
                self.model, pack, inputs, state, sampler=self.sampler, key=key
            )
            return self.loss(sequence, output, key=key)

        values, auxiliary = jax.vmap(single_sample)(jax.random.split(key, self.num_samples))
        return reduce_samples(values, auxiliary, self.reduction, self.num_samples)

