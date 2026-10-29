"""Mol-1 (``jol1``) plumbing: binder sequence injection, in-loop binder repacking, the
trunk/heads forward as a `StructureModelOutput`, and the multi-sample loss.

Convention (as for every backend): the binder is chain 0, at token positions
``[0, L)``. The soft sequence enters through every sequence channel Mol-1 reads --
``restype``, ``profile`` and the MSA query row (the binder has no MSA, so its profile
*is* its query row). Its atoms -- the reference conformers Mol-1 embeds per atom -- are
repacked from ``argmax(PSSM)`` every step (see `refresh_binder_geometry`).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from jol1 import Features, InputEmbedding, Mol1, TrunkState

from mosaic.common import TOKENS, LinearCombination, LossTerm
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import (
    PAE_BINS,
    StructureModelOutput,
    reduce_samples,
)
from mosaic.sampling import Sampler, sample
from mosaic.sampling.mol1 import mol1_denoiser

# Mol-1's residue alphabet (Protenix's) starts with the 20 amino acids in mosaic order.
NUM_RESTYPES = 32


def bin_centers(min_bin: float, max_bin: float, num_bins: int) -> np.ndarray:
    """Mol-1's `get_bin_centers`."""
    width = (max_bin - min_bin) / num_bins
    return (min_bin + width * (np.arange(num_bins) + 0.5)).astype(np.float32)


DISTOGRAM_BINS = bin_centers(2.3125, 21.6875, 64)
PLDDT_BINS = bin_centers(0.0, 1.0, 50)


class Mol1FeaturePack(eqx.Module):
    """Model inputs plus the per-atom / per-token indices `StructureModelOutput` needs."""

    features: Features
    atom37_idx: Int[Array, " N"]  # atom37 slot of each atom, -1 for none (non-protein)
    backbone_atom_idx: Int[Array, "4 T"]  # N, CA, C, O atom of each protein token
    plddt_atom_idx: Int[Array, " T"]  # CA for protein tokens, else the representative atom
    # Repack the binder's atoms from argmax(PSSM) every step (a `binder_features` pack,
    # whose binder is allocated at the 14-atom maximum). Static: selects the forward.
    refresh: bool = eqx.field(static=True, default=False)


def build_pack(feats: dict, atom_array, *, refresh: bool) -> Mol1FeaturePack:
    """From the reference feature dict and ``AtomArray`` (`jol1.reference`)."""
    features = Features.from_reference_dict(feats)
    names = [str(n) for n in atom_array.atom_name]
    is_protein = np.asarray(feats["is_protein"]).astype(bool)
    atom_to_token = np.asarray(feats["atom_to_token_idx"])
    num_tokens = features.num_tokens

    atom37_idx = np.full(len(names), -1, np.int32)
    backbone = np.zeros((4, num_tokens), np.int32)
    plddt_atom = np.asarray(features.tokens.rep_atom).copy()
    for a, (name, protein, tok) in enumerate(zip(names, is_protein, atom_to_token)):
        if not protein:
            continue
        atom37_idx[a] = ATOM37_INDEX.get(name, -1)
        if name in ("N", "CA", "C", "O"):
            backbone[("N", "CA", "C", "O").index(name), tok] = a
        if name == "CA":
            plddt_atom[tok] = a
    return Mol1FeaturePack(
        features=features,
        atom37_idx=jnp.asarray(atom37_idx),
        backbone_atom_idx=jnp.asarray(backbone),
        plddt_atom_idx=jnp.asarray(plddt_atom, jnp.int32),
        refresh=refresh,
    )


# ---------------------------------------------------------------------------
# Binder sequence
# ---------------------------------------------------------------------------


def set_binder_sequence(pssm: Float[Array, "L 20"], features: Features) -> Features:
    """Splice the soft sequence into ``restype``, ``profile`` and the MSA query row of
    the first ``L`` tokens. The binder is featurized without an MSA, so its other MSA
    rows are gaps and its profile is exactly its query row."""
    L = pssm.shape[0]
    # Padding, not a matmul: GPU default (TF32) matmuls would round the PSSM.
    soft = jnp.pad(pssm, ((0, 0), (0, NUM_RESTYPES - 20)))
    tokens, msa = features.tokens, features.msa
    features = eqx.tree_at(
        lambda f: (f.tokens.restype, f.tokens.profile, f.msa.msa),
        features,
        (
            tokens.restype.at[:L].set(soft),
            tokens.profile.at[:L].set(soft),
            msa.msa.at[0, :L].set(soft),
        ),
    )
    return features


# ---------------------------------------------------------------------------
# In-loop binder repack
# ---------------------------------------------------------------------------
#
# Mol-1 embeds every atom's reference conformer (position, element, name, charge), so a
# binder featurized as one fixed sequence would carry that sequence's side chains
# whatever the PSSM says. As for ESMFold2, the binder is allocated at the per-residue
# maximum (Trp: 14 atoms, plus OXT on the C-terminal residue), and every step the
# argmax residues' atoms are gathered from per-residue templates and packed tightly at
# the front of the atom axis, the target atoms shifted to follow and the freed tail
# masked as padding (jol1 ignores padding atoms). Static shapes, no recompiles. The
# argmax is detached: the PSSM gradient flows through the sequence channels.

MAX_ATOMS_PER_RESIDUE = 15  # Trp + OXT


class ResidueAtomTemplates(eqx.Module):
    """Per-residue (mosaic-20 order) reference atoms, mid-chain (variant 0) and
    C-terminal (variant 1, with OXT). Real atoms fill the first ``num_atoms`` slots in
    Mol-1's order (N, CA, C, O, side chain[, OXT])."""

    ref_pos: Float[Array, "2 20 A 3"]
    ref_charge: Float[Array, "2 20 A"]
    ref_mask: Float[Array, "2 20 A"]
    ref_element: Float[Array, "2 20 A 128"]
    ref_atom_name_chars: Float[Array, "2 20 A 256"]
    atom_to_tokatom: Int[Array, "2 20 A"]
    atom37_slot: Int[Array, "2 20 A"]
    num_atoms: Int[Array, "2 20"]
    rep_offset: Int[Array, "2 20"]  # distogram representative atom (CB; CA for Gly)


def residue_atom_templates(data_dir) -> ResidueAtomTemplates:
    """Featurize each amino acid mid-chain (``G?G``) and C-terminal (``G?``) once."""
    from jol1.reference import reference_feature_dict

    A = MAX_ATOMS_PER_RESIDUE
    out = {
        "ref_pos": np.zeros((2, 20, A, 3), np.float32),
        "ref_charge": np.zeros((2, 20, A), np.float32),
        "ref_mask": np.zeros((2, 20, A), np.float32),
        "ref_element": np.zeros((2, 20, A, 128), np.float32),
        "ref_atom_name_chars": np.zeros((2, 20, A, 256), np.float32),
        "atom_to_tokatom": np.zeros((2, 20, A), np.int32),
        "atom37_slot": np.full((2, 20, A), -1, np.int32),
        "num_atoms": np.zeros((2, 20), np.int32),
        "rep_offset": np.zeros((2, 20), np.int32),
    }
    for variant, peptide in enumerate(("G{}G", "G{}")):
        for i, aa in enumerate(TOKENS):
            entry = {
                "name": "template",
                "sequences": [{"proteinChain": {"sequence": peptide.format(aa), "count": 1}}],
            }
            feats, atom_array = reference_feature_dict(entry, use_msa=False, data_dir=data_dir)
            m = np.asarray(feats["atom_to_token_idx"]) == 1
            k = int(m.sum())
            names = [str(n) for n in atom_array.atom_name[m]]
            assert names[:4] == ["N", "CA", "C", "O"], (aa, names)
            assert (names[-1] == "OXT") == (variant == 1), (aa, names)
            assert int(np.argmax(feats["restype"][1])) == i, aa
            out["ref_pos"][variant, i, :k] = feats["ref_pos"][m]
            out["ref_charge"][variant, i, :k] = feats["ref_charge"][m]
            out["ref_mask"][variant, i, :k] = feats["ref_mask"][m]
            out["ref_element"][variant, i, :k] = feats["ref_element"][m]
            out["ref_atom_name_chars"][variant, i, :k] = feats["ref_atom_name_chars"][m].reshape(
                k, -1
            )
            out["atom_to_tokatom"][variant, i, :k] = feats["atom_to_tokatom_idx"][m]
            out["atom37_slot"][variant, i, :k] = [ATOM37_INDEX.get(n, -1) for n in names]
            out["num_atoms"][variant, i] = k
            rep = np.flatnonzero(np.asarray(feats["distogram_rep_atom_mask"])[m])
            out["rep_offset"][variant, i] = int(rep[0])
    return ResidueAtomTemplates(**{k: jnp.asarray(v) for k, v in out.items()})


def refresh_binder_geometry(
    pack: Mol1FeaturePack, features: Features, pssm: Float[Array, "L 20"], tmpl: ResidueAtomTemplates
) -> Mol1FeaturePack:
    """Repack the binder's atoms as ``argmax(pssm)`` (see the comment above). Returns the
    pack with ``features`` and every atom index rebuilt for the new layout."""
    L = pssm.shape[0]
    atoms = features.atoms
    N = atoms.ref_pos.shape[0]
    allocated = 14 * L + 1  # the all-Trp binder of `binder_features`
    num_target = N - allocated
    ids = jax.lax.stop_gradient(jnp.argmax(pssm, axis=-1))
    variant = (jnp.arange(L) == L - 1).astype(jnp.int32)  # C-terminal residue has OXT

    k = tmpl.num_atoms[variant, ids]
    offset = jnp.cumsum(k) - k  # first atom of each binder residue
    shift = jnp.sum(k) - allocated  # target atoms move by this much
    slot = jnp.arange(MAX_ATOMS_PER_RESIDUE)[None, :]
    dest = jnp.where(slot < k[:, None], offset[:, None] + slot, N).reshape(-1)  # N: dropped
    target_dest = jnp.arange(num_target) + allocated + shift

    def repack(binder, field):
        buf = jnp.zeros_like(field)
        buf = buf.at[dest].set(binder.reshape(-1, *field.shape[1:]).astype(field.dtype), mode="drop")
        return buf.at[target_dest].set(field[allocated:])

    def binder_template(x):
        return x[variant, ids]  # [L, A, ...]

    token = jnp.broadcast_to(jnp.arange(L)[:, None], (L, MAX_ATOMS_PER_RESIDUE))
    real_atoms = jnp.arange(N) < N + shift
    new_atoms = eqx.tree_at(
        lambda a: (
            a.ref_pos, a.ref_charge, a.ref_mask, a.ref_element, a.ref_atom_name_chars,
            a.ref_space_uid, a.atom_to_token, a.atom_to_tokatom, a.atom_mask,
        ),
        atoms,
        (
            repack(binder_template(tmpl.ref_pos), atoms.ref_pos),
            repack(binder_template(tmpl.ref_charge), atoms.ref_charge),
            repack(binder_template(tmpl.ref_mask), atoms.ref_mask),
            repack(binder_template(tmpl.ref_element), atoms.ref_element),
            repack(binder_template(tmpl.ref_atom_name_chars), atoms.ref_atom_name_chars),
            repack(token, atoms.ref_space_uid),  # binder residues own uids 0..L-1
            repack(token, atoms.atom_to_token),
            repack(binder_template(tmpl.atom_to_tokatom), atoms.atom_to_tokatom),
            repack(jnp.ones_like(token, bool), atoms.atom_mask),  # zero-filled tail: padding
        ),
    )  # fmt: skip

    def token_atoms(binder, target):
        """Per-token atom indices: binder from the new offsets, target shifted."""
        return jnp.concatenate([binder, target[L:] + shift]).astype(target.dtype)

    rep_atom = token_atoms(offset + tmpl.rep_offset[variant, ids], features.tokens.rep_atom)
    features = eqx.tree_at(
        lambda f: (f.atoms, f.tokens.rep_atom), features, (new_atoms, rep_atom)
    )
    target_backbone = pack.backbone_atom_idx[:, L:]  # 0 = no backbone (non-protein)
    backbone = jnp.concatenate(
        [
            offset[None] + jnp.arange(4)[:, None],
            jnp.where(target_backbone == 0, 0, target_backbone + shift),
        ],
        axis=1,
    )
    return Mol1FeaturePack(
        features=features,
        atom37_idx=jnp.where(
            real_atoms, repack(binder_template(tmpl.atom37_slot), pack.atom37_idx), -1
        ),
        backbone_atom_idx=backbone.astype(pack.backbone_atom_idx.dtype),
        plddt_atom_idx=token_atoms(offset + 1, pack.plddt_atom_idx),  # CA
        refresh=pack.refresh,
    )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def prepare_features(
    pack: Mol1FeaturePack,
    pssm: Float[Array, "L 20"] | None,
    templates: ResidueAtomTemplates | None,
) -> tuple[Mol1FeaturePack, Float[Array, "T 20"]]:
    """Apply the sequence (and, for a refresh pack, the repack); returns the updated
    pack and the per-token sequence in mosaic order."""
    features = pack.features
    if pssm is not None:
        features = set_binder_sequence(pssm, features)
        if pack.refresh:
            pack = refresh_binder_geometry(pack, features, pssm, templates)
        else:
            pack = eqx.tree_at(lambda p: p.features, pack, features)
    full_sequence = pack.features.tokens.restype[:, :20]
    return pack, full_sequence


def mol1_trunk(
    model: Mol1, features: Features, *, recycling_steps: int
) -> tuple[InputEmbedding, TrunkState]:
    """``recycling_steps`` trunk passes over every MSA row. Upstream draws a random MSA
    subset each pass (`Mol1.trunk`); for design that makes the gradient direction
    change with the key, so the trunk here is deterministic. Cap the MSA at
    featurization (``msa_max_depth``) instead. Only the last pass is differentiated."""
    inputs = model.embed(features)
    all_rows = jnp.ones(features.msa.msa.shape[0], bool)
    state = TrunkState(s=jnp.zeros_like(inputs.s_init), z=jnp.zeros_like(inputs.z_init))
    state = jax.lax.fori_loop(
        0,
        recycling_steps - 1,
        lambda _, st: jax.lax.stop_gradient(model.cycle(features, inputs, st, all_rows)),
        state,
    )
    return inputs, model.cycle(features, inputs, jax.lax.stop_gradient(state), all_rows)


def mol1_forward_from_trunk(
    model: Mol1,
    pack: Mol1FeaturePack,
    full_sequence: Float[Array, "T 20"],
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
        backbone_coordinates=jnp.transpose(coords[pack.backbone_atom_idx], (1, 0, 2)),
        full_sequence=full_sequence,
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
    templates: ResidueAtomTemplates | None  # required for a refresh pack
    recycling_steps: int = eqx.field(static=True, default=1)
    num_samples: int = eqx.field(static=True, default=1)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)
    name: str = "mol1"

    def __call__(self, sequence: Float[Array, "N 20"], *, key):
        pack, full_sequence = prepare_features(self.pack, sequence, self.templates)
        inputs, state = mol1_trunk(
            self.model, pack.features, recycling_steps=self.recycling_steps
        )

        def single_sample(key):
            output = mol1_forward_from_trunk(
                self.model, pack, full_sequence, inputs, state, sampler=self.sampler, key=key
            )
            return self.loss(sequence, output, key=key)

        values, auxiliary = jax.vmap(single_sample)(jax.random.split(key, self.num_samples))
        return reduce_samples(values, auxiliary, self.reduction, self.num_samples)

