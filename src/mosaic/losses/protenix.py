import copy

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PyTree
from protenix.data.constants import GAP, PRO_STD_RESIDUES, RES_ATOMS_DICT

from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX
from mosaic.sampling.core import random_rotation


def biotite_atom_to_gemmi_atom(atom):
    ga = gemmi.Atom()
    ga.pos = gemmi.Position(*atom.coord)
    ga.element = gemmi.Element(atom.element)
    ga.name = atom.atom_name
    return ga


def new_gemmi_residue(atom):
    r = gemmi.Residue()
    r.name = atom.res_name
    r.seqid = gemmi.SeqId(atom.res_id, " ")
    r.entity_type = gemmi.EntityType.Polymer
    return r


def biotite_array_to_gemmi_struct(atom_array, pred_coord=None, per_atom_plddt=None):
    if pred_coord is not None:
        atom_array = copy.deepcopy(atom_array)
        atom_array.coord = pred_coord
    structure = gemmi.Structure()
    model = gemmi.Model("0")
    chains = {}
    for atom_idx, atom in enumerate(atom_array):
        chain = chains.setdefault(atom.chain_id, {})
        residue = chain.setdefault(int(atom.res_id), new_gemmi_residue(atom))
        gemmi_atom = biotite_atom_to_gemmi_atom(atom)
        if per_atom_plddt is not None:
            gemmi_atom.b_iso = per_atom_plddt[atom_idx]
        residue.add_atom(gemmi_atom)
    for k in chains:
        chain = gemmi.Chain(k)
        chain.append_residues(list(chains[k].values()))
        model.add_chain(chain)
    structure.add_model(model)
    return structure


def _build_boltz_to_protenix_matrix():
    T = np.zeros((len(TOKENS), 32))
    for i, tok in enumerate(TOKENS):
        protenix_idx = PRO_STD_RESIDUES[
            gemmi.expand_one_letter(tok, gemmi.ResidueKind.AA)
        ]
        T[i, protenix_idx] = 1
    return T

BOLTZ_TO_PROTENIX = _build_boltz_to_protenix_matrix()


def _build_protenix_atom37_table() -> np.ndarray:
    """`tokatom_to_atom37[protenix_idx, tokatom_idx]` → atom37 slot, -1 if
    the atom isn't in atom37 (e.g. OXT) or the residue isn't a standard AA.
    """
    n_protenix = 32
    max_tokatom = 1 + max(
        max(atoms.values()) for atoms in RES_ATOMS_DICT.values() if atoms
    )
    tokatom_to_atom37 = -np.ones((n_protenix, max_tokatom), dtype=np.int32)
    for three_letter, protenix_idx in PRO_STD_RESIDUES.items():
        for atom_name, tokatom_idx in RES_ATOMS_DICT.get(three_letter, {}).items():
            if atom_name in ATOM37_INDEX:
                tokatom_to_atom37[protenix_idx, tokatom_idx] = ATOM37_INDEX[atom_name]
    return tokatom_to_atom37


_TOKATOM_TO_ATOM37 = _build_protenix_atom37_table()


def _set_binder_identity(new_sequence: Float[Array, "N 20"], features: PyTree):
    """Set the binder's (first `N` tokens) `restype`, `profile` and MSA query row
    to `new_sequence`; atom features are left as featurized.

    Protenix takes `profile` per chain, as the mean over that chain's own MSA
    rows: the rows with a residue in its columns (unpaired MSAs are merged
    row-aligned, gap-padded). Swapping the query row is then a rank-one update
    by 1 / that depth, which keeps the rest of a binder MSA's statistics; a
    binder without an MSA has depth 1, so its profile becomes the sequence.
    Exact unless the featurizer cropped the binder's MSA rows below the depth
    its profile was taken over (more than 16384 rows)."""
    binder_len = new_sequence.shape[0]
    protenix_sequence = new_sequence @ BOLTZ_TO_PROTENIX
    msa = features["msa"]
    depth = (msa[:, :binder_len] != GAP["-"]).any(-1).sum()
    old_query = features["restype"][:binder_len]  # one-hot of msa[0] at the binder positions
    return features | {
        "restype": features["restype"].at[:binder_len].set(protenix_sequence),
        "profile": features["profile"].at[:binder_len].add((protenix_sequence - old_query) / depth),
        "msa": msa.at[0, :binder_len].set(protenix_sequence.argmax(-1).astype(msa.dtype)),
    }


# ---------------------------------------------------------------------------
# Binder atom refresh
# ---------------------------------------------------------------------------
# `binder_features` featurizes a poly-Trp binder (Trp is the largest residue), so
# the binder's atom block fits any sequence without a recompile. Protenix's
# atom-level features are those of the featurized residues, so
# `refresh_binder_atoms` rebuilds every atom-axis feature (and the token
# features that index atoms) of the binder from `argmax(pssm)`: binder atoms
# are tight-packed in native order, the target atoms follow, and the freed tail
# is padding (`atom_pad_mask`, which Protenij masks throughout). The real-atom
# prefix then equals Protenix's own featurization of the designed sequence, up
# to the random per-residue rigid augmentation of `ref_pos`.

MAX_ATOMS_PER_RES = 15  # C-terminal Trp: 14 heavy atoms + OXT
# Read from per-residue templates.
_TEMPLATE_ATOM_FIELDS = (
    "ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
    "atom_to_tokatom_idx", "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
    "distogram_rep_atom_mask",
)
# Constant over a residue: copied from the placeholder residue at the same position.
_RESIDUE_ATOM_FIELDS = (
    "ref_space_uid", "is_protein", "is_ligand", "is_dna", "is_rna", "mol_id",
    "entity_mol_id", "modified_res_mask",
)
# Rebuilt from the packing (`bond_mask` is all-zero for protein-only complexes).
_DERIVED_ATOM_FIELDS = ("atom_to_token_idx", "mol_atom_index", "bond_mask", "atom_pad_mask")


class ProtenixAtomTemplates(eqx.Module):
    """Protenix atom features of each residue type (mosaic `TOKENS` order) in two
    contexts: 0 = internal (or N-terminal), 1 = C-terminal (carries OXT).
    Atom slots are in native order, zero past `n_atoms`; `ref_pos` is the
    reference conformer centred over the residue (pre-augmentation)."""

    atoms: dict  # name -> [2, 20, MAX_ATOMS_PER_RES, ...]
    n_atoms: Int[Array, "2 20"]
    frame: Int[Array, "2 20 3"]  # local N, CA, C (`frame_atom_index`)
    rep: Int[Array, "2 20"]  # local representative atom (`atom_rep_atom_idx`: CB, Gly CA)


def build_atom_templates(features: dict, ref_pos, tokens) -> ProtenixAtomTemplates:
    """Read templates off a featurization: `tokens[context, aa]` is the token
    holding residue `TOKENS[aa]` in that context; `ref_pos` is the unaugmented
    conformer (`atom_array.ref_pos`)."""
    a2t = np.asarray(features["atom_to_token_idx"])
    atoms = {
        name: np.zeros((2, 20, MAX_ATOMS_PER_RES, *np.shape(features[name])[1:]), np.asarray(features[name]).dtype)
        for name in _TEMPLATE_ATOM_FIELDS
    }
    n_atoms, frame, rep = np.zeros((2, 20), np.int32), np.zeros((2, 20, 3), np.int32), np.zeros((2, 20), np.int32)
    for c, i in np.ndindex(2, 20):
        token = tokens[c][i]
        idx = np.flatnonzero(a2t == token)
        assert len(idx) <= MAX_ATOMS_PER_RES and np.all(np.diff(idx) == 1)
        assert features["has_frame"][token] == 1
        for name in _TEMPLATE_ATOM_FIELDS:
            atoms[name][c, i, : len(idx)] = features[name][idx]
        atoms["ref_pos"][c, i, : len(idx)] = ref_pos[idx] - ref_pos[idx].mean(0)
        n_atoms[c, i] = len(idx)
        frame[c, i] = features["frame_atom_index"][token] - idx[0]
        rep[c, i] = features["atom_rep_atom_idx"][token] - idx[0]
    return ProtenixAtomTemplates(
        atoms={k: jnp.asarray(v) for k, v in atoms.items()},
        n_atoms=jnp.asarray(n_atoms),
        frame=jnp.asarray(frame),
        rep=jnp.asarray(rep),
    )


class ProtenixDesignFeatures(eqx.Module):
    """A design pack from `Protenix.binder_features`: placeholder features plus
    what `set_binder_sequence` needs to rebuild the binder's atoms for any
    sequence of `binder_length`."""

    features: dict
    atom_templates: ProtenixAtomTemplates
    binder_length: int = eqx.field(static=True)
    binder_atom_alloc: int = eqx.field(static=True)


def binder_design_features(
    features: dict, binder_length: int, templates: ProtenixAtomTemplates
) -> ProtenixDesignFeatures:
    """Wrap a featurization whose first chain is a `binder_length` placeholder
    binder with enough atoms for any sequence (poly-Trp)."""
    from protenix.atom_padding import ATOM_FEATURE_AXES

    a2t = np.asarray(features["atom_to_token_idx"])
    asym = np.asarray(features["asym_id"])
    alloc = int((a2t < binder_length).sum())
    assert (asym == asym[0]).sum() == binder_length, "the binder must be the first chain"
    assert alloc >= (MAX_ATOMS_PER_RES - 1) * binder_length + 1, "placeholder binder too small (use poly-Trp)"
    assert not np.any(features["bond_mask"]), "bond_mask must be empty (protein-only complex)"
    handled = {*_TEMPLATE_ATOM_FIELDS, *_RESIDUE_ATOM_FIELDS, *_DERIVED_ATOM_FIELDS}
    assert not (unhandled := (ATOM_FEATURE_AXES.keys() & features.keys()) - handled), unhandled
    features = features | {"atom_pad_mask": np.ones(len(a2t), dtype=bool)}
    return ProtenixDesignFeatures(
        features=jax.tree.map(jnp.asarray, features),
        atom_templates=templates,
        binder_length=binder_length,
        binder_atom_alloc=alloc,
    )


def refresh_binder_atoms(pack: ProtenixDesignFeatures, pssm: Float[Array, "L 20"], key: jax.Array) -> dict:
    """Rebuild the binder's atom-axis features (and `frame_atom_index`,
    `atom_rep_atom_idx`) for `argmax(pssm)`.

    `ref_pos` gets the featurizer's augmentation: per residue, a random
    translation in [-1, 1]^3 and then a uniform random rotation (`key`)."""
    features, templates = pack.features, pack.atom_templates
    L, alloc = pack.binder_length, pack.binder_atom_alloc
    n_atoms = features["atom_to_token_idx"].shape[0]
    M = MAX_ATOMS_PER_RES
    ids = pssm.argmax(-1)
    context = (jnp.arange(L) == L - 1).astype(jnp.int32)

    def gather(table):  # [2, 20, ...] -> [L, ...]
        return table[context, ids]

    k = gather(templates.n_atoms)
    start = jnp.cumsum(k) - k
    shift = k.sum() - alloc  # target atoms move left by the binder's unused slots
    slot = jnp.arange(M)
    dest = jnp.where(slot < k[:, None], start[:, None] + slot, n_atoms).reshape(-1)  # n_atoms: dropped
    target = jnp.arange(alloc, n_atoms) + shift

    def pack(name, binder):
        old = features[name]
        new = jnp.zeros_like(old).at[dest].set(binder.reshape(L * M, *old.shape[1:]).astype(old.dtype), mode="drop")
        return new.at[target].set(old[alloc:])

    kt, kr = jax.random.split(key)
    t = jax.random.uniform(kt, (L, 1, 3), minval=-1.0, maxval=1.0)
    ref_pos = jnp.einsum("lac,ldc->lad", gather(templates.atoms["ref_pos"]) + t, jax.vmap(random_rotation)(jax.random.split(kr, L)))
    first = jnp.searchsorted(features["atom_to_token_idx"][:alloc], jnp.arange(L))  # placeholder residue starts
    binder = {name: gather(table) for name, table in templates.atoms.items()} | {
        name: jnp.broadcast_to(features[name][first][:, None], (L, M)) for name in _RESIDUE_ATOM_FIELDS if name in features
    } | {
        "ref_pos": ref_pos,
        "atom_to_token_idx": jnp.broadcast_to(jnp.arange(L)[:, None], (L, M)),
        "mol_atom_index": start[:, None] + slot,
    }
    frame, rep = features["frame_atom_index"], features["atom_rep_atom_idx"]
    return features | {name: pack(name, values) for name, values in binder.items()} | {
        "atom_pad_mask": jnp.arange(n_atoms) < n_atoms + shift,
        "frame_atom_index": jnp.concatenate([start[:, None] + gather(templates.frame), frame[L:] + shift]).astype(frame.dtype),
        "atom_rep_atom_idx": jnp.concatenate([start + gather(templates.rep), rep[L:] + shift]).astype(rep.dtype),
    }


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree | ProtenixDesignFeatures,
    key: jax.Array,
):
    """Set the binder (the first `N` tokens) to `new_sequence`.

    `restype`, `profile` and the MSA query row are set as Protenix would
    featurize the sequence (`_set_binder_identity`). A `ProtenixDesignFeatures`
    pack (from `binder_features`) also gets the binder's atoms and frames
    rebuilt from `argmax(PSSM)` (`key` drives the `ref_pos` augmentation). A
    plain feature dict (e.g. `target_only_features`) keeps the atoms of the
    residues it was featurized with."""
    if not isinstance(features, ProtenixDesignFeatures):
        return _set_binder_identity(new_sequence, features)
    assert new_sequence.shape == (features.binder_length, 20), new_sequence.shape
    # A separate stream for the `ref_pos` augmentation, leaving `key` to the fold.
    feat = refresh_binder_atoms(features, new_sequence, jax.random.fold_in(key, 0x70A7))
    return _set_binder_identity(new_sequence, feat)


PROTENIX_DISTOGRAM_BINS = np.linspace(start=2.3125, stop=21.6875, num=64)
