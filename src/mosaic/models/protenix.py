"""Protenix backend running on jfold.

The forward pass uses jfold's converted Protenix weights from `jfold.pretrained`. Inputs
are featurized with `jfold.featurize`, which builds the same features as upstream Protenix
apart from the random placement of reference conformers. It supports protein chains of the
20 amino acids and X, with templates of the same length as their chain, and MSAs for at
most one distinct sequence.

Other inputs, such as nucleic acids, ligands, modified residues or paired MSAs, can be
featurized from a Protenix input JSON with `mosaic.models.upstream.protenix.features`,
which needs the `protenix` package. `Protenix.design_pack` turns either kind of features
into binder design features.
"""

from collections.abc import Callable

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jfold import pretrained
from jfold.featurize.system import THREE_LETTER
from jfold.featurize.views import protenix as view
from jfold.model import StructureModel
from jfold.models import protenix as jprotenix
from jfold.sampling import random_rotation
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.models.jfold_featurize import (
    BinderAtomSitesWriter,
    check_chains,
    jfold_system,
    residue_type_features,
)
from mosaic.structure_prediction import (
    ATOM37_INDEX,
    Atom37,
    DiffusionStructurePredictionModel,
    StructureModelOutput,
    TargetChain,
    TokenInfo,
    scatter_atom37,
)

PROTENIX_DISTOGRAM_BINS = np.linspace(start=2.3125, stop=21.6875, num=64)
# Upstream's number of trunk passes at inference (`N_cycle` in `configs_model_type.py`).
_INFERENCE_RECYCLING_STEPS = {
    "protenix_mini": 4,
    "protenix_tiny": 4,
    "protenix_base": 10,
    "protenix_base_20250630": 10,
    "protenix_v2": 10,
}
# Hint added to errors for inputs that jfold cannot featurize.
UPSTREAM = (
    "mosaic.models.upstream.protenix.features featurizes Protenix's own input JSON "
    "(install mosaic[all])"
)
# Row `i` is the one-hot Protenix residue class of `TOKENS[i]`.
BOLTZ_TO_PROTENIX = np.eye(view.NUM_CLASSES)[[view.RESTYPE[aa] for aa in TOKENS]]


def _tokatom_to_atom37() -> np.ndarray:
    """Build a table mapping residue class and atom position in the residue to atom37.

    Entries are -1 for atoms outside atom37, such as `OXT`, and for non-residue classes."""
    table = view.residues("protenix")
    atoms = {i: table[THREE_LETTER[aa]].atoms for aa, i in view.RESTYPE.items()}
    out = -np.ones((view.NUM_CLASSES, max(map(len, atoms.values()))), np.int32)
    for i, names in atoms.items():
        for j, name in enumerate(names):
            out[i, j] = ATOM37_INDEX.get(name, -1)
    return out


TOKATOM_TO_ATOM37 = _tokatom_to_atom37()


def check_trp_placeholder(features: dict[str, Array], binder_length: int) -> None:
    """Raise `ValueError` unless the first chain of `features` is a poly-Trp placeholder
    of `binder_length` residues."""
    asym = np.asarray(features["asym_id"])
    first_chain = int((asym == asym[0]).sum())
    if first_chain != binder_length:
        raise ValueError(
            f"the binder must be the first chain, of {binder_length} residues; the first "
            f"chain has {first_chain} tokens"
        )
    restype = np.asarray(features["restype"][:binder_length]).argmax(-1)
    if not np.all(restype == view.RESTYPE["W"]):
        raise ValueError(
            "the binder (the first chain) must be a Trp placeholder ('W' * binder_length), "
            "which has the atoms of any sequence"
        )


def check_protenix_chains(chains: list[TargetChain]) -> None:
    """Run `check_chains` and also require MSAs for at most one distinct sequence.

    Upstream pairs the MSAs of several sequences, which jfold does not support."""
    check_chains(chains, "Protenix", templates=True, upstream=UPSTREAM)
    if len({c.sequence for c in chains if c.use_msa}) > 1:
        raise NotImplementedError(
            "Protenix on jfold does not pair MSAs: give at most one sequence an MSA "
            f"(use_msa); {UPSTREAM}"
        )


def featurize(
    chains: list[TargetChain], *, key: PRNGKeyArray
) -> tuple[dict[str, Array], BinderAtomSitesWriter]:
    """Featurize `chains` with jfold and return the features with a structure writer.

    `key` randomly rotates and translates each residue's reference conformer."""
    check_protenix_chains(chains)
    system = jfold_system(chains)
    features = view.features(system, key=key)
    writer = BinderAtomSitesWriter(system, view.atom_sites)
    return jax.tree.map(jnp.asarray, features), writer


def set_binder_identity(
    sequence: Float[Array, "N 20"], features: dict[str, Array]
) -> dict[str, Array]:
    """Set the `restype`, `profile` and MSA query row of the binder, the first `N` tokens.

    Atom features are not changed. Protenix computes `profile` per chain as the mean over
    the MSA rows that are not gaps in that chain's columns. Replacing the query row
    therefore changes the profile by the difference divided by that row count. A binder
    without an MSA has one row, so its profile becomes the sequence. The update is exact
    unless the featurizer cropped the binder's MSA, which happens above 16384 rows."""
    binder_len = sequence.shape[0]
    protenix_sequence = sequence @ BOLTZ_TO_PROTENIX
    msa = features["msa"]
    depth = (msa[:, :binder_len] != view.GAP).any(-1).sum()
    old_query = features["restype"][:binder_len]
    return features | {
        "restype": features["restype"].at[:binder_len].set(protenix_sequence),
        "profile": features["profile"]
        .at[:binder_len]
        .add((protenix_sequence - old_query) / depth),
        "msa": msa.at[0, :binder_len].set(
            protenix_sequence.argmax(-1).astype(msa.dtype)
        ),
    }


# For binder design, the binder is featurized as poly-Trp. Trp is the largest residue, so
# the binder's atoms fit any sequence without recompiling. `refresh_binder_atoms` rebuilds
# the binder's atom features from `argmax(pssm)`, packing them at the front, followed by
# the target atoms and then padding.

MAX_ATOMS_PER_RES = 15  # C-terminal Trp: 14 heavy atoms + OXT
# Atom features taken from the per-residue templates.
_TEMPLATE_ATOM_FIELDS = (
    "ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
    "atom_to_tokatom_idx", "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
    "distogram_rep_atom_mask",
)  # fmt: skip
# Atom features that are constant within a residue, copied from the placeholder residue.
_RESIDUE_ATOM_FIELDS = (
    "ref_space_uid", "is_protein", "is_ligand", "is_dna", "is_rna", "mol_id",
    "entity_mol_id", "modified_res_mask",
)  # fmt: skip
# Atom features computed from the packing. `bond_mask` is all zero for proteins.
_DERIVED_ATOM_FIELDS = (
    "atom_to_token_idx", "mol_atom_index", "bond_mask", "atom_pad_mask",
)  # fmt: skip


class ProtenixAtomTemplates(eqx.Module):
    """Protenix atom features of each amino acid, used to rebuild binder residues.

    Arrays have leading axes `[2, 20]`. The first axis is 0 for an interior or N-terminal
    residue and 1 for a C-terminal residue with `OXT`. The second is the amino acid in
    `TOKENS` order. Atom slots past `n_atoms` are zero, and `ref_pos` is the centred
    reference conformer without random placement."""

    atoms: dict  # name -> [2, 20, MAX_ATOMS_PER_RES, ...]
    n_atoms: Int[Array, "2 20"]
    frame: Int[Array, "2 20 3"]  # local N, CA, C (`frame_atom_index`)
    rep: Int[Array, "2 20"]  # local representative atom index (`atom_rep_atom_idx`)


def atom_templates() -> ProtenixAtomTemplates:
    """Build the atom templates by featurizing each amino acid in `G{}G` and `G{}`."""
    contexts = [residue_type_features(view.features, c) for c in ("G{}G", "G{}")]
    sample = contexts[0][TOKENS[0]]
    atoms = {
        name: np.zeros(
            (2, 20, MAX_ATOMS_PER_RES, *sample[name].shape[1:]), sample[name].dtype
        )
        for name in _TEMPLATE_ATOM_FIELDS
    }
    n_atoms = np.zeros((2, 20), np.int32)
    frame = np.zeros((2, 20, 3), np.int32)
    rep = np.zeros((2, 20), np.int32)
    for c, i in np.ndindex(2, 20):
        features = contexts[c][TOKENS[i]]
        idx = np.flatnonzero(features["atom_to_token_idx"] == 1)
        for name in _TEMPLATE_ATOM_FIELDS:
            atoms[name][c, i, : len(idx)] = features[name][idx]
        n_atoms[c, i] = len(idx)
        frame[c, i] = features["frame_atom_index"][1] - idx[0]
        rep[c, i] = features["atom_rep_atom_idx"][1] - idx[0]
    return ProtenixAtomTemplates(
        atoms={k: jnp.asarray(v) for k, v in atoms.items()},
        n_atoms=jnp.asarray(n_atoms),
        frame=jnp.asarray(frame),
        rep=jnp.asarray(rep),
    )


class ProtenixDesignFeatures(eqx.Module):
    """Features for binder design, whose binder is a poly-Trp placeholder.

    `atom_templates` lets `set_binder_sequence` rebuild the binder's atoms for any
    sequence of length `binder_length`."""

    features: dict
    atom_templates: ProtenixAtomTemplates
    binder_length: int = eqx.field(static=True)
    binder_atom_alloc: int = eqx.field(static=True)


def binder_design_features(
    features: dict[str, Array], binder_length: int, templates: ProtenixAtomTemplates
) -> ProtenixDesignFeatures:
    """Wrap features whose first chain is a poly-Trp binder of length `binder_length`."""
    a2t = np.asarray(features["atom_to_token_idx"])
    alloc = int((a2t < binder_length).sum())
    assert not np.any(features["bond_mask"]), "bond_mask must be empty (protein only)"
    atom_fields = {k for k, v in features.items() if v.shape[:1] == a2t.shape}
    handled = {*_TEMPLATE_ATOM_FIELDS, *_RESIDUE_ATOM_FIELDS, *_DERIVED_ATOM_FIELDS}
    assert not (unhandled := atom_fields - handled), unhandled
    return ProtenixDesignFeatures(
        features=features | {"atom_pad_mask": jnp.ones(len(a2t), bool)},
        atom_templates=templates,
        binder_length=binder_length,
        binder_atom_alloc=alloc,
    )


def refresh_binder_atoms(
    pack: ProtenixDesignFeatures, pssm: Float[Array, "L 20"], key: PRNGKeyArray
) -> dict[str, Array]:
    """Rebuild the binder's atom features, frames and representative atoms for
    `argmax(pssm)`.

    As in the featurizer, each residue's `ref_pos` is translated by a uniform random
    offset in [-1, 1]^3 and then randomly rotated."""
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
    shift = k.sum() - alloc  # target atoms move left over the binder's unused slots
    slot = jnp.arange(M)
    # Unused binder slots point to `n_atoms`, so their writes are dropped.
    dest = jnp.where(slot < k[:, None], start[:, None] + slot, n_atoms).reshape(-1)
    target = jnp.arange(alloc, n_atoms) + shift

    def pack_atoms(name, binder):
        old = features[name]
        binder = binder.reshape(L * M, *old.shape[1:]).astype(old.dtype)
        new = jnp.zeros_like(old).at[dest].set(binder, mode="drop")
        return new.at[target].set(old[alloc:])

    kt, kr = jax.random.split(key)
    t = jax.random.uniform(kt, (L, 1, 3), minval=-1.0, maxval=1.0)
    rotations = jax.vmap(random_rotation)(jax.random.split(kr, L))
    ref_pos = jnp.einsum(
        "lac,ldc->lad", gather(templates.atoms["ref_pos"]) + t, rotations
    )
    first = jnp.searchsorted(features["atom_to_token_idx"][:alloc], jnp.arange(L))
    binder = (
        {name: gather(table) for name, table in templates.atoms.items()}
        | {
            name: jnp.broadcast_to(features[name][first][:, None], (L, M))
            for name in _RESIDUE_ATOM_FIELDS
        }
        | {
            "ref_pos": ref_pos,
            "atom_to_token_idx": jnp.broadcast_to(jnp.arange(L)[:, None], (L, M)),
            "mol_atom_index": start[:, None] + slot,
        }
    )
    frame, rep = features["frame_atom_index"], features["atom_rep_atom_idx"]
    # Shift the target's frame atoms with its atoms, keeping -1 for missing frame atoms.
    target_frame = jnp.where(frame[L:] >= 0, frame[L:] + shift, frame[L:])
    return (
        features
        | {name: pack_atoms(name, values) for name, values in binder.items()}
        | {
            "atom_pad_mask": jnp.arange(n_atoms) < n_atoms + shift,
            "frame_atom_index": jnp.concatenate(
                [start[:, None] + gather(templates.frame), target_frame]
            ).astype(frame.dtype),
            "atom_rep_atom_idx": jnp.concatenate(
                [start + gather(templates.rep), rep[L:] + shift]
            ).astype(rep.dtype),
        }
    )


def set_binder_sequence(
    sequence: Float[Array, "N 20"],
    features: dict[str, Array] | ProtenixDesignFeatures,
    key: PRNGKeyArray,
) -> dict[str, Array] | ProtenixDesignFeatures:
    """Set the binder, the first `N` tokens, to `sequence`.

    For binder design features, also rebuild the binder's atoms and frames from
    `argmax(sequence)`, using `key` to place the reference conformers. A plain feature dict
    keeps its original atoms."""
    if not isinstance(features, ProtenixDesignFeatures):
        return set_binder_identity(sequence, features)
    assert sequence.shape == (features.binder_length, 20), sequence.shape
    # Derive a separate key for the conformers so that `key` is not reused.
    refreshed = refresh_binder_atoms(
        features, sequence, jax.random.fold_in(key, 0x70A7)
    )
    return eqx.tree_at(
        lambda f: f.features, features, set_binder_identity(sequence, refreshed)
    )


def _feat(features: dict[str, Array] | ProtenixDesignFeatures) -> dict[str, Array]:
    if isinstance(features, ProtenixDesignFeatures):
        return features.features
    return features


# Converts predicted coordinates, and for binder design the binder PSSM, to a
# `gemmi.Structure`.
Writer = Callable[..., gemmi.Structure]


class Protenix(DiffusionStructurePredictionModel[dict | ProtenixDesignFeatures]):
    """Protenix with upstream's recycling counts.

    The number of sampling steps comes from the converted model. Recycling steps count
    trunk passes, like upstream's `N_cycle`."""

    model: StructureModel
    inference_recycling_steps: int = eqx.field(static=True)
    design_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    distogram_bins = PROTENIX_DISTOGRAM_BINS

    def __init__(
        self,
        model: StructureModel,
        *,
        inference_recycling_steps: int,
        design_recycling_steps: int = 1,
        backprop_all_passes: bool = False,
    ):
        """Wrap a converted Protenix model. Use `Protenix.mini()` and the other class
        methods to load released checkpoints."""
        self.model = model
        self.inference_recycling_steps = inference_recycling_steps
        self.design_recycling_steps = design_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    @classmethod
    def _release(cls, name: str, **settings) -> "Protenix":
        """Load the `jfold.pretrained` checkpoint `name`, downloading it on first use.

        Inference uses upstream's number of trunk passes unless `settings` override it."""
        settings.setdefault("inference_recycling_steps", _INFERENCE_RECYCLING_STEPS[name])
        return cls(pretrained.load(name, cache=cache_dir("jfold")), **settings)

    @classmethod
    def mini(cls, **settings) -> "Protenix":
        """Load Protenix-Mini. `settings` go to `__init__`."""
        return cls._release("protenix_mini", **settings)

    @classmethod
    def tiny(cls, **settings) -> "Protenix":
        """Load Protenix-Tiny."""
        return cls._release("protenix_tiny", **settings)

    @classmethod
    def base(cls, **settings) -> "Protenix":
        """Load Protenix base v1.0.0."""
        return cls._release("protenix_base", **settings)

    @classmethod
    def base_20250630(cls, **settings) -> "Protenix":
        """Load Protenix base with the 2025-06-30 data cutoff."""
        return cls._release("protenix_base_20250630", **settings)

    @classmethod
    def v2(cls, **settings) -> "Protenix":
        """Load Protenix v2."""
        return cls._release("protenix_v2", **settings)

    def target_only_features(
        self, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[dict[str, Array], Writer]:
        """Featurize `chains` with jfold. `key` randomly places the reference conformers."""
        return featurize(chains, key=key)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[ProtenixDesignFeatures, Writer]:
        """Featurize a poly-Trp binder followed by `chains` for binder design."""
        binder = TargetChain(sequence="W" * binder_length, use_msa=False)
        features, writer = featurize([binder, *chains], key=key)
        return self.design_pack(features, writer, binder_length)

    @staticmethod
    def design_pack(
        features: dict[str, Array], writer: Writer, binder_length: int
    ) -> tuple[ProtenixDesignFeatures, Writer]:
        """Turn features whose first chain is `"W" * binder_length` into design features.

        The features can come from jfold or from `mosaic.models.upstream.protenix.features`.
        Setting a sequence on the result rebuilds the binder's atoms. Raise `ValueError` if
        the first chain is not such a poly-Trp binder."""
        check_trp_placeholder(features, binder_length)
        return binder_design_features(features, binder_length, atom_templates()), writer

    def set_sequence(
        self, features, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ):
        """Set the binder's sequence, also rebuilding its atoms for design features."""
        return set_binder_sequence(sequence, features, key)

    def jfold_features(self, features) -> JFeatures:
        return jprotenix.features(_feat(features))

    def tokens(self, features) -> TokenInfo:
        feats = _feat(features)
        return TokenInfo(
            sequence=feats["restype"] @ BOLTZ_TO_PROTENIX.T,
            asym_id=feats["asym_id"],
            residue_index=feats["residue_index"],
            mask=jnp.ones(feats["restype"].shape[0], bool),
        )

    def atom37(self, features, coords: Float[Array, "A 3"]) -> Atom37:
        feats = _feat(features)
        a2t = feats["atom_to_token_idx"]
        restype = feats["restype"].argmax(-1)[a2t]
        atom37_idx = jnp.array(TOKATOM_TO_ATOM37)[restype, feats["atom_to_tokatom_idx"]]
        # Only amino acids have atom37 slots.
        real = (feats["ref_mask"] > 0.5) & (feats["is_protein"] > 0)
        atom37_idx = jnp.where(real, atom37_idx, jnp.int32(-1))
        coords37, mask37 = scatter_atom37(
            coords, a2t, atom37_idx, feats["restype"].shape[0]
        )
        return Atom37(coords=coords37, mask=mask37)

    def to_structure(
        self, features, writer, output: StructureModelOutput, sequence
    ) -> gemmi.Structure:
        """Build a `gemmi.Structure`, writing a design's binder as `argmax(sequence)`."""
        if isinstance(features, ProtenixDesignFeatures):
            return writer(output.structure_coordinates, sequence)
        return writer(output.structure_coordinates)
