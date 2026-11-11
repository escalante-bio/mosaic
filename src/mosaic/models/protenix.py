"""Protenix on jfold: jfold's converted Protenix (`jfold.pretrained`) for the forward
pass, features by `jfold.featurize` (torch-free: the dict upstream's
`protenix.data.template.featurize` builds from the same MSAs, up to the random pose of
each residue's reference conformer, without `constraint_feature` and `resolution`, which
jfold's model does not read).

Features are that dict (as JAX arrays), or for design a `ProtenixDesignFeatures` pack,
which rebuilds the binder's atoms for each sequence. jfold's view takes protein chains of
the 20 amino acids and X, with templates of those; a template must have its chain's
length (residue `i` standing for the chain's residue `i`), or featurization raises, as
upstream's does. MSAs are the shared unpaired ColabFold ones (`mosaic.msa.fetch_a3m`), for
at most one sequence: upstream pairs the MSAs of several, which jfold's view does not.
Everything else (DNA and RNA, ligands and ions, modified residues, paired MSAs) is
featurized from Protenix's own input JSON by `mosaic.models.upstream.protenix.features`
(which needs the `protenix` package), into the same dict; `Protenix.design_pack` makes
those features a design pack.

Featurizing draws each reference conformer's pose at random, as upstream does (from
numpy's global RNG), so `target_only_features` and `binder_features` take a `key`.
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
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.models.jfold_featurize import (
    BinderAtomSitesWriter,
    check_chains,
    jfold_system,
    residue_type_features,
)
from mosaic.sampling.core import random_rotation
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
)

PROTENIX_DISTOGRAM_BINS = np.linspace(start=2.3125, stop=21.6875, num=64)
# Upstream's trunk passes at inference (`N_cycle` in `configs_model_type.py`).
_INFERENCE_RECYCLING_STEPS = {
    "protenix_mini": 4,
    "protenix_tiny": 4,
    "protenix_base": 10,
    "protenix_base_20250630": 10,
    "protenix_v2": 10,
}
# How the jfold view's refusals point to upstream's featurization.
UPSTREAM = (
    "mosaic.models.upstream.protenix.features featurizes Protenix's own input JSON "
    "(install mosaic[all])"
)
# `BOLTZ_TO_PROTENIX[i]`: the one-hot Protenix residue class of `TOKENS[i]`.
BOLTZ_TO_PROTENIX = np.eye(view.NUM_CLASSES)[[view.RESTYPE[aa] for aa in TOKENS]]


def _tokatom_to_atom37() -> np.ndarray:
    """`[residue class, position in the residue's CCD atoms]` → atom37 slot; -1 where the
    atom is not in atom37 (e.g. `OXT`) or the class is not a residue."""
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
    """`check_chains`, and at most one sequence with an MSA: upstream pairs the MSAs of
    several."""
    check_chains(chains, "Protenix", templates=True, upstream=UPSTREAM)
    if len({c.sequence for c in chains if c.use_msa}) > 1:
        raise NotImplementedError(
            "Protenix on jfold does not pair MSAs: give at most one sequence an MSA "
            f"(use_msa); {UPSTREAM}"
        )


def featurize(
    chains: list[TargetChain], *, key: PRNGKeyArray
) -> tuple[dict[str, Array], BinderAtomSitesWriter]:
    """`chains` featurized by jfold's view (`key` poses the reference conformers), and
    their writer."""
    check_protenix_chains(chains)
    system = jfold_system(chains)
    features = view.features(system, key=key)
    writer = BinderAtomSitesWriter(system, view.atom_sites)
    return jax.tree.map(jnp.asarray, features), writer


def set_binder_identity(
    sequence: Float[Array, "N 20"], features: dict[str, Array]
) -> dict[str, Array]:
    """Set the binder's (first `N` tokens) `restype`, `profile` and MSA query row to
    `sequence`; atom features are left as featurized.

    Protenix takes `profile` per chain, as the mean over that chain's own MSA rows: the
    rows with a residue in its columns (unpaired MSAs are merged row-aligned,
    gap-padded). Swapping the query row is then a rank-one update by 1 / that depth,
    which keeps the rest of a binder MSA's statistics; a binder without an MSA has depth
    1, so its profile becomes the sequence. Exact unless the featurizer cropped the
    binder's MSA rows below the depth its profile was taken over (more than 16384
    rows)."""
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


# The design pack (`Protenix.design_pack`): the binder is featurized as a poly-Trp
# placeholder (Trp is the largest residue), so the binder's atom block fits any sequence
# without a recompile. `refresh_binder_atoms` rebuilds every atom-axis feature (and the
# token features that index atoms) of the binder from `argmax(pssm)`: binder atoms are
# tight-packed in native order, the target atoms follow, and the freed tail is padding
# (`atom_pad_mask`, which the model masks throughout). The real-atom prefix then equals
# the featurization of the designed sequence, up to the random per-residue pose of
# `ref_pos`.

MAX_ATOMS_PER_RES = 15  # C-terminal Trp: 14 heavy atoms + OXT
# Read from per-residue templates.
_TEMPLATE_ATOM_FIELDS = (
    "ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
    "atom_to_tokatom_idx", "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
    "distogram_rep_atom_mask",
)  # fmt: skip
# Constant over a residue: copied from the placeholder residue at the same position.
_RESIDUE_ATOM_FIELDS = (
    "ref_space_uid", "is_protein", "is_ligand", "is_dna", "is_rna", "mol_id",
    "entity_mol_id", "modified_res_mask",
)  # fmt: skip
# Rebuilt from the packing (`bond_mask` is all-zero for protein-only complexes).
_DERIVED_ATOM_FIELDS = (
    "atom_to_token_idx", "mol_atom_index", "bond_mask", "atom_pad_mask",
)  # fmt: skip


class ProtenixAtomTemplates(eqx.Module):
    """Protenix atom features of each residue type (`TOKENS` order) in two contexts:
    0 = internal (or N-terminal), 1 = C-terminal (carries OXT). Atom slots are in native
    order, zero past `n_atoms`; `ref_pos` is the reference conformer centred over the
    residue (unposed)."""

    atoms: dict  # name -> [2, 20, MAX_ATOMS_PER_RES, ...]
    n_atoms: Int[Array, "2 20"]
    frame: Int[Array, "2 20 3"]  # local N, CA, C (`frame_atom_index`)
    rep: Int[Array, "2 20"]  # local representative atom (`atom_rep_atom_idx`)


def atom_templates() -> ProtenixAtomTemplates:
    """Per-residue atom templates: the middle residue of `G{aa}G` (internal) and the last
    of `G{aa}` (C-terminal), as the view featurizes them unposed."""
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
    """A design pack (`Protenix.design_pack`, `Protenix.binder_features`): the
    placeholder's features plus what `set_binder_sequence` needs to rebuild the binder's
    atoms for any sequence of `binder_length`."""

    features: dict
    atom_templates: ProtenixAtomTemplates
    binder_length: int = eqx.field(static=True)
    binder_atom_alloc: int = eqx.field(static=True)


def binder_design_features(
    features: dict[str, Array], binder_length: int, templates: ProtenixAtomTemplates
) -> ProtenixDesignFeatures:
    """Wrap features whose first chain is a poly-Trp placeholder binder of
    `binder_length` (checked by `Protenix.design_pack`)."""
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
    """Rebuild the binder's atom-axis features (and `frame_atom_index`,
    `atom_rep_atom_idx`) for `argmax(pssm)`.

    `ref_pos` gets the featurizer's pose: per residue, a random translation in
    [-1, 1]^3 and then a uniform random rotation (`key`)."""
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
    # Binder atom slots past a residue's atoms go to `n_atoms`: dropped.
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
    # Where each placeholder residue's atoms start.
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
    # The target's frame atoms move with its atoms; -1 (no such atom, e.g. an ion's) stays.
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
    """Set the binder (the first `N` tokens) to `sequence`: `restype`, `profile` and the
    MSA query row as Protenix would featurize it (`set_binder_identity`). A design pack
    also gets the binder's atoms and frames rebuilt from `argmax(sequence)` (`key` poses
    their reference conformers); a plain dict keeps the atoms it was featurized with."""
    if not isinstance(features, ProtenixDesignFeatures):
        return set_binder_identity(sequence, features)
    assert sequence.shape == (features.binder_length, 20), sequence.shape
    # A separate stream for the conformer pose, leaving `key` to the fold.
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


# Writes a prediction's coordinates (with a design pack's, the binder's PSSM too) as a
# `gemmi.Structure`: `BinderAtomSitesWriter`, or upstream's `AtomArrayWriter`.
Writer = Callable[..., gemmi.Structure]


class Protenix(DiffusionStructurePredictionModel[dict | ProtenixDesignFeatures]):
    """Protenix on jfold, with upstream's recycling counts; the upstream sampler's step
    count is the converted model's (`model.sampling.num_steps`). `recycling_steps` counts
    trunk passes (upstream's `N_cycle`). The two featurizations: see the module
    docstring."""

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
        """`model`: a converted Protenix. `Protenix.mini()`, `.tiny()`, `.base()`,
        `.base_20250630()` and `.v2()` load the releases."""
        self.model = model
        self.inference_recycling_steps = inference_recycling_steps
        self.design_recycling_steps = design_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    @classmethod
    def _release(cls, name: str, **settings) -> "Protenix":
        """The `jfold.pretrained` release `name` (from mosaic's `jfold` cache, downloaded
        on first use), with upstream's trunk passes at inference unless `settings`
        (`__init__`'s) say otherwise."""
        settings.setdefault("inference_recycling_steps", _INFERENCE_RECYCLING_STEPS[name])
        return cls(pretrained.load(name, cache=cache_dir("jfold")), **settings)

    @classmethod
    def mini(cls, **settings) -> "Protenix":
        """Protenix-Mini. `settings`: `__init__`'s."""
        return cls._release("protenix_mini", **settings)

    @classmethod
    def tiny(cls, **settings) -> "Protenix":
        """Protenix-Tiny. `settings`: `__init__`'s."""
        return cls._release("protenix_tiny", **settings)

    @classmethod
    def base(cls, **settings) -> "Protenix":
        """Protenix base (v1.0.0). `settings`: `__init__`'s."""
        return cls._release("protenix_base", **settings)

    @classmethod
    def base_20250630(cls, **settings) -> "Protenix":
        """Protenix base with the 2025-06-30 data cutoff. `settings`: `__init__`'s."""
        return cls._release("protenix_base_20250630", **settings)

    @classmethod
    def v2(cls, **settings) -> "Protenix":
        """Protenix v2. `settings`: `__init__`'s."""
        return cls._release("protenix_v2", **settings)

    def target_only_features(
        self, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[dict[str, Array], Writer]:
        """`chains` featurized by jfold's view (`featurize`); `key` poses the reference
        conformers."""
        return featurize(chains, key=key)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[ProtenixDesignFeatures, Writer]:
        """A poly-Trp binder followed by `chains`, featurized by jfold's view, as a design
        pack (`design_pack`)."""
        binder = TargetChain(sequence="W" * binder_length, use_msa=False)
        features, writer = featurize([binder, *chains], key=key)
        return self.design_pack(features, writer, binder_length)

    @staticmethod
    def design_pack(
        features: dict[str, Array], writer: Writer, binder_length: int
    ) -> tuple[ProtenixDesignFeatures, Writer]:
        """Features whose first chain is a poly-Trp placeholder binder
        (`"W" * binder_length`), from jfold's view or
        `mosaic.models.upstream.protenix.features`, as a `ProtenixDesignFeatures` pack: a
        sequence set on it rebuilds the binder's atoms. Raises `ValueError` for another
        first chain. The writer, given the binder's PSSM, writes the binder as its
        argmax."""
        check_trp_placeholder(features, binder_length)
        return binder_design_features(features, binder_length, atom_templates()), writer

    def set_sequence(
        self, features, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ):
        """Set the binder's sequence channels; a design pack also rebuilds the binder's
        atoms from `argmax(sequence)`."""
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
        # Amino acids only (ligand atoms are tokens of their own).
        real = (feats["ref_mask"] > 0.5) & (feats["is_protein"] > 0)
        atom37_idx = jnp.where(real, atom37_idx, jnp.int32(-1))
        coords37, mask37 = scatter_atom37(
            coords, a2t, atom37_idx, feats["restype"].shape[0]
        )
        return Atom37(coords=coords37, mask=mask37)

    def to_structure(
        self, features, writer, output: StructureModelOutput, sequence
    ) -> gemmi.Structure:
        """With `writer`; a design pack's binder as `sequence`'s argmax."""
        if isinstance(features, ProtenixDesignFeatures):
            return writer(output.structure_coordinates, sequence)
        return writer(output.structure_coordinates)
