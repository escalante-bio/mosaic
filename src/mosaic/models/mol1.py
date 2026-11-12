"""Mol-1 (`mol-1-base`) on jfold: featurized by `jfold.featurize` (torch-free; the dict
Mol-1's own featurizer builds from the same MSAs, up to the random pose of each residue's
reference conformer), with jfold's converted Mol-1 (`jfold.pretrained`) for the forward
pass.

The view takes protein chains of the 20 amino acids and X, each with its unpaired MSA (as
upstream's, for any number of chains) and optionally a template chain. As upstream's JSON
templates, a template must have its chain's length, residue `i` standing for the chain's
residue `i`, or featurization raises. Anything else (DNA/RNA, other residues) raises
`NotImplementedError`; `mosaic.models.upstream.mol1.features` featurizes Mol-1's own input
JSON instead (molence, torch), into the same features.

A design pack (`binder_features`, or `Mol1.design_pack` of features whose first chain is
the all-Trp placeholder binder) repacks the binder's atoms from `argmax(PSSM)` every step
(see `repack_binder`); a `target_only_features` pack is used as featurized. The model
draws the MSA rows each trunk pass reads as upstream does (`MSAModule.draw_rows`)."""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from jfold import pretrained
from jfold.featurize import Chain, System
from jfold.featurize.views import mol1 as view
from jfold.model import StructureModel
from jfold.models import mol1 as jmol1
from jfold.sampling import random_rotation
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.models.jfold_featurize import (
    AtomSitesWriter,
    check_chains,
    jfold_system,
    residue_type_features,
)
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
    Trunk,
)

# A writer of every entity: jfold's from the view's atom sites, or
# `mosaic.models.upstream.mol1`'s from upstream's `AtomArray`.
type Writer = Callable[[Array], gemmi.Structure]

# How the jfold view's refusals point to upstream's featurization.
UPSTREAM = (
    "mosaic.models.upstream.mol1.features featurizes Mol-1's own input JSON "
    "(install mosaic[all])"
)
# `mol pred`'s trunk passes.
_INFERENCE_RECYCLING_STEPS = 20
# Mol-1's residue alphabet (Protenix's) starts with the 20 amino acids in mosaic order.
NUM_RESTYPES = 32
MSA_GAP = 31  # "-"
# Mol-1's distogram bin centres (`get_bin_centers(2.3125, 21.6875, 64)`).
DISTOGRAM_BINS = np.linspace(2.3125, 21.6875, 129)[1::2].astype(np.float32)
MAX_ATOMS_PER_RESIDUE = 15  # Trp + OXT
# The per-atom features a binder repack rebuilds: the view's, each atom's atom37 slot
# (`atom37_idx`, -1 for padding) and whether it is real (`atom_pad_mask`).
ATOM_KEYS = (
    "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
    "ref_space_uid", "atom_to_token_idx", "atom_to_tokatom_idx",
    "distogram_rep_atom_mask", "atom37_idx", "atom_pad_mask",
)  # fmt: skip


def binder_atom_budget(binder_length: int) -> int:
    """Atoms of the all-Trp placeholder binder of `binder_features` (14 per residue, plus
    OXT on the C-terminal one): enough for any sequence."""
    return (MAX_ATOMS_PER_RESIDUE - 1) * binder_length + 1


def featurize(system: System, *, key: PRNGKeyArray | None = None) -> dict:
    """jfold's Mol-1 dict for `system` (without `is_protein`) plus `atom37_idx` and
    `atom_pad_mask` (all real). `key` poses the reference conformers, as upstream does with
    numpy's global RNG; None leaves them centred and unposed."""
    features = view.features(system, key=key, use_msa=True)
    del features["is_protein"]
    names = view.atom_sites(system).atom_name
    return features | {
        "atom37_idx": np.array([ATOM37_INDEX.get(n, -1) for n in names], np.int32),
        "atom_pad_mask": np.ones(len(names), bool),
    }


def mol1_system(chains: list[TargetChain]) -> System:
    """`chains` as a jfold `System` (`jfold_system`), if jfold's Mol-1 view takes them."""
    check_chains(chains, "Mol-1", templates=True, upstream=UPSTREAM)
    return jfold_system(chains)


def residue_atom_tables() -> dict[str, Array]:
    """Each amino acid's atoms (`TOKENS` order) as `featurize` lays them out, unposed:
    mid-chain (variant 0, the middle residue of `G{aa}G`) and C-terminal (variant 1, with
    OXT, of `G{aa}`). `ATOM_KEYS` tables `[2, 20, MAX_ATOMS_PER_RESIDUE, ...]`, the real
    atoms in the leading slots (`atom_pad_mask`)."""

    def residue(features: dict) -> dict:
        atoms = np.flatnonzero(features["atom_to_token_idx"] == 1)
        padding = MAX_ATOMS_PER_RESIDUE - len(atoms)
        return {
            k: np.pad(
                features[k][atoms], [(0, padding)] + [(0, 0)] * (features[k].ndim - 1)
            )
            for k in ATOM_KEYS
        }

    variants = [residue_type_features(featurize, c) for c in ("G{}G", "G{}")]
    residues = [residue(f) for v in variants for f in v.values()]
    return jax.tree.map(
        lambda *xs: jnp.asarray(np.stack(xs).reshape(2, 20, *xs[0].shape)), *residues
    )


class Mol1Features(eqx.Module):
    """Mol-1's features: `featurize`'s dict (JAX arrays, the MSA one-hot `[S, T, 32]` so
    the binder's query row can be soft). A design pack (`design_pack`) also has the
    binder's length (chain 0, allocated at `binder_atom_budget`) and the per-amino-acid
    atom tables (`residue_atom_tables`) it is repacked from."""

    features: dict[str, Array]
    residue_atoms: dict[str, Array] | None = None
    binder_length: int = eqx.field(static=True, default=0)


def set_binder_sequence(pssm: Float[Array, "L 20"], features: dict) -> dict:
    """Splice the soft sequence into `restype`, `profile` and the MSA query row of the
    first `L` tokens.

    Mol-1 takes `profile` per chain, as the mean over that chain's own MSA rows: the rows
    with a residue in its columns. Swapping the query row is then a rank-one update by
    1 / that depth, which keeps the rest of a binder MSA's statistics; a binder without an
    MSA (`binder_features`) has depth 1, so its profile becomes the sequence. Exact unless
    the featurizer cropped the binder's MSA rows."""
    L = pssm.shape[0]
    # Padding, not a matmul: GPU default (TF32) matmuls would round the PSSM.
    soft = jnp.pad(pssm, ((0, 0), (0, NUM_RESTYPES - 20)))
    restype, profile, msa = features["restype"], features["profile"], features["msa"]
    depth = (msa[:, :L, MSA_GAP] < 0.5).any(-1).sum()
    # Subtract first: exact (`soft`) for depth 1.
    binder_profile = profile[:L] - restype[:L] / depth + soft / depth
    return features | {
        "restype": restype.at[:L].set(soft),
        "profile": profile.at[:L].set(binder_profile),
        "msa": msa.at[0, :L].set(soft),
    }


def feature_pack(features: dict[str, np.ndarray]) -> Mol1Features:
    """`featurize`'s dict (or the upstream path's) as a pack (not a design pack)."""
    return Mol1Features(
        {k: jnp.asarray(v) for k, v in features.items()}
        | {"msa": jax.nn.one_hot(features["msa"], NUM_RESTYPES)}
    )


# Mol-1 embeds every atom's reference conformer (position, element, name, charge), so a
# binder featurized as one fixed sequence would carry that sequence's side chains whatever
# the PSSM says. The binder is allocated at the per-residue maximum (`binder_atom_budget`),
# and every step the argmax residues' atoms are gathered from the per-residue tables and
# packed tightly at the front of the atom axis, the target atoms shifted to follow and the
# freed tail masked as padding. Static shapes, no recompiles. The argmax has no gradient:
# the PSSM gradient flows through the sequence channels. Each binder residue's reference
# conformer gets a fresh random pose every step, as the featurizer's.


def repack_binder(
    pack: Mol1Features, pssm: Float[Array, "L 20"], key: PRNGKeyArray
) -> Mol1Features:
    """Repack the binder's atoms as `argmax(pssm)` (see the comment above): every
    `ATOM_KEYS` feature, rebuilt for the new layout. `ref_pos` gets the featurizer's pose:
    per residue, a random translation in [-1, 1]^3 and then a uniform random rotation."""
    L, features = pack.binder_length, pack.features
    assert pssm.shape == (L, 20), (pssm.shape, L)
    num_atoms = features["ref_pos"].shape[0]
    allocated = binder_atom_budget(L)
    variant = (jnp.arange(L) == L - 1).astype(jnp.int32)  # C-terminal residue has OXT
    ids = jnp.argmax(pssm, axis=-1)
    binder = {k: table[variant, ids] for k, table in pack.residue_atoms.items()}

    count = binder["atom_pad_mask"].sum(-1)
    first = jnp.cumsum(count) - count  # first atom of each binder residue
    slot = jnp.arange(MAX_ATOMS_PER_RESIDUE)
    # Each table slot's destination; empty slots go out of range (dropped).
    real = slot < count[:, None]
    dest = jnp.where(real, first[:, None] + slot, num_atoms).reshape(-1)
    target_dest = jnp.arange(allocated, num_atoms) + jnp.sum(count) - allocated

    k_translation, k_rotation = jax.random.split(key)
    translation = jax.random.uniform(k_translation, (L, 1, 3), minval=-1.0, maxval=1.0)
    rotation = jax.vmap(random_rotation)(jax.random.split(k_rotation, L))
    # Binder residues own tokens and reference-space uids 0..L-1.
    token = jnp.broadcast_to(jnp.arange(L)[:, None], (L, MAX_ATOMS_PER_RESIDUE))
    posed = jnp.einsum("lac,ldc->lad", binder["ref_pos"] + translation, rotation)
    binder |= {
        "ref_pos": posed,
        "ref_space_uid": token,
        "atom_to_token_idx": token,
    }

    def repack(name: str) -> Array:
        """Binder atoms packed tightly, then the target atoms; the freed tail is padding."""
        field = features[name]
        atoms = binder[name].reshape(-1, *field.shape[1:]).astype(field.dtype)
        packed = jnp.full_like(field, -1 if name == "atom37_idx" else 0)
        packed = packed.at[dest].set(atoms, mode="drop")
        return packed.at[target_dest].set(field[allocated:])

    return dataclasses.replace(
        pack, features=features | {name: repack(name) for name in ATOM_KEYS}
    )


class DesignWriter:
    """Writes a design pack's coordinates `[A, 3]` as a `gemmi.Structure`. With
    `binder_pssm` (`[L, 20]`), the binder (the first chain) as its argmax, from the atoms
    `repack_binder` packs at the front, and every other entity as `writer` (of the pack as
    featurized, around the all-Trp placeholder) writes it; without, the pack as
    featurized."""

    def __init__(self, writer: Writer, binder_length: int):
        self.writer = writer
        self.binder_length = binder_length
        # Each amino acid's atoms (name, atomic number) as a chain's last residue.
        self.residues = {}
        for aa in TOKENS:
            sites = view.atom_sites(System((Chain("A", aa),)))
            self.residues[aa] = list(zip(sites.atom_name, sites.element.tolist()))

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        if binder_pssm is None:
            return self.writer(coords)
        coords = np.asarray(coords, np.float64).reshape(-1, 3)
        sequence = [TOKENS[i] for i in np.asarray(binder_pssm).argmax(-1)]
        # OXT on the C-terminal residue only.
        layout = [[a for a in self.residues[aa] if a[0] != "OXT"] for aa in sequence]
        layout[-1] = self.residues[sequence[-1]]
        num_binder_atoms = sum(map(len, layout))
        # The other entities' atoms follow the binder's: put them back after the
        # placeholder's.
        placeholder = np.zeros((binder_atom_budget(self.binder_length), 3))
        st = self.writer(np.concatenate([placeholder, coords[num_binder_atoms:]]))
        positions = iter(coords[:num_binder_atoms])
        for residue, aa, atoms in zip(st[0][0], sequence, layout, strict=True):
            prototype = residue[0].clone()
            residue.name = gemmi.expand_one_letter(aa, gemmi.ResidueKind.AA)
            while len(residue):
                del residue[0]
            for name, element in atoms:
                atom = prototype.clone()
                atom.name = name
                atom.element = gemmi.Element(element)
                atom.pos = gemmi.Position(*next(positions))
                residue.add_atom(atom)
        return st


class Mol1(DiffusionStructurePredictionModel[Mol1Features]):
    """Mol-1 (`mol-1-base`) on jfold. `recycling_steps` counts trunk passes (upstream's
    `N_cycle`); the upstream sampler's step count is the converted model's
    (`model.sampling.num_steps`)."""

    model: StructureModel
    residue_atoms: dict[str, Array]
    inference_recycling_steps: int = eqx.field(static=True)
    design_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    distogram_bins = DISTOGRAM_BINS

    def __init__(
        self,
        *,
        model: StructureModel | None = None,
        inference_recycling_steps: int = _INFERENCE_RECYCLING_STEPS,
        design_recycling_steps: int = 1,
        backprop_all_passes: bool = False,
    ):
        """Loads the converted model unless `model` is passed: from mosaic's `jfold`
        cache, converted from the upstream checkpoint on first use."""
        if model is None:
            model = pretrained.load("mol1_base", cache=cache_dir("jfold"))
        self.model = model
        self.residue_atoms = residue_atom_tables()
        self.inference_recycling_steps = inference_recycling_steps
        self.design_recycling_steps = design_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    def target_only_features(
        self, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[Mol1Features, Writer]:
        """Featurize the chains as given; `key` poses the reference conformers."""
        system = mol1_system(chains)
        writer = AtomSitesWriter(view.atom_sites(system))
        return feature_pack(featurize(system, key=key)), writer

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[Mol1Features, DesignWriter]:
        """Design pack (`design_pack`): an all-Trp placeholder binder (chain 0) plus the
        targets."""
        chains = [TargetChain("W" * binder_length, use_msa=False), *chains]
        features, writer = self.target_only_features(chains, key=key)
        return self.design_pack(features, writer, binder_length)

    def design_pack(
        self, features: Mol1Features, writer: Writer, binder_length: int
    ) -> tuple[Mol1Features, DesignWriter]:
        """`features` (of either featurization) as a design pack, its binder repacked from
        `argmax(PSSM)` every step, and `writer` (of every entity) as its `DesignWriter`.
        The binder is the first chain, featurized as the all-Trp placeholder
        `"W" * binder_length`; anything else raises `ValueError`."""
        f = features.features
        asym_id = np.asarray(f["asym_id"])
        residues = int(np.sum(asym_id == asym_id[0]))
        atoms = int(np.sum(np.asarray(f["atom_to_token_idx"]) < binder_length))
        budget = binder_atom_budget(binder_length)
        if (residues, atoms) != (binder_length, budget):
            raise ValueError(
                "a design pack's binder is its first chain, featurized as the all-Trp "
                f"placeholder 'W' * {binder_length}: {binder_length} residues, {budget} "
                f"atoms; this first chain has {residues} residues and its first "
                f"{binder_length} tokens {atoms} atoms"
            )
        pack = dataclasses.replace(
            features, residue_atoms=self.residue_atoms, binder_length=binder_length
        )
        return pack, DesignWriter(writer, binder_length)

    def set_sequence(
        self,
        features: Mol1Features,
        sequence: Float[Array, "L 20"],
        *,
        key: PRNGKeyArray,
    ) -> Mol1Features:
        """Set the binder's sequence channels; a design pack also repacks the binder's
        atoms from `argmax(sequence)`."""
        pack = features
        if pack.residue_atoms is not None:
            pack = repack_binder(pack, sequence, key)
        return dataclasses.replace(
            pack, features=set_binder_sequence(sequence, pack.features)
        )

    def jfold_features(self, features: Mol1Features) -> JFeatures:
        return jmol1.features(features.features)

    def token_plddt(self, plddt, trunk: Trunk):
        """The per-atom head at each token's CA, or at its representative atom for a token
        without one (a nucleotide)."""
        f = trunk.features.features
        num_tokens = f["asym_id"].shape[0]
        is_ca = f["atom37_idx"] == ATOM37_INDEX["CA"]
        ca_token = jnp.where(is_ca, f["atom_to_token_idx"], num_tokens)  # out of range
        ca = jnp.zeros(num_tokens, jnp.int32).at[ca_token].set(
            jnp.arange(is_ca.shape[0]), mode="drop"
        )
        has_ca = jnp.zeros(num_tokens, bool).at[ca_token].set(True, mode="drop")
        return plddt[jnp.where(has_ca, ca, trunk.inputs.features.rep_atom)]

    def tokens(self, features: Mol1Features) -> TokenInfo:
        f = features.features
        return TokenInfo(
            sequence=f["restype"][:, :20],
            asym_id=f["asym_id"],
            residue_index=f["residue_index"],
            mask=jnp.ones(f["asym_id"].shape, bool),
        )

    def atom37(self, features: Mol1Features, coords: Float[Array, "A 3"]) -> Atom37:
        f = features.features
        coords37, mask37 = scatter_atom37(
            atom_coords=coords,
            atom_to_token=f["atom_to_token_idx"],
            atom37_idx=f["atom37_idx"],
            n_token=f["asym_id"].shape[0],
        )
        return Atom37(coords=coords37, mask=mask37)

    def to_structure(
        self,
        features: Mol1Features,
        writer: Writer | DesignWriter,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """A design pack's binder is written as `argmax(sequence)` (as featurized without
        one)."""
        if features.residue_atoms is not None:
            return writer(output.structure_coordinates, sequence)
        return writer(output.structure_coordinates)
