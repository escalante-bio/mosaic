"""Mol-1 (`mol-1-base`) backend running on jfold.

Inputs are featurized with `jfold.featurize`, which builds the same features as Mol-1's own
featurizer apart from the random placement of reference conformers. The forward pass uses
jfold's converted Mol-1 weights from `jfold.pretrained`.

Supported inputs are protein chains of the 20 amino acids and X, each with an unpaired MSA
and an optional template of the same length as the chain. Other inputs, such as nucleic
acids, can be featurized from a Mol-1 input JSON with
`mosaic.models.upstream.mol1.features`.

For binder design, the binder's atoms are rebuilt from `argmax(PSSM)` at every step (see
`repack_binder`)."""

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
from mosaic.models.jfold_featurize import (
    AtomSitesWriter,
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
    Trunk,
    scatter_atom37,
)

# Converts predicted coordinates to a `gemmi.Structure`.
type Writer = Callable[[Array], gemmi.Structure]

# Hint added to errors for inputs that jfold cannot featurize.
UPSTREAM = (
    "mosaic.models.upstream.mol1.features featurizes Mol-1's own input JSON "
    "(install mosaic[all])"
)
# Number of trunk passes used by `mol pred`.
_INFERENCE_RECYCLING_STEPS = 20
# Mol-1's residue alphabet (Protenix's) starts with the 20 amino acids in mosaic order.
NUM_RESTYPES = 32
MSA_GAP = 31  # "-"
# Distogram bin centres, equal to upstream's `get_bin_centers(2.3125, 21.6875, 64)`.
DISTOGRAM_BINS = np.linspace(2.3125, 21.6875, 129)[1::2].astype(np.float32)
MAX_ATOMS_PER_RESIDUE = 15  # Trp + OXT
# Per-atom features that are rebuilt for the binder during design. `atom37_idx` is -1 for
# padding.
ATOM_KEYS = (
    "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
    "ref_space_uid", "atom_to_token_idx", "atom_to_tokatom_idx",
    "distogram_rep_atom_mask", "atom37_idx", "atom_pad_mask",
)  # fmt: skip


def binder_atom_budget(binder_length: int) -> int:
    """Return the number of atoms in a poly-Trp binder, which fits any sequence."""
    return (MAX_ATOMS_PER_RESIDUE - 1) * binder_length + 1


def featurize(system: System, *, key: PRNGKeyArray | None = None) -> dict:
    """Featurize `system` with jfold and add `atom37_idx` and `atom_pad_mask`.

    `key` randomly rotates and translates each residue's reference conformer, as upstream
    does with numpy's global RNG. If `key` is None, the conformers are left centred."""
    features = view.features(system, key=key, use_msa=True)
    del features["is_protein"]
    names = view.atom_sites(system).atom_name
    return features | {
        "atom37_idx": np.array([ATOM37_INDEX.get(n, -1) for n in names], np.int32),
        "atom_pad_mask": np.ones(len(names), bool),
    }


def mol1_system(chains: list[TargetChain]) -> System:
    """Check that jfold can featurize `chains` for Mol-1 and build their `System`."""
    check_chains(chains, "Mol-1", templates=True, upstream=UPSTREAM)
    return jfold_system(chains)


def residue_atom_tables() -> dict[str, Array]:
    """Return the `ATOM_KEYS` features of each amino acid, used to rebuild the binder.

    Each table has shape `[2, 20, MAX_ATOMS_PER_RESIDUE, ...]`. The first axis is 0 for an
    interior residue and 1 for a C-terminal residue with `OXT`, and the second is the amino
    acid in `TOKENS` order. Real atoms come first and conformers are not randomly placed."""

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
    """Mol-1 input features.

    `features` holds JAX arrays, with the MSA one-hot encoded as `[S, T, 32]` so that the
    binder's query row can hold soft probabilities. For binder design, `residue_atoms` and
    `binder_length` are set so that the binder, the first chain, can be rebuilt."""

    features: dict[str, Array]
    residue_atoms: dict[str, Array] | None = None
    binder_length: int = eqx.field(static=True, default=0)


def set_binder_sequence(pssm: Float[Array, "L 20"], features: dict) -> dict:
    """Set `restype`, `profile` and the MSA query row of the first `L` tokens to `pssm`.

    Mol-1 computes `profile` per chain as the mean over the MSA rows that are not gaps in
    that chain's columns. Replacing the query row therefore changes the profile by the
    difference divided by that row count. A binder without an MSA has one row, so its
    profile becomes the sequence. The update is exact unless the featurizer cropped the
    binder's MSA."""
    L = pssm.shape[0]
    # Pad instead of using a matmul, which would round the PSSM under TF32 on GPU.
    soft = jnp.pad(pssm, ((0, 0), (0, NUM_RESTYPES - 20)))
    restype, profile, msa = features["restype"], features["profile"], features["msa"]
    depth = (msa[:, :L, MSA_GAP] < 0.5).any(-1).sum()
    # Subtracting first makes the result exactly `soft` when the depth is 1.
    binder_profile = profile[:L] - restype[:L] / depth + soft / depth
    return features | {
        "restype": restype.at[:L].set(soft),
        "profile": profile.at[:L].set(binder_profile),
        "msa": msa.at[0, :L].set(soft),
    }


def feature_pack(features: dict[str, np.ndarray]) -> Mol1Features:
    """Wrap a feature dict from jfold or upstream as `Mol1Features`."""
    return Mol1Features(
        {k: jnp.asarray(v) for k, v in features.items()}
        | {"msa": jax.nn.one_hot(features["msa"], NUM_RESTYPES)}
    )


# Mol-1 embeds every atom's reference conformer, so a binder featurized with one fixed
# sequence would keep that sequence's side chains whatever the PSSM says. Instead, the
# binder gets enough atom slots for poly-Trp, and at every step the atoms of the argmax
# residues are packed at the front, followed by the target atoms and then padding. Shapes
# remain static, so nothing recompiles. The PSSM gradient flows only through the sequence
# features.


def repack_binder(
    pack: Mol1Features, pssm: Float[Array, "L 20"], key: PRNGKeyArray
) -> Mol1Features:
    """Rebuild the binder's `ATOM_KEYS` features for `argmax(pssm)`.

    As in the featurizer, each residue's `ref_pos` is translated by a uniform random offset
    in [-1, 1]^3 and then randomly rotated."""
    L, features = pack.binder_length, pack.features
    assert pssm.shape == (L, 20), (pssm.shape, L)
    num_atoms = features["ref_pos"].shape[0]
    allocated = binder_atom_budget(L)
    variant = (jnp.arange(L) == L - 1).astype(jnp.int32)  # the C-terminal residue has OXT
    ids = jnp.argmax(pssm, axis=-1)
    binder = {k: table[variant, ids] for k, table in pack.residue_atoms.items()}

    count = binder["atom_pad_mask"].sum(-1)
    first = jnp.cumsum(count) - count
    slot = jnp.arange(MAX_ATOMS_PER_RESIDUE)
    # Empty slots point past the end, so their writes are dropped.
    real = slot < count[:, None]
    dest = jnp.where(real, first[:, None] + slot, num_atoms).reshape(-1)
    target_dest = jnp.arange(allocated, num_atoms) + jnp.sum(count) - allocated

    k_translation, k_rotation = jax.random.split(key)
    translation = jax.random.uniform(k_translation, (L, 1, 3), minval=-1.0, maxval=1.0)
    rotation = jax.vmap(random_rotation)(jax.random.split(k_rotation, L))
    # Binder residues have token indices and reference-space uids 0 to L-1.
    token = jnp.broadcast_to(jnp.arange(L)[:, None], (L, MAX_ATOMS_PER_RESIDUE))
    posed = jnp.einsum("lac,ldc->lad", binder["ref_pos"] + translation, rotation)
    binder |= {
        "ref_pos": posed,
        "ref_space_uid": token,
        "atom_to_token_idx": token,
    }

    def repack(name: str) -> Array:
        field = features[name]
        atoms = binder[name].reshape(-1, *field.shape[1:]).astype(field.dtype)
        packed = jnp.full_like(field, -1 if name == "atom37_idx" else 0)
        packed = packed.at[dest].set(atoms, mode="drop")
        return packed.at[target_dest].set(field[allocated:])

    return dataclasses.replace(
        pack, features=features | {name: repack(name) for name in ATOM_KEYS}
    )


class DesignWriter:
    """Convert design coordinates of shape `[A, 3]` to a `gemmi.Structure`.

    If `binder_pssm` is given, the binder, the first chain, is written as its argmax using
    the atoms that `repack_binder` placed at the front. The other chains are written by
    `writer`. Without `binder_pssm`, `writer` writes the features as featurized."""

    def __init__(self, writer: Writer, binder_length: int):
        self.writer = writer
        self.binder_length = binder_length
        # Atom names and atomic numbers of each amino acid as a C-terminal residue.
        self.residues = {}
        for aa in TOKENS:
            sites = view.atom_sites(System((Chain("A", aa),)))
            self.residues[aa] = list(zip(sites.atom_name, sites.element.tolist()))

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        if binder_pssm is None:
            return self.writer(coords)
        coords = np.asarray(coords, np.float64).reshape(-1, 3)
        sequence = [TOKENS[i] for i in np.asarray(binder_pssm).argmax(-1)]
        # Only the C-terminal residue has OXT.
        layout = [[a for a in self.residues[aa] if a[0] != "OXT"] for aa in sequence]
        layout[-1] = self.residues[sequence[-1]]
        num_binder_atoms = sum(map(len, layout))
        # Place the target atoms after the placeholder binder's atoms for `writer`.
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
    """Mol-1 (`mol-1-base`) on jfold.

    Recycling steps count trunk passes, like upstream's `N_cycle`. The number of sampling
    steps comes from the converted model."""

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
        """Wrap `model`, or load the converted Mol-1 checkpoint if it is None."""
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
        """Featurize `chains`. `key` randomly places the reference conformers."""
        system = mol1_system(chains)
        writer = AtomSitesWriter(view.atom_sites(system))
        return feature_pack(featurize(system, key=key)), writer

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[Mol1Features, DesignWriter]:
        """Featurize a poly-Trp binder followed by `chains` for binder design."""
        chains = [TargetChain("W" * binder_length, use_msa=False), *chains]
        features, writer = self.target_only_features(chains, key=key)
        return self.design_pack(features, writer, binder_length)

    def design_pack(
        self, features: Mol1Features, writer: Writer, binder_length: int
    ) -> tuple[Mol1Features, DesignWriter]:
        """Turn features whose first chain is `"W" * binder_length` into design features.

        The features can come from jfold or from upstream. Setting a sequence on the result
        rebuilds the binder's atoms. Raise `ValueError` if the first chain is not such a
        poly-Trp binder."""
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
        """Set the binder's sequence, also rebuilding its atoms for design features."""
        pack = features
        if pack.residue_atoms is not None:
            pack = repack_binder(pack, sequence, key)
        return dataclasses.replace(
            pack, features=set_binder_sequence(sequence, pack.features)
        )

    def jfold_features(self, features: Mol1Features) -> JFeatures:
        return jmol1.features(features.features)

    def token_plddt(self, plddt, trunk: Trunk):
        """Take each token's pLDDT from its CA atom, or from its representative atom if it
        has no CA."""
        f = trunk.features.features
        num_tokens = f["asym_id"].shape[0]
        is_ca = f["atom37_idx"] == ATOM37_INDEX["CA"]
        ca_token = jnp.where(is_ca, f["atom_to_token_idx"], num_tokens)
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
        """Build a `gemmi.Structure`, writing a design's binder as `argmax(sequence)`."""
        if features.residue_atoms is not None:
            return writer(output.structure_coordinates, sequence)
        return writer(output.structure_coordinates)
