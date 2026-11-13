"""OpenDDE backend running on jfold.

Inputs are featurized with `jfold.featurize`, which builds the same features as upstream
OpenDDE apart from the random placement of reference conformers. The forward pass uses
jfold's converted OpenDDE weights from `jfold.pretrained`. Only protein chains of the 20
amino acids and X are supported, without templates."""

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jfold import pretrained
from jfold.featurize.views import opendde as view
from jfold.featurize.views.protenix import NUM_CLASSES
from jfold.model import StructureModel
from jfold.models import opendde as jfold_opendde
from jfold.sampling import random_rotation
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.models.jfold_featurize import (
    AtomSitesWriter,
    BinderAtomSitesWriter,
    check_chains,
    jfold_system,
    residue_type_features,
)
from mosaic.structure_prediction import (
    ATOM37_NAMES,
    Atom37,
    DiffusionStructurePredictionModel,
    StructureModelOutput,
    TargetChain,
    TokenInfo,
    scatter_atom37,
)

# Number of trunk passes for design and for `predict` (upstream's `N_cycle`).
DESIGN_RECYCLING_STEPS = 4
INFERENCE_RECYCLING_STEPS = 10
# Distogram bin centres, from `confidence.distogram` in OpenDDE's config.
OPENDDE_DISTOGRAM_BINS = 2.25 + (np.arange(96) + 0.5) * (25.75 - 2.25) / 96

_CHECKPOINTS = {"opendde_v1": "opendde.pt", "opendde_abag": "opendde_abag.pt"}

# The most atoms a residue can have, reached by a C-terminal Trp.
MAX_RESIDUE_ATOMS = 15
# Per-atom features that are rebuilt for the binder during design.
RESIDUE_ATOM_KEYS = (
    "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
    "atom_to_tokatom_idx", "is_ligand", "distogram_rep_atom_mask", "pae_rep_atom_mask",
    "atom_to_structural_tokatom_idx", "structural_distogram_rep_atom_mask",
    "structural_pae_rep_atom_mask",
)  # fmt: skip
# Atom37 names encoded like `ref_atom_name_chars`: character codes minus 32, space-padded.
_ATOM37_CHARS = np.array([[ord(c) - 32 for c in n.ljust(4)] for n in ATOM37_NAMES])


class ResidueTables(eqx.Module):
    """Atom features of each amino acid, used to rebuild binder residues during design.

    Every array has leading axes `[2, 20]`. The first axis is the residue's position in the
    chain (0 for interior, 1 for C-terminal with `OXT`) and the second is the amino acid in
    `TOKENS` order. Reference conformers are centred and not randomly placed. Atom slots
    past `count` are zero, and atom indices count from the residue's first atom."""

    atoms: dict[str, Array]  # `RESIDUE_ATOM_KEYS`, each `[2, 20, A, ...]`
    count: Int[Array, "2 20"]
    structural_token: Int[Array, "2 20 A"]  # 0 backbone, 1 side chain
    frame: Int[Array, "2 20 3"]
    # Frames of the backbone and side-chain tokens. Glycine repeats its single frame.
    structural_frame: Int[Array, "2 20 2 3"]


def _residue_table(f: dict) -> dict:
    """Extract the second residue's atom features from a featurized `G{}G` or `G{}`."""
    atoms = np.flatnonzero(f["atom_to_token_idx"] == 1)
    tokens = np.flatnonzero(f["parent_residue_idx"] == 1)
    first = atoms[0]

    def pad(x):
        return np.pad(x, [(0, MAX_RESIDUE_ATOMS - len(x))] + [(0, 0)] * (x.ndim - 1))

    return {
        "atoms": {k: pad(f[k][atoms]) for k in RESIDUE_ATOM_KEYS},
        "count": len(atoms),
        "structural_token": pad(f["atom_to_structural_token_idx"][atoms] - tokens[0]),
        "frame": f["frame_atom_index"][1] - first,
        "structural_frame": f["structural_frame_atom_index"][tokens[[0, -1]]] - first,
    }


def _stack(trees):
    return jax.tree.map(lambda *x: np.stack(x), *trees)


def residue_tables() -> ResidueTables:
    """Build `ResidueTables` by featurizing each amino acid in `G{}G` and `G{}`."""
    contexts = [residue_type_features(view.features, c) for c in ("G{}G", "G{}")]
    tables = [_stack([_residue_table(f) for f in c.values()]) for c in contexts]
    return ResidueTables(**jax.tree.map(jnp.asarray, _stack(tables)))


class DesignFeatures(eqx.Module):
    """Features for binder design.

    `features` holds the features of a poly-Trp binder, the first chain, followed by the
    target chains. `residues` is used to rebuild the binder's atoms for each sequence."""

    features: dict
    residues: ResidueTables
    binder_length: int = eqx.field(static=True)
    binder_atoms: int = eqx.field(static=True)


def _dict(features: dict | DesignFeatures) -> dict:
    return features.features if isinstance(features, DesignFeatures) else features


def model_features(f: dict) -> dict:
    """Convert jfold's feature dict to JAX arrays, one-hot encoding the MSA.

    The one-hot MSA lets a design's query row hold soft probabilities. All MSA rows are
    kept because the model samples rows on each trunk pass, as upstream does."""
    f = f | {"msa": np.eye(NUM_CLASSES, dtype=np.float32)[f["msa"]]}
    return {k: jnp.asarray(v) for k, v in f.items()}


def target_only_features(chains: list[TargetChain], *, key: PRNGKeyArray):
    """Featurize `chains` and return the features with an `AtomSitesWriter`.

    `key` randomly rotates and translates each residue's reference conformer, as upstream
    does with numpy's global RNG."""
    check_chains(chains, "OpenDDE", templates=False)
    system = jfold_system(chains)
    features = model_features(view.features(system, key=key))
    return features, AtomSitesWriter(view.atom_sites(system))


def binder_features(
    binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
) -> tuple[DesignFeatures, BinderAtomSitesWriter]:
    """Featurize a poly-Trp binder followed by `chains` for design.

    Trp has the most atoms, so the atoms of any binder sequence fit in its slots."""
    if binder_length < 1:
        raise ValueError("OpenDDE binder_length must be positive")
    check_chains(chains, "OpenDDE", templates=False)
    system = jfold_system([TargetChain("W" * binder_length, use_msa=False), *chains])
    features = model_features(view.features(system, key=key))
    design = DesignFeatures(
        features=features,
        residues=residue_tables(),
        binder_length=binder_length,
        binder_atoms=int((features["atom_to_token_idx"] < binder_length).sum()),
    )
    return design, BinderAtomSitesWriter(system, view.atom_sites)


def rebuild_binder(design: DesignFeatures, sequence, key: PRNGKeyArray) -> dict:
    """Return the design's atom features with binder residues set to `argmax(sequence)`.

    The binder's atoms are packed at the front, followed by the target atoms, and the
    remaining slots become padding. As upstream does, each binder residue's reference
    conformer is randomly rotated and then translated by up to 1 Å along each axis. Each
    binder residue keeps two structural tokens, and glycine's side-chain token has no
    atoms."""
    f, tables = design.features, design.residues
    length, binder_atoms = design.binder_length, design.binder_atoms
    num_atoms = f["ref_pos"].shape[0]
    num_tokens = f["restype"].shape[0]
    num_structural = f["structural_token_index"].shape[0]
    context = (jnp.arange(length) == length - 1).astype(jnp.int32)
    aa = sequence.argmax(-1)

    def pick(table):
        return table[context, aa]

    count = pick(tables.count)
    start = jnp.cumsum(count) - count
    packed = count.sum()
    shift = packed - binder_atoms
    slot = jnp.arange(MAX_RESIDUE_ATOMS)
    dest = jnp.where(slot < count[:, None], start[:, None] + slot, num_atoms)

    def repack(binder, name, fill=0):
        """Return `f[name]` with the binder atoms replaced by `binder`."""
        out = jnp.full_like(f[name], fill).at[dest].set(binder, mode="drop")
        targets = f[name][binder_atoms:]
        return jax.lax.dynamic_update_slice_in_dim(out, targets, packed, 0)

    def frames(binder, name, tokens):
        """Return the frame atom indices `f[name]` with the first `tokens` rows replaced."""
        return jnp.concatenate([binder, f[name][tokens:] + shift])

    k_rotation, k_translation = jax.random.split(key)
    rotation = jax.vmap(random_rotation)(jax.random.split(k_rotation, length))
    translation = jax.random.uniform(k_translation, (length, 3), minval=-1, maxval=1)
    ref_pos = jnp.einsum("lij,laj->lai", rotation, pick(tables.atoms["ref_pos"]))
    residue = jnp.broadcast_to(jnp.arange(length)[:, None], dest.shape)
    structural_token = 2 * residue + pick(tables.structural_token)
    structural_frame = start[:, None, None] + pick(tables.structural_frame)
    return {k: repack(pick(tables.atoms[k]), k) for k in RESIDUE_ATOM_KEYS} | {
        "ref_pos": repack(ref_pos + translation[:, None], "ref_pos"),
        "atom_to_token_idx": repack(residue, "atom_to_token_idx", num_tokens),
        "ref_space_uid": repack(residue, "ref_space_uid", num_tokens),
        "atom_to_structural_token_idx": repack(
            structural_token, "atom_to_structural_token_idx", num_structural
        ),
        "atom_pad_mask": jnp.arange(num_atoms) < num_atoms + shift,
        "frame_atom_index": frames(
            start[:, None] + pick(tables.frame), "frame_atom_index", length
        ),
        "structural_frame_atom_index": frames(
            structural_frame.reshape(2 * length, 3),
            "structural_frame_atom_index",
            2 * length,
        ),
    }


def set_binder_sequence(
    design: DesignFeatures, sequence: Float[Array, "L 20"], key: PRNGKeyArray
) -> DesignFeatures:
    """Set the binder's `restype`, `profile` and MSA query row to the soft `sequence`.

    The binder's atoms are also rebuilt from `argmax(sequence)`, without gradients."""
    if not isinstance(design, DesignFeatures):
        raise TypeError(
            "OpenDDE sets the binder sequence of a design pack (from `binder_features`), "
            f"not of {type(design).__name__}"
        )
    if sequence.shape != (design.binder_length, 20):
        raise ValueError(
            f"OpenDDE binder PSSM must be [{design.binder_length}, 20], "
            f"got {list(sequence.shape)}"
        )
    f = design.features
    soft = jnp.pad(sequence, ((0, 0), (0, NUM_CLASSES - 20)))
    channels = {
        "restype": f["restype"].at[: len(soft)].set(soft),
        "profile": f["profile"].at[: len(soft)].set(soft),
        "msa": f["msa"].at[0, : len(soft)].set(soft),
    }
    features = f | rebuild_binder(design, sequence, key) | channels
    return eqx.tree_at(lambda d: d.features, design, features)


class OpenDDE(DiffusionStructurePredictionModel[dict | DesignFeatures]):
    """OpenDDE with upstream's recycling counts.

    The number of sampling steps and of MSA rows per trunk pass come from the converted
    model. Recycling steps count trunk passes, like upstream's `N_cycle`."""

    model: StructureModel
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    distogram_bins = OPENDDE_DISTOGRAM_BINS

    def __init__(
        self,
        model: StructureModel,
        *,
        design_recycling_steps: int = DESIGN_RECYCLING_STEPS,
        inference_recycling_steps: int = INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
    ):
        """Wrap a converted OpenDDE model. Use `OpenDDE.v1()` or `OpenDDE.abag()` to load
        released checkpoints."""
        self.model = model
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    @classmethod
    def _release(cls, name: str, **settings) -> "OpenDDE":
        """Load the `jfold.pretrained` checkpoint `name`, converting it on first use."""
        checkpoint = cache_dir("opendde", "checkpoint", _CHECKPOINTS[name])
        model = pretrained.load(name, checkpoint_path=checkpoint, cache=cache_dir("jfold"))
        return cls(model, **settings)

    @classmethod
    def v1(cls, **settings) -> "OpenDDE":
        """Load OpenDDE v1. `settings` go to `__init__`."""
        return cls._release("opendde_v1", **settings)

    @classmethod
    def abag(cls, **settings) -> "OpenDDE":
        """Load OpenDDE with the antibody-antigen weights."""
        return cls._release("opendde_abag", **settings)

    def target_only_features(self, chains: list[TargetChain], *, key: PRNGKeyArray):
        """Featurize `chains`. `key` randomly places the reference conformers."""
        return target_only_features(chains, key=key)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        """Featurize a binder followed by `chains` for design. `key` randomly places the
        reference conformers."""
        return binder_features(binder_length, chains, key=key)

    def set_sequence(
        self,
        features: DesignFeatures,
        sequence: Float[Array, "L 20"],
        *,
        key: PRNGKeyArray,
    ) -> DesignFeatures:
        """Set the binder's sequence and rebuild its atoms."""
        return set_binder_sequence(features, sequence, key)

    def jfold_features(self, features) -> JFeatures:
        return jfold_opendde.features(_dict(features))

    def tokens(self, features) -> TokenInfo:
        f = _dict(features)
        return TokenInfo(
            sequence=f["restype"][:, :20],
            asym_id=f["asym_id"],
            residue_index=f["residue_index"],
            mask=jnp.ones(f["asym_id"].shape, bool),
        )

    def atom37(self, features, coords: Float[Array, "A 3"]) -> Atom37:
        """Scatter atoms to atom37 by atom name, including `OXT`."""
        f = _dict(features)
        names = f["ref_atom_name_chars"].argmax(-1)
        match = (names[:, None] == _ATOM37_CHARS).all(-1)
        slot = jnp.where(match.any(-1), match.argmax(-1), -1)
        coords37, mask37 = scatter_atom37(
            coords, f["atom_to_token_idx"], slot, f["restype"].shape[0]
        )
        return Atom37(coords=coords37, mask=mask37)

    def to_structure(
        self, features, writer, output: StructureModelOutput, sequence
    ) -> gemmi.Structure:
        """Build a `gemmi.Structure`, using `sequence` for the binder when it is given."""
        del features
        coords = output.structure_coordinates
        return writer(coords) if sequence is None else writer(coords, sequence)
