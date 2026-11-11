"""OpenFold3 on jfold: features by `jfold.featurize`'s OpenFold3 view (torch-free; the dict
upstream OpenFold3 builds, up to each residue's reference conformer, drawn from a stored
table, and its random pose), jfold's converted OpenFold3 (`jfold.pretrained`) for the
forward pass, and the view's atom sites for writing structures.

Features are the view's dict, unbatched, with `restype` and `msa` as float32 so that a
binder's soft sequence can be set, plus `rep_atom`: each token's representative atom in the
featurized atoms (a binder keeps its placeholder's whatever its soft sequence). The view
takes protein chains of the 20 amino acids and X, unpaired MSAs, and templates as upstream's
pipeline takes them (realigned to their chain with Kalign, shared by chains with the same
sequence); anything else raises `NotImplementedError`.
`mosaic.models.upstream.openfold3.features` featurizes OpenFold3's own query instead (DNA,
RNA, ligands, MSAs for several sequences, any template), into the same features."""

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from jfold import pretrained
from jfold.featurize.views import openfold3 as view
from jfold.model import StructureModel
from jfold.models import openfold3 as jfold_of3
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.models.jfold_featurize import (
    AtomSitesWriter,
    check_chains,
    check_unpaired_msas,
    jfold_system,
)
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
)

# 64 bins from 2 to 22 Å, at their centres.
_DISTOGRAM_EDGES = np.linspace(2.0, 22.0, 65)
DISTOGRAM_BINS = (_DISTOGRAM_EDGES[1:] + _DISTOGRAM_EDGES[:-1]) / 2
# Upstream's three recycles plus the first pass.
INFERENCE_RECYCLING_STEPS = 4


def _atom37_table() -> np.ndarray:
    """Each of OpenFold3's residue classes' atoms (`[32, max atoms]`, in the view's atom
    order) as atom37 slots; -1 where there is none (and for the classes the view does not
    featurize)."""
    residues = view.residues()
    atoms = [
        residues[name].atoms if name in residues else () for name in jfold_of3.RESIDUES
    ]
    table = -np.ones((view.NUM_CLASSES, max(map(len, atoms))), np.int32)
    for restype, names in enumerate(atoms):
        table[restype, : len(names)] = [ATOM37_INDEX.get(name, -1) for name in names]
    return table


_ATOM37_TABLE = _atom37_table()


UPSTREAM = (
    "mosaic.models.upstream.openfold3.features featurizes OpenFold3's own query "
    "(install mosaic[all])"
)


def backend_features(features: dict) -> dict:
    """The backend's features from the view's (or upstream's) dict: with each token's
    representative atom, and `restype` and `msa` as float32, as JAX arrays."""
    features = features | {
        "rep_atom": jfold_of3.representative_atoms(
            features["restype"],
            features["is_protein"],
            features["is_dna"],
            features["is_rna"],
            features["is_atomized"],
            features["start_atom_index"],
        ),
        "restype": features["restype"].astype(np.float32),
        "msa": features["msa"].astype(np.float32),
    }
    return jax.tree.map(jnp.asarray, features)


def target_only_features(chains: list[TargetChain], *, key: PRNGKeyArray):
    """The view's features for `chains` (see the module docstring) and a writer. `key`
    draws the reference conformers and their poses, as upstream does with Python's and
    torch's global RNGs."""
    check_chains(chains, "OpenFold3", templates=True, upstream=UPSTREAM)
    check_unpaired_msas(chains, "OpenFold3", upstream=UPSTREAM)
    system = jfold_system(chains)
    features = view.features(system, key=key)
    return backend_features(features), AtomSitesWriter(view.atom_sites(system))


def set_binder_sequence(sequence: Float[Array, "N 20"], features: dict) -> dict:
    """`features` with the first `N` tokens (a binder without MSA hits) set to the soft
    `sequence`: residue type, profile and query MSA row; the binder's other MSA rows are
    gaps."""
    num_residues = sequence.shape[0]
    restype = jnp.pad(sequence, ((0, 0), (0, view.NUM_CLASSES - 20)))
    gap = jax.nn.one_hot(view.GAP, view.NUM_CLASSES)
    return features | {
        "restype": features["restype"].at[:num_residues].set(restype),
        "profile": features["profile"].at[:num_residues].set(restype),
        "msa": features["msa"]
        .at[0, :num_residues]
        .set(restype)
        .at[1:, :num_residues]
        .set(gap),
    }


def of3_atom37(features: dict, coords: Float[Array, "A 3"]):
    """Atom37 coordinates and mask of OpenFold3 coordinates (`[A, 3]`), each token's atoms
    named by its residue type's argmax."""
    atom_to_token = features["atom_to_token_index"]
    position = (
        jnp.arange(atom_to_token.shape[0]) - features["start_atom_index"][atom_to_token]
    )
    residue = features["restype"].argmax(-1)[atom_to_token]
    slot = jnp.asarray(_ATOM37_TABLE)[residue, position]
    slot = jnp.where(features["atom_mask"] > 0, slot, -1)
    return scatter_atom37(coords, atom_to_token, slot, features["restype"].shape[0])


class OpenFold3(DiffusionStructurePredictionModel[dict]):
    """OpenFold3 on jfold (see the module docstring): upstream's recycling count for
    predictions, a poly-X placeholder for the binder."""

    model: StructureModel
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    distogram_bins = DISTOGRAM_BINS

    def __init__(
        self,
        *,
        model: StructureModel | None = None,
        design_recycling_steps: int = 4,
        inference_recycling_steps: int = INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
    ):
        """Loads the converted model from mosaic's `jfold` cache unless `model` is passed
        (downloaded, or converted from OpenFold3's torch checkpoint, on first use)."""
        if model is None:
            model = pretrained.load("openfold3", cache=cache_dir("jfold"))
        self.model = model
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    def target_only_features(self, chains: list[TargetChain], *, key: PRNGKeyArray):
        return target_only_features(chains, key=key)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        """A poly-X (UNK) placeholder binder before `chains`: uninformative, so its atoms
        do not contradict the designed sequence."""
        return self.target_only_features(
            [TargetChain(sequence="X" * binder_length, use_msa=False)] + chains, key=key
        )

    def set_sequence(
        self, features: dict, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ) -> dict:
        del key
        return set_binder_sequence(sequence, features)

    def jfold_features(self, features: dict) -> JFeatures:
        return jfold_of3.features(features, rep_atom=features["rep_atom"])

    def tokens(self, features: dict) -> TokenInfo:
        return TokenInfo(
            sequence=features["restype"][:, :20],
            asym_id=features["asym_id"],
            residue_index=features["residue_index"],
            mask=features["token_mask"].astype(bool),
        )

    def atom37(self, features: dict, coords: Float[Array, "A 3"]) -> Atom37:
        atom37_coords, atom37_mask = of3_atom37(features, coords)
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(
        self, features: dict, writer, output: StructureModelOutput, sequence
    ) -> gemmi.Structure:
        del features, sequence
        return writer(output.structure_coordinates)
