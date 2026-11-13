"""ProMERA backend running on jfold.

Inputs are featurized with `jfold.featurize`, which builds the same features as upstream
ProMERA apart from the random placement of reference conformers. The forward pass uses
jfold's converted ProMERA weights from `jfold.pretrained`.

Converting the checkpoint needs the upstream `promera` package and CUDA torch, which mosaic
does not install. Convert it once from jfold's environment into mosaic's cache:

    jfold.pretrained.load("promera", cache="~/.cache/mosaic/jfold")

Use `$MOSAIC_CACHE_DIR/jfold` instead if that variable is set. Loading the converted
checkpoint needs neither torch nor `promera`.

jfold supports protein chains of the 20 amino acids and X, without templates. MSA rows are
ordered by species using tinyprot's taxonomy database. Other inputs, such as nucleic acids
or MSAs paired across chains, can be featurized from tinyprot's input with
`mosaic.models.upstream.promera.features`.
"""

import os
from collections.abc import Callable

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jfold import pretrained
from jfold.blocks.inputs import probabilities
from jfold.featurize import Chain, System
from jfold.featurize.padding import padded_length
from jfold.featurize.views import promera as view
from jfold.model import StructureModel
from jfold.models import promera as jfold_promera
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

DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
# Number of trunk passes for design and for `predict`, which matches upstream's
# `inference/cofolding.yaml`.
DESIGN_RECYCLING_STEPS = 3
INFERENCE_RECYCLING_STEPS = 4
# The amino acids have token types 2 to 21 in `TOKENS` order, and UNK has 22.
FIRST_AMINO_ACID = 2
UNK = view.TOKEN["X"]
# Per-atom features that depend on a binder residue's type.
RESIDUE_FIELDS = (
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_hydrogens",
    "ref_atom_name_chars",
    "atom_is_std",
)


def load_model() -> StructureModel:
    """Load the converted ProMERA checkpoint from mosaic's `jfold` cache."""
    cache = cache_dir("jfold")
    try:
        return pretrained.load("promera", cache=cache)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "No converted ProMERA in mosaic's cache, and converting it needs the upstream "
            "`promera` package. Convert it once from jfold's environment: "
            f'`jfold.pretrained.load("promera", cache="{cache}")`.'
        ) from error


def uniref_taxonomy() -> Callable[[str], int]:
    """Return a function mapping a UniRef100 id to its NCBI taxonomy id, or -1.

    jfold uses it to order MSA rows by species. The lookup uses tinyprot's database of
    about 10 GB, which is downloaded to `$TINYPROT_CACHE` on first use."""
    from tinyprot import init

    database = os.path.join(init._tax_path, "data.mdb")
    if not os.path.exists(database):
        init._download_file(init._HF_TAX_URL, database, "Downloading taxonomy")
    from tinyprot.msa import _get_taxonomy

    return _get_taxonomy


def atom37_index(features: dict) -> np.ndarray:
    """Return each atom's atom37 index from its name, or -1 if it has none."""
    names = np.asarray(features["ref_atom_name_chars"]).tolist()
    return np.array(
        [ATOM37_INDEX.get("".join(chr(c + 32) for c in n if c), -1) for n in names],
        np.int32,
    )


class BinderPack(eqx.Module):
    """Data that `set_sequence` uses to rebuild the binder's atoms.

    `residues` holds the atom features of each residue type, with shape `[21, R, ...]`:
    the 20 amino acids in `TOKENS` order and then UNK, each padded to `R` atoms. `target`
    holds the target's atom features. `num_atoms` is the padded size of the atom axis,
    large enough for any binder sequence."""

    residues: dict[str, Array]
    residue_num_atoms: Int[Array, " 21"]
    residue_rep_atom: Int[Array, " 21"]
    target: dict[str, Array]
    target_rep_atom: Int[Array, " T"]
    num_atoms: int = eqx.field(static=True)


def residue_types() -> list[dict[str, np.ndarray]]:
    """Featurize each residue type on its own: the 20 amino acids, then UNK."""
    unk = view.features(System((Chain("A", "X"),)))
    return [*residue_type_features(view.features).values(), unk]


def binder_pack(features: dict, binder_len: int) -> BinderPack:
    """Build a `BinderPack` for features whose first `binder_len` tokens are the binder."""
    types = residue_types()
    num_atoms = np.array([int(f["atom_pad_mask"].sum()) for f in types])
    width = int(num_atoms.max())
    residues = {k: np.stack([f[k][:width] for f in types]) for k in RESIDUE_FIELDS} | {
        "atom37_index": np.stack([atom37_index(f)[:width] for f in types])
    }

    atom_to_token = features["atom_to_token"]
    real = np.flatnonzero(features["atom_pad_mask"])
    start = int((atom_to_token[real] < binder_len).sum())
    target = slice(start, len(real))
    target_fields = (*RESIDUE_FIELDS, "atom_to_token")
    return BinderPack(
        residues=jax.tree.map(jnp.asarray, residues),
        residue_num_atoms=jnp.asarray(num_atoms),
        residue_rep_atom=jnp.asarray([int(f["token_to_rep_atom"][0]) for f in types]),
        target=jax.tree.map(
            jnp.asarray,
            {k: features[k][target] for k in target_fields}
            | {"atom37_index": atom37_index(features)[target]},
        ),
        target_rep_atom=jnp.asarray(features["token_to_rep_atom"][binder_len:] - start),
        num_atoms=padded_length(
            binder_len * width + target.stop - start, view.ATOM_MULTIPLE
        ),
    )


def binder_atoms(pack: BinderPack, residue_type: Int[Array, " B"]) -> dict[str, Array]:
    """Build the atom features for a binder with residue types `residue_type` followed by
    the target."""
    num_binder, width = residue_type.shape[0], pack.residues["ref_pos"].shape[1]
    counts = pack.residue_num_atoms[residue_type]
    start = jnp.cumsum(counts) - counts
    target_start = counts.sum()
    slot = jnp.arange(width)
    binder_index = jnp.where(
        slot < counts[:, None], start[:, None] + slot, pack.num_atoms
    ).reshape(-1)
    target_index = target_start + jnp.arange(pack.target["ref_pos"].shape[0])

    def place(binder, target, fill=0):
        binder = binder.reshape(num_binder * width, *binder.shape[2:])
        out = jnp.full((pack.num_atoms + 1, *binder.shape[1:]), fill, binder.dtype)
        out = out.at[binder_index].set(binder).at[target_index].set(target)
        return out[:-1]

    atoms = {
        k: place(pack.residues[k][residue_type], pack.target[k]) for k in RESIDUE_FIELDS
    }
    token = jnp.broadcast_to(jnp.arange(num_binder)[:, None], (num_binder, width))
    atoms["atom_to_token"] = place(token, pack.target["atom_to_token"])
    atoms["ref_space_uid"] = atoms["atom_to_token"]
    atoms["atom_pad_mask"] = (
        jnp.arange(pack.num_atoms) < target_start + target_index.shape[0]
    ).astype(jnp.int32)
    atoms["token_to_rep_atom"] = jnp.concatenate(
        [
            start + pack.residue_rep_atom[residue_type],
            target_start + pack.target_rep_atom,
        ]
    )
    atoms["atom37_index"] = place(
        pack.residues["atom37_index"][residue_type], pack.target["atom37_index"], -1
    )
    return atoms


class PromeraFeatures(eqx.Module):
    """ProMERA input features.

    `features` is the feature dict with `restype` one-hot encoded. `atom37_index` holds
    each atom's atom37 index. For binder design, `pack` holds the data needed to rebuild
    the binder's atoms."""

    features: dict[str, Array]
    atom37_index: Int[Array, " A"]
    pack: BinderPack | None = None


UPSTREAM = (
    "mosaic.models.upstream.promera.features featurizes tinyprot's own schema "
    "(install mosaic[all])"
)


def promera_features(features: dict[str, np.ndarray]) -> PromeraFeatures:
    """Wrap a feature dict from jfold or upstream as `PromeraFeatures`."""
    features = features | {
        "restype": np.eye(view.NUM_TOKEN_TYPES, dtype=np.float32)[features["restype"]]
    }
    return PromeraFeatures(
        features=jax.tree.map(jnp.asarray, features),
        atom37_index=jnp.asarray(atom37_index(features)),
    )


def view_features(
    chains: list[TargetChain], *, key: PRNGKeyArray
) -> tuple[PromeraFeatures, BinderAtomSitesWriter]:
    """Featurize `chains` with jfold and return the features with a structure writer.

    `key` randomly rotates and translates each residue's reference conformer, as upstream
    does with numpy's global RNG."""
    check_chains(chains, "ProMERA", templates=False, upstream=UPSTREAM)
    system = jfold_system(chains)
    taxonomy = uniref_taxonomy() if any(c.use_msa for c in chains) else None
    features = view.features(system, key=key, taxonomy=taxonomy)
    return promera_features(features), BinderAtomSitesWriter(system, view.atom_sites)


class Promera(DiffusionStructurePredictionModel[PromeraFeatures]):
    """ProMERA with upstream's recycling counts.

    `set_sequence` randomly masks a `mask_fraction` of the binder residues to UNK. Each
    trunk pass samples up to 1024 MSA rows, always including the query."""

    model: StructureModel
    # Fraction of binder tokens masked to UNK at each design step.
    mask_fraction: float | Array
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    distogram_bins = DISTOGRAM_BINS

    def __init__(
        self,
        model: StructureModel | None = None,
        mask_fraction=0.0,
        *,
        design_recycling_steps: int = DESIGN_RECYCLING_STEPS,
        inference_recycling_steps: int = INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
    ):
        self.model = load_model() if model is None else model
        self.mask_fraction = mask_fraction
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    def target_only_features(self, chains: list[TargetChain], *, key: PRNGKeyArray):
        """Featurize `chains`. `key` randomly places the reference conformers."""
        return view_features(list(chains), key=key)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        """Featurize a binder of X followed by `chains` for binder design.

        `key` randomly places the reference conformers."""
        binder = TargetChain(sequence="X" * binder_length, use_msa=False)
        return self.design_pack(
            *view_features([binder, *chains], key=key), binder_length
        )

    def design_pack(
        self, features: PromeraFeatures, writer, binder_length: int
    ) -> tuple[PromeraFeatures, Callable[..., gemmi.Structure]]:
        """Turn features whose first chain is `binder_length` residues of X into design
        features.

        The features can come from jfold or from `mosaic.models.upstream.promera.features`.
        Setting a sequence on the result rebuilds the binder's atoms, and `writer` then
        writes the binder as `argmax(PSSM)`."""
        f = jax.tree.map(np.asarray, features.features)
        restype, asym_id = f["restype"].argmax(-1), f["asym_id"]
        binder = asym_id == asym_id[0]
        if not (
            0 < binder_length <= len(restype)
            and binder[:binder_length].all()
            and not binder[binder_length:].any()
            and (restype[:binder_length] == UNK).all()
        ):
            raise ValueError(
                f"design_pack needs features whose first chain is the binder's "
                f"placeholder, a protein chain of {binder_length} X"
            )
        pack = binder_pack(f, binder_length)
        return PromeraFeatures(features.features, features.atom37_index, pack), writer

    def set_sequence(
        self,
        features: PromeraFeatures,
        sequence: Float[Array, "N 20"],
        *,
        key: PRNGKeyArray,
    ) -> PromeraFeatures:
        """Set the binder's sequence and rebuild its atoms from `argmax(sequence)`.

        A random `mask_fraction` of the binder residues is set to UNK."""
        if features.pack is None:
            raise ValueError("set_sequence needs features from binder_features")
        num_binder = sequence.shape[0]
        masked = jax.random.uniform(key, (num_binder,)) < self.mask_fraction
        soft = jnp.pad(
            sequence,
            ((0, 0), (FIRST_AMINO_ACID, view.NUM_TOKEN_TYPES - FIRST_AMINO_ACID - 20)),
        )
        soft = jnp.where(
            masked[:, None], jax.nn.one_hot(UNK, view.NUM_TOKEN_TYPES), soft
        )
        residue_type = jnp.where(
            masked, len(TOKENS), jnp.argmax(jax.lax.stop_gradient(sequence), -1)
        )
        f = features.features
        msa = probabilities(f["msa"], view.NUM_TOKEN_TYPES)
        atoms = binder_atoms(features.pack, residue_type)
        return PromeraFeatures(
            features=f
            | {
                "restype": f["restype"].at[:num_binder].set(soft),
                "profile": f["profile"].at[:num_binder].set(soft),
                "msa": msa.at[0, :num_binder].set(soft),
            }
            | {k: v for k, v in atoms.items() if k != "atom37_index"},
            atom37_index=atoms["atom37_index"],
            pack=features.pack,
        )

    def jfold_features(self, features: PromeraFeatures) -> JFeatures:
        return jfold_promera.features(features.features)

    def tokens(self, features: PromeraFeatures) -> TokenInfo:
        f = features.features
        return TokenInfo(
            sequence=f["restype"][:, FIRST_AMINO_ACID : FIRST_AMINO_ACID + 20],
            asym_id=f["asym_id"],
            residue_index=f["residue_index"],
            mask=f["token_pad_mask"].astype(bool),
        )

    def atom37(self, features: PromeraFeatures, coords: Float[Array, "A 3"]) -> Atom37:
        f = features.features
        atom37_coords, atom37_mask = scatter_atom37(
            coords,
            f["atom_to_token"],
            features.atom37_index,
            f["restype"].shape[0],
        )
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(
        self, features, writer, output: StructureModelOutput, sequence
    ) -> gemmi.Structure:
        del features
        return writer(output.structure_coordinates, binder_pssm=sequence)
