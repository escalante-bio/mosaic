"""ProMERA on jfold: featurized by `jfold.featurize` (torch-free; the same dict upstream
ProMERA builds, up to the random pose of each residue's reference conformer), with jfold's
converted ProMERA (`jfold.pretrained`) for the forward pass.

Converting needs the upstream `promera` package (with CUDA torch), which mosaic does not
install. Convert once from jfold's environment into mosaic's cache,

    jfold.pretrained.load("promera", cache="~/.cache/mosaic/jfold")

(`$MOSAIC_CACHE_DIR/jfold` if set); mosaic then only loads the converted copy, which
needs neither torch nor `promera`.

jfold's view takes protein chains of the 20 amino acids and X, without templates; MSA
rows are ordered by species, read from tinyprot's taxonomy database (`uniref_taxonomy`).
Other systems (DNA and RNA chains, MSAs paired across chains) go through ProMERA's own
featurizer, `mosaic.models.upstream.promera.features`, on tinyprot's own input; its
features feed the same model, and `Promera.design_pack` makes them design features.
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
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.models.jfold_featurize import (
    BinderAtomSitesWriter,
    check_chains,
    jfold_system,
    residue_type_features,
)
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
)

DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
# Trunk passes for design (`model_output`, losses) and for `predict` (upstream's
# `inference/cofolding.yaml`).
DESIGN_RECYCLING_STEPS = 3
INFERENCE_RECYCLING_STEPS = 4
# The amino acids' token types are 2..21 (`TOKENS` order); UNK's is 22.
FIRST_AMINO_ACID = 2
UNK = view.TOKEN["X"]
# The atom fields a binder residue's type determines.
RESIDUE_FIELDS = (
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_hydrogens",
    "ref_atom_name_chars",
    "atom_is_std",
)


def load_model() -> StructureModel:
    """The converted ProMERA from mosaic's `jfold` cache."""
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
    """UniRef100 id to NCBI taxonomy id (-1 if unknown), as upstream reads a3m headers:
    the view orders MSA rows by it (`view.pair_rows`). The lookup is tinyprot's LMDB
    database (~10 GB, under `$TINYPROT_CACHE`, downloaded on first use)."""
    from tinyprot import init

    database = os.path.join(init._tax_path, "data.mdb")
    if not os.path.exists(database):
        init._download_file(init._HF_TAX_URL, database, "Downloading taxonomy")
    from tinyprot.msa import _get_taxonomy

    return _get_taxonomy


def atom37_index(features: dict) -> np.ndarray:
    """Each atom's atom37 slot, from its name (-1 for padding and names outside atom37)."""
    names = np.asarray(features["ref_atom_name_chars"]).tolist()
    return np.array(
        [ATOM37_INDEX.get("".join(chr(c + 32) for c in n if c), -1) for n in names],
        np.int32,
    )


class BinderPack(eqx.Module):
    """What `set_sequence` rebuilds the atoms from: each residue type's atoms (`residues`,
    `[21, R, ...]`: the 20 amino acids in `TOKENS` order, then UNK, each padded to `R`
    atoms) and the target's (`target`: the featurized atoms after the binder's, with
    their `atom_to_token`), as `RESIDUE_FIELDS` and `atom37_index`; each residue type's
    and target token's representative atom; and `num_atoms`, the rebuilt atom axis (room
    for any binder, padded)."""

    residues: dict[str, Array]
    residue_num_atoms: Int[Array, " 21"]
    residue_rep_atom: Int[Array, " 21"]
    target: dict[str, Array]
    target_rep_atom: Int[Array, " T"]
    num_atoms: int = eqx.field(static=True)


def residue_types() -> list[dict[str, np.ndarray]]:
    """The view's features of each residue type alone: the 20 amino acids (`TOKENS`
    order), then UNK."""
    unk = view.features(System((Chain("A", "X"),)))
    return [*residue_type_features(view.features).values(), unk]


def binder_pack(features: dict, binder_len: int) -> BinderPack:
    """The design pack for `features` (from the view) whose first `binder_len` tokens are
    the binder."""
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
    """The atom fields (and `atom37_index`) for a binder of `residue_type` (rows of
    `pack.residues`) followed by the target, the binder's atoms in token order."""
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
    """ProMERA's features: the view's dict (`restype` one-hot; after `set_sequence`, the
    binder's soft sequence in `restype`, `profile` and the query row of `msa`, and its
    atoms rebuilt from the sequence's argmax), each atom's atom37 slot, and for binder
    design the design pack."""

    features: dict[str, Array]
    atom37_index: Int[Array, " A"]
    pack: BinderPack | None = None


UPSTREAM = (
    "mosaic.models.upstream.promera.features featurizes tinyprot's own schema "
    "(install mosaic[promera])"
)


def promera_features(features: dict[str, np.ndarray]) -> PromeraFeatures:
    """`PromeraFeatures` (without design pack) from ProMERA's feature dict, the view's or
    upstream's."""
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
    """`chains` by jfold's view, and a writer. `key` poses the reference conformers, as
    upstream does with numpy's global RNG."""
    check_chains(chains, "ProMERA", templates=False, upstream=UPSTREAM)
    system = jfold_system(chains)
    taxonomy = uniref_taxonomy() if any(c.use_msa for c in chains) else None
    features = view.features(system, key=key, taxonomy=taxonomy)
    return promera_features(features), BinderAtomSitesWriter(system, view.atom_sites)


class Promera(DiffusionStructurePredictionModel[PromeraFeatures]):
    """ProMERA on jfold, with upstream's recycling counts. `set_sequence` masks
    `mask_fraction` of the binder to UNK. Each trunk pass reads a random subset of the
    converted model's `msas_per_trunk_iter` (1024) MSA rows, the query row always kept.
    The two featurizations: see the module docstring."""

    model: StructureModel
    # Fraction of binder tokens masked to UNK each step (design losses only).
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
        """`key` poses the reference conformers."""
        return view_features(list(chains), key=key)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        """`key` poses the reference conformers."""
        binder = TargetChain(sequence="X" * binder_length, use_msa=False)
        return self.design_pack(
            *view_features([binder, *chains], key=key), binder_length
        )

    def design_pack(
        self, features: PromeraFeatures, writer, binder_length: int
    ) -> tuple[PromeraFeatures, Callable[..., gemmi.Structure]]:
        """Binder-design features from target-only `features` (jfold's view or
        `mosaic.models.upstream.promera.features`) whose first chain is the binder's
        placeholder, `binder_length` residues of X: `features` with the binder's design
        pack, which `set_sequence` reads, and `writer`, which writes the binder as
        `argmax(PSSM)` and every other chain as featurized."""
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
        """Set the binder's sequence channels and rebuild its atoms from
        `argmax(sequence)`, masking `mask_fraction` of it to UNK."""
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
