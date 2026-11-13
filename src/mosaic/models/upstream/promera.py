"""Featurize ProMERA inputs with its upstream featurizer, `tinyprot`.

This runs `tinyprot` as ProMERA's inference does, for inputs jfold cannot featurize, such
as nucleic acids or MSAs paired across chains. `features` takes the schema that tinyprot's
`Structure.from_schema` reads, mapping chain ids to entities, for example:

    {"A": {"type": "protein", "sequence": "MKTAYIAKQR", "use_msa": False},
     "B": {"type": "rna", "sequence": "GGACUUCGGUCC"},
     "C": {"type": "dna", "sequence": "GCGATCGC"}}

MSAs are read from a directory of a3m files named by `tinyprot.msa.hash_sequence` of each
chain's sequence, and are paired across chains by species. A chain with `use_msa` False
gets a single-row MSA, and a protein chain with `use_msa` True must have an a3m file.
`features` returns the same features as `Promera.target_only_features`, with a structure
writer.

For binder design, put the binder first as a protein chain of X without an MSA, as
`Promera.binder_features` does, and convert the features with `Promera.design_pack`:

    from mosaic.models.promera import Promera
    from mosaic.models.upstream import promera as upstream

    model = Promera()
    schema = {
        "A": {"type": "protein", "sequence": "X" * 80, "use_msa": False},
        "B": {"type": "protein", "sequence": target},
        "C": {"type": "dna", "sequence": "GCGATCGC"},
    }
    features, writer = model.design_pack(
        *upstream.features(schema, key=key, msa_dir="msas"), binder_length=80
    )
    prediction = model.predict(PSSM=pssm, features=features, writer=writer, key=key)

`tinyprot` uses only numpy. ProMERA's `finalize_feats` and `collate` are torch code, so
they are reimplemented here.
"""

import copy
import os
import tempfile

import gemmi
import numpy as np
from jaxtyping import PRNGKeyArray
from jfold.featurize.padding import pad_to, padded_length

from mosaic.common import TOKENS
from mosaic.models.promera import PromeraFeatures, promera_features
from mosaic.models.upstream import global_rngs

# The upstream features that jfold's featurizer also returns: those that
# `jfold.models.promera.features` reads, plus `frames_mask` and `msa_paired`.
KEYS = (
    "restype",
    "profile",
    "deletion_mean",
    "is_std",
    "is_epitope",
    "token_pad_mask",
    "token_to_rep_atom",
    "token_bonds",
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "frames_mask",
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_hydrogens",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_to_token",
    "atom_pad_mask",
    "atom_is_std",
    "msa",
    "msa_mask",
    "msa_paired",
    "has_deletion",
    "deletion_value",
)
# Per-atom features, which `collate` pads to a multiple of `ATOM_MULTIPLE` atoms.
ATOM_KEYS = (
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_hydrogens",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_to_token",
    "atom_pad_mask",
    "atom_is_std",
)
ATOM_MULTIPLE = 32


def ensure_data() -> None:
    """Download tinyprot's CCD and taxonomy databases to `$TINYPROT_CACHE` if missing.

    The taxonomy database is about 10 GB, and `tinyprot.msa` opens it on import."""
    from tinyprot import init

    for path, url, name in (
        (init._ccd_path, init._HF_CCD_URL, "CCD"),
        (init._tax_path, init._HF_TAX_URL, "taxonomy"),
    ):
        database = os.path.join(path, "data.mdb")
        if not os.path.exists(database):
            init._download_file(url, database, f"Downloading tinyprot's {name}")


def finalize(feats: dict) -> dict[str, np.ndarray]:
    """Finalize tinyprot's features as ProMERA's `finalize_feats` and `collate` do.

    Return the `KEYS` features for a single example, without a batch axis."""
    num_tokens = len(feats["restype"])
    bonds = np.zeros((num_tokens, num_tokens), np.int64)
    bonds[tuple(feats["token_bonds"].T)] = 1
    bonds[tuple(feats["token_bonds"].T[::-1])] = 1
    feats = feats | {
        "atom_pad_mask": np.ones_like(feats["ref_element"]),
        "token_pad_mask": np.ones_like(feats["restype"]),
        "is_epitope": np.zeros_like(feats["restype"]),
        "token_bonds": bonds,
        "atom_is_std": feats["is_std"][feats["atom_to_token"]],
    }
    num_atoms = padded_length(len(feats["ref_pos"]), ATOM_MULTIPLE)
    out = {}
    for key in KEYS:
        value = np.asarray(feats[key])
        if key in ATOM_KEYS:
            value = pad_to(value, num_atoms)
        if value.dtype.kind == "f":
            value = value.astype(np.float32)
        elif value.dtype.kind in "iu":
            value = value.astype(np.int64)
        out[key] = value
    return out


def features(
    input: dict, *, key: PRNGKeyArray, msa_dir: str | os.PathLike | None = None
) -> tuple[PromeraFeatures, "StructureWriter"]:
    """Featurize the tinyprot schema `input` and return the features and a writer.

    `msa_dir` is the directory of a3m files. If None, tinyprot's default
    `$TINYPROT_CACHE/msa` is used. `key` seeds numpy's global RNG, which upstream uses to
    randomly place the reference conformers."""
    ensure_data()
    from tinyprot.feature import AF3Featurizer
    from tinyprot.msa import construct_paired_msa, load_msa_from_dir, make_dummy_msa
    from tinyprot.structure import Structure

    struct = Structure.from_schema(input)
    msas = load_msa_from_dir(
        None if msa_dir is None else os.fspath(msa_dir), struct.chains
    )
    for chain_id, chain in struct.chains.items():
        use_msa = input[chain_id].get("use_msa")
        if use_msa is False:
            msas[chain_id] = make_dummy_msa(chain.seq)
        elif use_msa and "polypeptide" in chain.type and msas[chain_id].path is None:
            raise ValueError(f"chain {chain_id} takes an MSA, but has no a3m")
    pairing = construct_paired_msa(msas)
    with global_rngs(key):
        feats = AF3Featurizer(struct, msas, pairing).featurize(compute_frames=True)
    return promera_features(finalize(feats)), StructureWriter(struct)


class StructureWriter:
    """Convert coordinates of shape `[A, 3]` to a `gemmi.Structure` with tinyprot's writer.

    Padding atoms must come after the real ones. For binder design, the first chain is
    written as the argmax of `binder_pssm`, which has shape `[N, 20]`."""

    def __init__(self, struct):
        self.struct = struct

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        from tinyprot.chain import Chain

        struct = copy.deepcopy(self.struct)
        if binder_pssm is not None:
            binder_id, placeholder = next(iter(struct.chains.items()))
            sequence = "".join(TOKENS[i] for i in np.asarray(binder_pssm).argmax(-1))
            binder = Chain.from_sequence(sequence, placeholder.type)
            binder.asym_id = placeholder.asym_id
            binder.entity_id = placeholder.entity_id
            struct.chains[binder_id] = binder
        # Fill atoms in tinyprot's order, as `promera.inference.cofolding` does.
        coords = iter(np.asarray(coords).reshape(-1, 3))
        for chain in struct.chains.values():
            for j, names in enumerate(chain.aname):
                for k, name in enumerate(names):
                    if name != "":
                        chain.coords[j, k] = next(coords)
                        chain.mask[j, k] = True
        with tempfile.NamedTemporaryFile(suffix=".cif") as f:
            struct.to_mmcif(f.name, metadata=False)
            return gemmi.read_structure(f.name)
