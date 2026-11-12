"""Featurize Protenix inputs with the `protenix` package's data pipeline.

The `protenix` package is a fork of upstream Protenix's data pipeline that runs without
torch. Use it for inputs jfold cannot featurize, such as nucleic acids, ligands, ions,
modified residues and paired MSAs.

`features` takes one entry of Protenix's input JSON as a dict. It supports
`proteinChain`, `dnaSequence`, `rnaSequence`, `ligand` and `ion` entities. A ligand is
given as a CCD code such as `"CCD_ATP"` or as a SMILES string. A protein's MSA is given as
`"msa": {"precomputed_msa_dir": ..., "pairing_db": "uniref100"}`, pointing to a directory
with `non_pairing.a3m` and `pairing.a3m` as written by `protenix msa`. A protein without
an `msa` gets an MSA containing only its query.

Protenix's input JSON has no templates, so the template features are empty, as in
upstream's inference. Use `Protenix.target_only_features` for protein templates.

`features` returns the same features as `Protenix.target_only_features`, with reference
conformers randomly placed from a JAX key in the same way, and a structure writer.

For design, make the binder the first entity as poly-Trp of the binder's length, since Trp
has the most atoms of any residue, and convert the features with `Protenix.design_pack`::

    from mosaic.models.protenix import Protenix
    from mosaic.models.upstream import protenix as upstream

    model = Protenix.mini()
    binder_length = 60
    features, writer = upstream.features(
        {
            "name": "design",
            "sequences": [
                {"proteinChain": {"sequence": "W" * binder_length, "count": 1}},
                {
                    "proteinChain": {
                        "sequence": target,
                        "count": 1,
                        "msa": {
                            "precomputed_msa_dir": "msas/target",
                            "pairing_db": "uniref100",
                        },
                    }
                },
                {"ligand": {"ligand": "CCD_ATP", "count": 1}},
                {"ion": {"ion": "MG", "count": 1}},
            ],
        },
        key=jax.random.key(0),
    )
    pack, writer = model.design_pack(features, writer, binder_length)

Featurization reads Protenix's CCD data from `~/.protenix` and downloads it on first use.
"""

import copy
import dataclasses
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import gemmi
import jax.numpy as jnp
import numpy as np
import protenix.openfold_local.np.residue_constants as rc
from biotite.structure import AtomArray
from jaxtyping import Array, PRNGKeyArray
from jfold.featurize.views import protenix as view
from protenix import backend
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.template import Templates, template_featurizer
from protenix.data.utils import data_type_transform, make_dummy_feature

from mosaic.common import TOKENS


def with_query_msas(input: dict, workdir: Path) -> dict:
    """Return a copy of `input` in which proteins without an `msa` get a query-only MSA.

    The MSAs are written to `workdir`."""
    input = copy.deepcopy(input)
    for i, entity in enumerate(input["sequences"]):
        protein = entity.get("proteinChain")
        if protein is not None and "msa" not in protein:
            msa_dir = str(workdir / f"query_{i}")
            template_featurizer._make_dummy_msa(protein["sequence"], msa_dir)
            protein["msa"] = {"precomputed_msa_dir": msa_dir, "pairing_db": "uniref100"}
    return input


def empty_templates(num_tokens: int) -> dict[str, np.ndarray]:
    """Return Protenix's template features for a single empty template."""
    return Templates(
        np.full((1, num_tokens), rc.unk_restype_index, np.int64),
        np.zeros((1, num_tokens, 37, 3), np.float32),
        np.zeros((1, num_tokens, 37), np.float32),
    ).as_protenix_dict()


def posed_conformers(atom_array: AtomArray, key: PRNGKeyArray) -> np.ndarray:
    """Centre each reference conformer and randomly place it using `key`.

    A conformer is a residue or a whole ligand, that is, the atoms sharing a
    `ref_space_uid`. As in jfold and upstream, each one is translated by a uniform random
    offset in [-1, 1]^3 and then randomly rotated."""
    uid = np.asarray(atom_array.ref_space_uid)
    assert np.all(np.diff(uid) >= 0), "a conformer's atoms must be contiguous"
    _, space = np.unique(uid, return_inverse=True)
    conformers = np.split(atom_array.ref_pos, np.flatnonzero(np.diff(space)) + 1)
    centred = np.concatenate([c - c.mean(axis=0) for c in conformers])
    translation, rotation = view.random_pose(key, len(conformers))
    return view.pose(centred, space, translation, rotation, translate_first=True)


def features(
    input: dict, *, key: PRNGKeyArray
) -> tuple[dict[str, Array], "AtomArrayWriter"]:
    """Featurize one entry of Protenix's input JSON and return the features and a writer.

    `key` randomly places the reference conformers."""
    backend.download_data()  # downloads the CCD if missing
    with TemporaryDirectory() as tmp:
        sample = with_query_msas(input, Path(tmp))
        f, atom_array, token_array = SampleDictToFeatures(sample).get_feature_dict()
        entity_asym_ids = defaultdict(set)
        for entity, asym in zip(atom_array.label_entity_id, atom_array.asym_id_int):
            entity_asym_ids[entity].add(asym)
        msa_features = InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=sample["sequences"],
            entity_to_asym_id=dict(entity_asym_ids),
            token_array=token_array,
            atom_array=atom_array,
        )
    f["distogram_rep_atom_mask"] = np.asarray(
        atom_array.distogram_rep_atom_mask, np.int64
    )
    if msa_features:
        f |= msa_features
    else:  # without proteins, use a query-only MSA as upstream's `InferenceDataset` does
        f = make_dummy_feature(f, dummy_feats=["msa"])
    f = data_type_transform(f)
    f |= empty_templates(len(f["asym_id"]))
    f["atom_rep_atom_idx"] = np.flatnonzero(f["distogram_rep_atom_mask"])
    f["ref_pos"] = posed_conformers(atom_array, key)
    keys = view.JFOLD_KEYS + view.EXTRA_KEYS
    return {k: jnp.asarray(f[k]) for k in keys}, AtomArrayWriter(atom_array)


@dataclasses.dataclass(frozen=True)
class Atoms:
    """Atom annotations from upstream's `AtomArray`, used to write structures."""

    chain_id: np.ndarray
    res_id: np.ndarray
    res_name: np.ndarray
    atom_name: np.ndarray
    element: np.ndarray
    hetero: np.ndarray

    @classmethod
    def of(cls, atom_array: AtomArray) -> "Atoms":
        return cls(
            **{
                field.name: np.asarray(atom_array.get_annotation(field.name))
                for field in dataclasses.fields(cls)
            }
        )

    def __len__(self) -> int:
        return len(self.chain_id)

    def __getitem__(self, index) -> "Atoms":
        return Atoms(**{k: v[index] for k, v in dataclasses.asdict(self).items()})

    def __add__(self, other: "Atoms") -> "Atoms":
        a, b = dataclasses.asdict(self), dataclasses.asdict(other)
        return Atoms(**{k: np.concatenate([a[k], b[k]]) for k in a})

    def structure(self, coords: np.ndarray) -> gemmi.Structure:
        """Build a `gemmi.Structure` from per-atom `coords` of shape `[A, 3]`.

        Polymer residues are written as ATOM records and ligands and ions as HETATM."""
        chains: dict[str, dict[int, gemmi.Residue]] = {}
        for i in range(len(self)):
            residues = chains.setdefault(str(self.chain_id[i]), {})
            residue = residues.get(int(self.res_id[i]))
            if residue is None:
                residue = residues[int(self.res_id[i])] = gemmi.Residue()
                residue.name = str(self.res_name[i])
                residue.seqid = gemmi.SeqId(int(self.res_id[i]), " ")
                residue.het_flag = "H" if self.hetero[i] else "A"
            atom = gemmi.Atom()
            atom.name = str(self.atom_name[i])
            atom.element = gemmi.Element(str(self.element[i]))
            atom.pos = gemmi.Position(*map(float, coords[i]))
            residue.add_atom(atom)
        model = gemmi.Model("1")
        for chain_id, residues in chains.items():
            chain = gemmi.Chain(chain_id)
            chain.append_residues(list(residues.values()))
            model.add_chain(chain)
        st = gemmi.Structure()
        st.add_model(model)
        st.setup_entities()
        return st


def protein_atoms(sequence: str) -> Atoms:
    """Return the atoms that upstream featurizes for a protein chain of `sequence`."""
    sample = {
        "name": "protein",
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
    }
    return Atoms.of(SampleDictToFeatures(sample).get_atom_array())


class AtomArrayWriter:
    """Convert coordinates of shape `[A, 3]` to a `gemmi.Structure` using upstream's atoms.

    Padding atoms must come after the real ones. For binder design, the first chain is
    written as the argmax of `binder_pssm`, which has shape `[N, 20]`."""

    def __init__(self, atom_array: AtomArray):
        self.atoms = Atoms.of(atom_array)

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        atoms = self.atoms
        if binder_pssm is not None:
            binder = "".join(TOKENS[i] for i in np.asarray(binder_pssm).argmax(-1))
            designed = protein_atoms(binder)
            designed = dataclasses.replace(
                designed, chain_id=np.full(len(designed), atoms.chain_id[0])
            )
            atoms = designed + atoms[atoms.chain_id != atoms.chain_id[0]]
        coords = np.asarray(coords).reshape(-1, 3)[: len(atoms)]
        return atoms.structure(coords)
