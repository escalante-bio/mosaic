"""Protenix featurized by protenix (`protenix.data`: our fork of upstream Protenix's data
pipeline, which runs without torch) from Protenix's own input: for what
`jfold.featurize`'s view does not take, i.e. DNA and RNA chains, ligands and ions,
modified residues, and paired MSAs.

`features` takes one entry of Protenix's input JSON, as a dict:

- `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand` (a CCD code, `"CCD_ATP"`, or a
  SMILES string) and `ion` entities, each with its `count`;
- a protein's MSA as the input JSON gives it: `"msa": {"precomputed_msa_dir": ...,
  "pairing_db": "uniref100"}`, a directory with `non_pairing.a3m` (unpaired rows) and
  `pairing.a3m` (rows Protenix pairs across chains by species), as Protenix's MSA search
  (`protenix msa`, `protenix.runner.msa_search.update_seq_msa`) writes them. A protein
  without an `msa` gets the query alone (upstream requires one on every protein).

Protenix's input JSON takes no templates: the template slot is empty, as in upstream's
inference. (jfold's view, `Protenix.target_only_features`, takes templates of protein
chains.)

It returns the jfold backend's feature dict (as `Protenix.target_only_features` returns
it): upstream's features (as numpy, then JAX) on the keys jfold's view gives, each
reference conformer (a residue, or a whole ligand) posed from a JAX key as the view poses
them; and a writer of every entity (polymers as ATOM records, ligands and ions as HETATM).

To design, give the binder as the input's first entity, a Trp placeholder of the binder's
length (`"W" * L`, the most atoms of any residue), and make the features a design pack with
`Protenix.design_pack`::

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

Featurization reads Protenix's CCD data from `~/.protenix`, downloaded there on first use.
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
    """A copy of `input` whose proteins without an `msa` have the query alone, written to
    `workdir` as a precomputed MSA."""
    input = copy.deepcopy(input)
    for i, entity in enumerate(input["sequences"]):
        protein = entity.get("proteinChain")
        if protein is not None and "msa" not in protein:
            msa_dir = str(workdir / f"query_{i}")
            template_featurizer._make_dummy_msa(protein["sequence"], msa_dir)
            protein["msa"] = {"precomputed_msa_dir": msa_dir, "pairing_db": "uniref100"}
    return input


def empty_templates(num_tokens: int) -> dict[str, np.ndarray]:
    """Protenix's one template slot without a template (`Templates.as_protenix_dict`):
    UNK and no atoms on every token."""
    return Templates(
        np.full((1, num_tokens), rc.unk_restype_index, np.int64),
        np.zeros((1, num_tokens, 37, 3), np.float32),
        np.zeros((1, num_tokens, 37), np.float32),
    ).as_protenix_dict()


def posed_conformers(atom_array: AtomArray, key: PRNGKeyArray) -> np.ndarray:
    """Each reference conformer (a residue, or a whole ligand: atoms sharing a
    `ref_space_uid`) centred and posed from `key`, as jfold's view poses residues
    (`random_pose`, `pose`) and upstream from numpy's global RNG: translated by
    U(-1, 1)^3, then rotated uniformly."""
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
    """`input` (one entry of Protenix's input JSON) featurized as
    `Protenix.target_only_features` featurizes (`key` poses the reference conformers),
    and the writer of its entities."""
    backend.download_data()  # the CCD
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
    else:  # no protein: the query alone, as upstream's `InferenceDataset`
        f = make_dummy_feature(f, dummy_feats=["msa"])
    f = data_type_transform(f)
    f |= empty_templates(len(f["asym_id"]))
    f["atom_rep_atom_idx"] = np.flatnonzero(f["distogram_rep_atom_mask"])
    f["ref_pos"] = posed_conformers(atom_array, key)
    keys = view.JFOLD_KEYS + view.EXTRA_KEYS
    return {k: jnp.asarray(f[k]) for k in keys}, AtomArrayWriter(atom_array)


@dataclasses.dataclass(frozen=True)
class Atoms:
    """What each atom is, for writing (annotations of upstream's `AtomArray`)."""

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
        """`coords` (`[A, 3]`, one per atom) as a `gemmi.Structure`: polymer residues as
        ATOM records, ligands and ions as HETATM."""
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
    """The atoms upstream featurizes for a protein chain of `sequence`."""
    sample = {
        "name": "protein",
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
    }
    return Atoms.of(SampleDictToFeatures(sample).get_atom_array())


class AtomArrayWriter:
    """Writes coordinates `[A, 3]` (padding atoms after the real ones) as a
    `gemmi.Structure` with every entity of upstream's `atom_array`. For a design pack's
    coordinates, whose binder (the first chain) carries the atoms of its sequence's argmax
    (`refresh_binder_atoms`), `binder_pssm` (`[N, 20]`) gives that sequence; without it,
    the entities are written as featurized."""

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
