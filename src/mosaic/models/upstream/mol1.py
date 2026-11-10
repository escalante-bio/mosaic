"""Mol-1 featurized by molence (our fork of upstream Mol-1; `mol.data`, torch) from Mol-1's
own input: one entry of its input JSON as a dict, so whatever Mol-1 takes (DNA and RNA
chains, ligands, MSAs as a3m text or files, template JSON, ...) featurizes as `mol pred`
does. `features` returns the jfold backend's pack (as `Mol1.target_only_features`) and a
writer of every entity.

To design, put the binder first as the all-Trp placeholder (`"W" * binder_length`, the
most atoms per residue) and turn the pack into a design pack with `Mol1.design_pack`:

    from mosaic.models.mol1 import Mol1
    from mosaic.models.upstream import mol1 as upstream

    model = Mol1()
    features, writer = upstream.features(
        {
            "name": "design",
            "sequences": [
                {"proteinChain": {"sequence": "W" * 80, "count": 1}},
                {"proteinChain": {
                    "sequence": target, "count": 1, "unpairedMsaPath": "target.a3m"
                }},
                {"dnaSequence": {"sequence": "GACTCG", "count": 1}},
            ],
        },
        key=key,
    )
    design, writer = model.design_pack(features, writer, binder_length=80)

The dict is the jfold backend's (`mosaic.models.mol1.featurize`): upstream's
`input_feature_dict` (as `mol pred` builds it: MSAs on, paired ones as unpaired; unbatched,
as numpy) on the keys jfold's Mol-1 reads, plus `atom37_idx` and `atom_pad_mask`.
Templates are taken as Mol-1's template JSON (`templatesPath` to a `.json` carrying each
template's mmCIF); search hits (`.hhr`, `.a3m`) would need the PDB's mmCIFs and release
dates, which this does not set up.

Featurization reads the CCD data of Mol-1's release (`common/`), downloaded into mosaic's
cache on first use (~0.7 GB)."""

import io
import shutil
import tarfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import gemmi
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from jaxtyping import PRNGKeyArray
from jfold.featurize.views.protenix import JFOLD_KEYS
from mol.data.core import ccd
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.data.template.template_utils import TemplateHitFeaturizer

from mosaic.cache import cache_dir
from mosaic.losses.atom37 import ATOM37_INDEX
from mosaic.models.mol1 import Mol1Features, feature_pack
from mosaic.models.upstream import global_rngs

RELEASE_URL = "https://mol-1-release.oss-cn-shanghai.aliyuncs.com"


def common_dir() -> Path:
    """The CCD data of Mol-1's release (`common/`: `components.cif` and its RDKit
    molecules), under mosaic's cache; downloaded on first use."""
    root = cache_dir("mol1")
    common = root / "common"
    if not (common / "components.cif").exists():
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "common.tar.gz"
        print(f"Downloading Mol-1's CCD data to {archive}")
        with (
            urllib.request.urlopen(f"{RELEASE_URL}/common.tar.gz") as response,
            open(archive, "wb") as f,
        ):
            shutil.copyfileobj(response, f)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        archive.unlink()
    return common


@contextmanager
def ccd_data() -> Iterator[None]:
    """Upstream reads the CCD from `$MOL_ROOT_DIR/common`, resolved at import; point it
    at `common_dir` instead."""
    common = common_dir()
    with (
        mock.patch.object(ccd, "COMPONENTS_FILE", str(common / "components.cif")),
        mock.patch.object(ccd, "RKDIT_MOL_PKL", common / "components.cif.rdkit_mol.pkl"),
    ):
        yield


def template_featurizer() -> TemplateHitFeaturizer:
    """The `TemplateHitFeaturizer` `InferenceDataset` builds (a template JSON reads only
    its centring setting)."""
    return TemplateHitFeaturizer(
        mmcif_dir=None,
        max_hits=4,
        max_template_date="2021-09-30",
        _shuffle_top_k_prefiltered=None,
        _max_template_candidates_num=20,
        fetch_remote=False,
    )


class AtomArrayWriter:
    """Writes coordinates `[A, 3]` (padding atoms after the real ones) as a
    `gemmi.Structure`, onto upstream's `AtomArray`."""

    def __init__(self, atoms: AtomArray):
        self.atoms = atoms

    def __call__(self, coords) -> gemmi.Structure:
        atoms = self.atoms.copy()
        atoms.coord = np.asarray(coords, np.float32).reshape(-1, 3)[: len(atoms)]
        cif = pdbx.CIFFile()
        pdbx.set_structure(cif, atoms)
        buffer = io.StringIO()
        cif.write(buffer)
        st = gemmi.make_structure_from_block(gemmi.cif.read_string(buffer.getvalue())[0])
        st.setup_entities()
        return st


def features(input: dict, *, key: PRNGKeyArray) -> tuple[Mol1Features, AtomArrayWriter]:
    """`input` (one entry of Mol-1's input JSON: `{"name": ..., "sequences": [...]}`)
    featurized as `mol pred` featurizes it (`InferenceDataset.process_one`: MSAs on, paired
    ones as unpaired, RNA MSAs where given), as the jfold backend's pack (see the module
    docstring), and a writer of every entity. `key` seeds numpy's global RNG, from which
    upstream draws the reference conformers' poses."""
    templates = any(
        "templatesPath" in entity.get("proteinChain", {}) for entity in input["sequences"]
    )
    dataset = InferenceDataset.__new__(InferenceDataset)  # skip the config-driven setup
    dataset.use_msa = True
    dataset.msa_pair_as_unpair = True
    dataset.use_rna_msa = True
    dataset.use_template = templates
    dataset.online_template_featurizer = template_featurizer() if templates else None
    with ccd_data(), global_rngs(key):
        data, atom_array, _ = dataset.process_one(input)
    upstream = {k: np.asarray(v) for k, v in data["input_feature_dict"].items()}
    names = np.asarray(atom_array.atom_name).astype(str)
    atom37_idx = np.array([ATOM37_INDEX.get(n, -1) for n in names], np.int32)
    is_protein = upstream["is_protein"].astype(bool)
    pack = feature_pack(
        {k: upstream[k] for k in JFOLD_KEYS}
        | {
            "atom37_idx": np.where(is_protein, atom37_idx, -1),
            "atom_pad_mask": np.ones(len(names), bool),
        }
    )
    return pack, AtomArrayWriter(atom_array)
