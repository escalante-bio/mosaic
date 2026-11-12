"""Featurize Mol-1 inputs with molence, a fork of upstream Mol-1 that needs torch.

`features` takes one entry of Mol-1's input JSON as a dict, so it supports everything
`mol pred` does, including nucleic acids, ligands, MSAs and template JSON. It returns the
same features as `Mol1.target_only_features`, with a structure writer.

For design, put the binder first as poly-Trp, which has the most atoms per residue, and
convert the features with `Mol1.design_pack`:

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

Templates must be given as Mol-1's template JSON, with `templatesPath` pointing to a
`.json` file that contains each template's mmCIF. Template search hits are not supported.

Featurization needs Mol-1's CCD data, about 0.7 GB, which is downloaded on first use."""

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
    """Return the directory of Mol-1's CCD data, downloading it on first use."""
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
    """Point upstream's CCD paths at `common_dir`.

    Upstream otherwise reads the CCD from `$MOL_ROOT_DIR/common`, resolved at import."""
    common = common_dir()
    with (
        mock.patch.object(ccd, "COMPONENTS_FILE", str(common / "components.cif")),
        mock.patch.object(ccd, "RKDIT_MOL_PKL", common / "components.cif.rdkit_mol.pkl"),
    ):
        yield


def template_featurizer() -> TemplateHitFeaturizer:
    """Build the same `TemplateHitFeaturizer` that `InferenceDataset` builds."""
    return TemplateHitFeaturizer(
        mmcif_dir=None,
        max_hits=4,
        max_template_date="2021-09-30",
        _shuffle_top_k_prefiltered=None,
        _max_template_candidates_num=20,
        fetch_remote=False,
    )


class AtomArrayWriter:
    """Convert coordinates of shape `[A, 3]` to a `gemmi.Structure` using upstream's atoms.

    Padding atoms must come after the real ones and are dropped."""

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
    """Featurize one entry of Mol-1's input JSON as `mol pred` does.

    Return the features and a structure writer. Paired MSAs are used as unpaired MSAs.
    `key` seeds numpy's global RNG, which upstream uses to randomly place the reference
    conformers."""
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
