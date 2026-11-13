"""Featurize Boltz-2 inputs with the upstream `boltz` package, which needs torch.

`features` takes boltz's input YAML as a dict, so it supports everything boltz does,
including nucleic acids, ligands, modifications, constraints, templates and paired MSAs.
It returns the same features as `mosaic.models.boltz2.Boltz2.target_only_features`, with a
writer that uses boltz's mmCIF writer.

For design, put the binder first, since `Boltz2.set_sequence` overwrites the first
`PSSM.shape[0]` tokens. For example, to design a binder to a protein with a bound ligand:

    import jax

    from mosaic.models.boltz2 import Boltz2
    from mosaic.models.upstream import boltz2 as upstream

    model = Boltz2()
    features, writer = upstream.features(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "X" * 80, "msa": "empty"}},
                {"protein": {"id": "B", "sequence": target_sequence}},
                {"ligand": {"id": "C", "ccd": "SAM"}},
            ],
        },
        key=jax.random.key(0),
    )
    loss = model.build_loss(loss=..., features=features)
    prediction = model.predict(PSSM=pssm, features=features, writer=writer, key=key)
"""

import dataclasses
import tarfile
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import boltz.main as bm
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml
from boltz.data import const
from boltz.data.types import Coords, Interface, StructureV2
from boltz.data.write.mmcif import to_mmcif
from jaxtyping import PRNGKeyArray

from mosaic.cache import resolve_cache
from mosaic.models.upstream import global_rngs

COLABFOLD_SERVER = "https://api.colabfold.com"


class StructureWriter:
    """Convert coordinates of shape `[A, 3]` to a `gemmi.Structure` with boltz's writer.

    Padding atoms must come after the real ones and are dropped."""

    def __init__(self, structure: StructureV2):
        self.structure = structure.remove_invalid_chains()

    def __call__(self, coords) -> gemmi.Structure:
        atoms = self.structure.atoms.copy()
        atoms["coords"] = np.asarray(coords).reshape(-1, 3)[: len(atoms)]
        atoms["is_present"] = True
        residues = self.structure.residues.copy()
        residues["is_present"] = True
        structure = dataclasses.replace(
            self.structure,
            atoms=atoms,
            residues=residues,
            interfaces=np.array([], dtype=Interface),
            coords=np.array([(x,) for x in atoms["coords"]], dtype=Coords),
        )
        block = gemmi.cif.read_string(to_mmcif(structure, boltz2=True))[0]
        return gemmi.make_structure_from_block(block)


def mol_dir() -> Path:
    """Return Boltz-2's CCD conformer library, downloading it on first use."""
    cache = resolve_cache(None, "boltz", create=True)
    mols = cache / "mols"
    if not mols.exists():
        archive = cache / "mols.tar"
        if not archive.exists():
            print(f"Downloading Boltz-2's CCD library to {archive}")
            urllib.request.urlretrieve(bm.MOL_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(cache, filter="data")
    return mols


def features(input: dict, *, key: PRNGKeyArray):
    """Featurize `input`, boltz's input YAML as a dict, and return features and a writer.

    Relative paths in `input` are resolved from the working directory. `key` seeds the
    RNGs that boltz uses to randomly place the reference conformers."""
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        path = directory / "input.yaml"
        path.write_text(yaml.safe_dump(input))
        mols = mol_dir()
        bm.process_inputs(
            data=bm.check_inputs(path),
            out_dir=directory,
            ccd_path=mols.parent / "ccd.pkl",
            mol_dir=mols,
            use_msa_server=True,
            msa_server_url=COLABFOLD_SERVER,
            msa_pairing_strategy="greedy",
            boltz2=True,
        )
        processed = directory / "processed"
        data = bm.Boltz2InferenceDataModule(
            manifest=bm.Manifest.load(processed / "manifest.json"),
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=mols,
            num_workers=0,
            constraints_dir=processed / "constraints",
            template_dir=processed / "templates",
            extra_mols_dir=processed / "mols",
            override_method=None,
        )
        with global_rngs(key, seed_torch=True):
            batch = next(iter(data.predict_dataloader()))
        record = batch["record"][0]
        structure = StructureV2.load(processed / "structures" / f"{record.id}.npz")

    arrays = {k: v[0].numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    arrays["msa"] = np.eye(const.num_tokens, dtype=np.float32)[arrays["msa"]]
    if input.get("templates"):
        assert arrays["template_mask"].sum() > 0, "boltz read no template"
    return jax.tree.map(jnp.asarray, arrays), StructureWriter(structure)
