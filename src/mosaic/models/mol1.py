"""Mol-1 (via the ``jol1`` JAX port) behind mosaic's structure-prediction interface.

`recycling_steps` counts trunk passes (upstream's ``N_cycle``). Every pass uses the
first `Mol1.msa_rows` MSA rows; upstream instead draws a random subset per pass, which
makes design gradients noisy. `model_output` and the losses default to a deterministic
ODE sampler; `predict` uses upstream's.

A `binder_features` pack repacks the binder's atoms from ``argmax(PSSM)`` every step
(see `mosaic.losses.mol1`); a `target_only_features` pack is used as featurized.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import jol1
import numpy as np
from jaxtyping import Array, Float

from mosaic.cache import cache_dir
from mosaic.losses.mol1 import (
    Mol1FeaturePack,
    MultiSampleMol1Loss,
    ResidueAtomTemplates,
    binder_atom_budget,
    build_pack,
    mol1_forward_from_trunk,
    prepare_features,
    residue_atom_templates,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.sampling import Sampler
from mosaic.sampling.mol1 import mol1_sampler
from mosaic.structure_prediction import (
    DiffusionStructurePredictionModel,
    PolymerType,
    StructurePrediction,
    TargetChain,
)

_RELEASE_URL = "https://mol-1-release.oss-cn-shanghai.aliyuncs.com"
_MSA_SERVER = "https://api.colabfold.com"
_DESIGN_SAMPLING_STEPS = 20
_MSA_ROWS = 1024
# `mol pred` defaults.
_INFERENCE_SAMPLING_STEPS = 200
_INFERENCE_RECYCLING_STEPS = 20


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(path)


def release_dir() -> Path:
    """The Mol-1 release files under mosaic's cache: the checkpoint and the reference
    data (``common/``: CCD components) that featurization reads. Downloaded on first use
    (~2.5 GB)."""
    root = cache_dir("mol1")
    checkpoint = root / "checkpoint" / "mol-1-base.pt"
    if not checkpoint.exists():
        _download(f"{_RELEASE_URL}/checkpoint/mol-1-base.pt", checkpoint)
    if not (root / "common" / "components.cif").exists():
        archive = root / "common.tar.gz"
        _download(f"{_RELEASE_URL}/common.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        archive.unlink()
    return root


def load_model(root: Path) -> jol1.Mol1:
    """The converted ``mol-1-base`` weights under `release_dir`, converting (and caching)
    them on first use."""
    converted = root / "jol1" / "mol1_base"
    if not converted.with_suffix(".eqx").exists():
        from jol1.convert import convert_checkpoint

        converted.parent.mkdir(parents=True, exist_ok=True)
        jol1.save_model(convert_checkpoint(root / "checkpoint" / "mol-1-base.pt"), converted)
    return jol1.load_model(converted)


def _fetch_msa(sequence: str) -> Path:
    """Unpaired ColabFold MSA (a3m) for ``sequence``, cached by sequence hash."""
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    msa_cache = cache_dir("msa")
    msa_cache.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    a3m = msa_cache / f"{seq_hash}.a3m"
    if not a3m.exists():
        result = run_mmseqs2(
            [sequence],
            prefix=str(msa_cache / seq_hash),
            use_env=True,
            use_filter=True,
            use_pairing=False,
            host_url=_MSA_SERVER,
        )
        a3m.write_text((result[0] if isinstance(result, tuple) else result)[0])
    return a3m


def _chain_entry(chain: TargetChain) -> dict:
    if chain.template_chain is not None:
        raise NotImplementedError("the Mol-1 wrapper does not support template chains yet")
    if chain.polymer_type == PolymerType.PROTEIN:
        entry = {"sequence": chain.sequence, "count": 1}
        if chain.use_msa:
            entry["unpairedMsaPath"] = str(_fetch_msa(chain.sequence))
        return {"proteinChain": entry}
    kind = {PolymerType.DNA: "dnaSequence", PolymerType.RNA: "rnaSequence"}[chain.polymer_type]
    return {kind: {"sequence": chain.sequence, "count": 1}}


def _to_gemmi(atom_array, coords: np.ndarray) -> gemmi.Structure:
    from biotite.structure.io import pdbx

    atom_array = atom_array.copy()
    atom_array.coord = coords
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atom_array)
    buffer = io.StringIO()
    cif.write(buffer)
    return gemmi.make_structure_from_block(gemmi.cif.read_string(buffer.getvalue())[0])


class Mol1(DiffusionStructurePredictionModel):
    """Mol-1 (``mol-1-base``). Construct with `Mol1Base()`."""

    mol1: jol1.Mol1
    templates: ResidueAtomTemplates
    data_dir: str = eqx.field(static=True)
    # Leading MSA rows (query first) every trunk pass uses; `None`: all.
    msa_rows: int | None = eqx.field(static=True)
    design_sampling_steps: int = eqx.field(static=True)
    inference_sampling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)

    def target_only_features(self, chains: list[TargetChain]):
        """Featurize the chains as given. Returns ``(pack, atom_array)``; the
        ``AtomArray`` is the writer for `predict`."""
        from jol1.reference import reference_feature_dict

        entry = {"name": "mosaic", "sequences": [_chain_entry(c) for c in chains]}
        feats, atom_array = reference_feature_dict(entry, use_msa=True, data_dir=self.data_dir)
        return build_pack(feats, atom_array), atom_array

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        """Design pack: a single-sequence binder (chain 0) plus the targets. The binder
        is allocated as all-Trp (the most atoms per residue) and repacked from
        ``argmax(PSSM)`` every step, so its placeholder sequence never reaches the
        model."""
        binder = TargetChain("W" * binder_length, use_msa=False)
        pack, atom_array = self.target_only_features([binder, *chains])
        binder_atoms = int(np.sum(pack.features.atoms.atom_to_token < binder_length))
        assert binder_atoms == binder_atom_budget(binder_length), binder_atoms
        pack = dataclasses.replace(pack, templates=self.templates, binder_length=binder_length)
        return pack, atom_array

    def design_sampler(self, num_steps: int | None = None) -> Sampler:
        """Upstream's sampler as a deterministic ODE, without random augmentation."""
        return dataclasses.replace(
            self.inference_sampler(num_steps or self.design_sampling_steps),
            gamma_0=0.0,
            step_scale=1.0,
            augment=False,
        )

    def inference_sampler(self, num_steps: int | None = None) -> Sampler:
        return mol1_sampler(self.mol1, num_steps or self.inference_sampling_steps)

    def build_loss(
        self,
        *,
        loss,
        features: Mol1FeaturePack,
        recycling_steps: int = 1,
        sampler: Sampler | None = None,
    ):
        return self.build_multisample_loss(
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            num_samples=1,
            sampler=sampler,
        )

    def build_multisample_loss(
        self,
        *,
        loss,
        features: Mol1FeaturePack,
        recycling_steps: int = 1,
        num_samples: int = 4,
        reduction=jnp.mean,
        sampler: Sampler | None = None,
    ):
        return MultiSampleMol1Loss(
            model=self.mol1,
            pack=features,
            loss=loss,
            sampler=sampler if sampler is not None else self.design_sampler(),
            recycling_steps=recycling_steps,
            msa_rows=self.msa_rows,
            num_samples=num_samples,
            reduction=reduction,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: Mol1FeaturePack,
        recycling_steps: int = 1,
        sampler: Sampler | None = None,
        key,
    ):
        pack = prepare_features(features, PSSM, key)
        inputs = self.mol1.embed(pack.features)
        state = self.mol1.trunk(
            pack.features, inputs, num_cycles=recycling_steps, msa_rows=self.msa_rows
        )
        return mol1_forward_from_trunk(
            self.mol1,
            pack,
            inputs,
            state,
            sampler=sampler if sampler is not None else self.design_sampler(),
            key=key,
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: Mol1FeaturePack,
        writer=None,
        recycling_steps: int | None = None,
        sampler: Sampler | None = None,
        key,
    ) -> StructurePrediction:
        """Fold with upstream's inference settings. A `binder_features` pack is written
        from the atom37 view (protein chains only), since the repacked binder no longer
        matches the featurized ``AtomArray``; a `target_only_features` pack needs its
        ``AtomArray`` as ``writer``."""
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=(
                recycling_steps if recycling_steps is not None else self.inference_recycling_steps
            ),
            sampler=sampler if sampler is not None else self.inference_sampler(),
            key=key,
        )
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
        if features.templates is not None:
            st = output.to_structure()
        else:
            st = _to_gemmi(writer, np.asarray(output.structure_coordinates[0]))
        return StructurePrediction(
            st=st, plddt=output.plddt, pae=output.pae, iptm=iptm, model_output=output
        )


def Mol1Base() -> Mol1:
    data_dir = release_dir()
    return Mol1(
        mol1=load_model(data_dir),
        templates=residue_atom_templates(data_dir),
        data_dir=str(data_dir),
        msa_rows=_MSA_ROWS,
        design_sampling_steps=_DESIGN_SAMPLING_STEPS,
        inference_sampling_steps=_INFERENCE_SAMPLING_STEPS,
        inference_recycling_steps=_INFERENCE_RECYCLING_STEPS,
    )
