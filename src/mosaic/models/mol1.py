"""Mol-1 (via the ``jol1`` JAX port) behind mosaic's structure-prediction interface.

`recycling_steps` counts trunk passes (upstream's ``N_cycle``). Every pass uses the
first `Mol1.msa_rows` MSA rows; upstream instead draws a random subset per pass, which
makes design gradients noisy.

A `binder_features` pack repacks the binder's atoms from ``argmax(PSSM)`` every step
(see `refresh_binder_atoms`); a `target_only_features` pack is used as featurized.
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
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jol1 import Features, InputEmbedding
from jol1.transformer import AtomFeatures

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput
from mosaic.sampling import Denoise, Sampler, schedules
from mosaic.sampling.core import masked_centering, no_alignment, random_rotation
from mosaic.sampling.schedules import Schedule
from mosaic.structure_prediction import (
    Atom37,
    Confidence,
    DiffusionStructurePredictionModel,
    Distogram,
    PolymerType,
    TargetChain,
    TokenInfo,
    Trunk,
    TrunkState,
)


def mol1_denoiser(
    model: jol1.Mol1, features: Features, inputs: InputEmbedding, state: jol1.TrunkState
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (`Mol1.embed`, `Mol1.trunk`); the step-invariant
    diffusion conditioning is computed here, once.

    Returns the denoiser and the atom mask.
    """
    context = model.diffusion_context(features, inputs, state)
    denoise = eqx.Partial(jol1.Mol1.denoise, model, context)
    return denoise, features.atoms.atom_mask.astype(jnp.float32)


def mol1_sampler(
    model: jol1.Mol1, num_steps: int, *, schedule: schedules.Schedule | None = None
) -> Sampler:
    """The `Sampler` of upstream Mol-1's `sample_diffusion` (AF3 Algorithm 18:
    re-centre, random rotation and translation, churn, Euler step; no alignment),
    with the schedule and churn parameters of the checkpoint's inference config.

    Upstream centres on the plain atom mean; it never pads atoms, so masked centring
    (which ignores the padding of a repacked binder) is the same. `schedule`
    defaults to the model's own `Karras`."""
    settings = model.sampler_settings
    if schedule is None:
        schedule = schedules.Karras(rho=settings.rho)
    sigmas = schedules.zero_last(
        schedule(
            np.linspace(0.0, 1.0, num_steps + 1),
            sigma_data=settings.sigma_data,
            s_max=settings.s_max,
            s_min=settings.s_min,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=settings.gamma0,
        gamma_min=settings.gamma_min,
        noise_scale=settings.noise_scale_lambda,
        step_scale=settings.step_scale_eta,
        center=masked_centering,
        augment=True,
        align=no_alignment,
    )

# Mol-1's residue alphabet (Protenix's) starts with the 20 amino acids in mosaic order.
NUM_RESTYPES = 32
MSA_GAP = 31  # "-"


def bin_centers(min_bin: float, max_bin: float, num_bins: int) -> np.ndarray:
    """Mol-1's `get_bin_centers`."""
    width = (max_bin - min_bin) / num_bins
    return (min_bin + width * (np.arange(num_bins) + 0.5)).astype(np.float32)


DISTOGRAM_BINS = bin_centers(2.3125, 21.6875, 64)
PLDDT_BINS = bin_centers(0.0, 1.0, 50)


MAX_ATOMS_PER_RESIDUE = 15  # Trp + OXT


def binder_atom_budget(binder_length: int) -> int:
    """Atoms of the all-Trp placeholder binder of `binder_features` (14 per residue,
    plus OXT on the C-terminal one): enough for any sequence."""
    return (MAX_ATOMS_PER_RESIDUE - 1) * binder_length + 1


class ResidueAtomTemplates(eqx.Module):
    """Per-residue (mosaic-20 order) reference atoms, mid-chain (variant 0) and
    C-terminal (variant 1, with OXT), as ``[2, 20, MAX_ATOMS_PER_RESIDUE, ...]`` tables.
    Real atoms fill the leading slots in Mol-1's order (N, CA, C, O, side chain[, OXT])
    and are marked by ``atoms.atom_mask``."""

    atoms: AtomFeatures
    atom37_idx: Int[Array, "2 20 A"]
    rep_offset: Int[Array, "2 20"]  # distogram representative atom (CB; CA for Gly)


class Mol1FeaturePack(eqx.Module):
    """Model inputs plus the per-atom / per-token indices `StructureModelOutput` needs."""

    features: Features
    atom37_idx: Int[Array, " N"]  # atom37 slot of each atom, -1 for none (non-protein)
    plddt_atom_idx: Int[Array, " T"]  # CA for protein tokens, else the representative atom
    # Set by `binder_features`: the binder (chain 0, allocated at `binder_atom_budget`)
    # is repacked from argmax(PSSM) every step (`refresh_binder_atoms`).
    templates: ResidueAtomTemplates | None = None
    binder_length: int = eqx.field(static=True, default=0)


def build_pack(feats: dict, atom_array) -> Mol1FeaturePack:
    """From the reference feature dict and ``AtomArray`` (`jol1.reference`)."""
    features = Features.from_reference_dict(feats)
    names = np.asarray(atom_array.atom_name).astype(str)
    is_protein = np.asarray(feats["is_protein"]).astype(bool)
    atom37_idx = np.array([ATOM37_INDEX.get(n, -1) for n in names], np.int32)
    ca = np.flatnonzero(is_protein & (names == "CA"))
    plddt_atom = np.asarray(features.tokens.rep_atom).copy()
    plddt_atom[np.asarray(feats["atom_to_token_idx"])[ca]] = ca
    return Mol1FeaturePack(
        features=features,
        atom37_idx=jnp.asarray(np.where(is_protein, atom37_idx, -1)),
        plddt_atom_idx=jnp.asarray(plddt_atom, jnp.int32),
    )


# ---------------------------------------------------------------------------
# Binder sequence
# ---------------------------------------------------------------------------


def set_binder_sequence(pssm: Float[Array, "L 20"], features: Features) -> Features:
    """Splice the soft sequence into ``restype``, ``profile`` and the MSA query row of
    the first ``L`` tokens.

    Mol-1 takes ``profile`` per chain, as the mean over that chain's own MSA rows: the
    rows with a residue in its columns. Swapping the query row is then a rank-one update
    by 1 / that depth, which keeps the rest of a binder MSA's statistics; a binder
    without an MSA (`binder_features`) has depth 1, so its profile becomes the sequence.
    Exact unless the featurizer cropped the binder's MSA rows."""
    L = pssm.shape[0]
    # Padding, not a matmul: GPU default (TF32) matmuls would round the PSSM.
    soft = jnp.pad(pssm, ((0, 0), (0, NUM_RESTYPES - 20)))
    tokens, msa = features.tokens, features.msa.msa
    depth = (msa[:, :L, MSA_GAP] < 0.5).any(-1).sum()
    old_query = tokens.restype[:L]  # one-hot of the MSA query row at the binder positions
    # Subtract first: exact (``soft``) for depth 1.
    profile = tokens.profile[:L] - old_query / depth + soft / depth
    return eqx.tree_at(
        lambda f: (f.tokens.restype, f.tokens.profile, f.msa.msa),
        features,
        (tokens.restype.at[:L].set(soft), tokens.profile.at[:L].set(profile), msa.at[0, :L].set(soft)),
    )


# ---------------------------------------------------------------------------
# In-loop binder repack
# ---------------------------------------------------------------------------
#
# Mol-1 embeds every atom's reference conformer (position, element, name, charge), so a
# binder featurized as one fixed sequence would carry that sequence's side chains
# whatever the PSSM says. As for ESMFold2 and Protenix, the binder is allocated at the
# per-residue maximum (`binder_atom_budget`), and every step the argmax residues' atoms
# are gathered from per-residue templates and packed tightly at the front of the atom
# axis, the target atoms shifted to follow and the freed tail masked as padding (jol1
# ignores padding atoms). Static shapes, no recompiles. The argmax has no gradient: the
# PSSM gradient flows through the sequence channels. Each binder residue's reference
# conformer gets a fresh random pose every step, as the featurizer's augmentation.


def residue_atom_templates(data_dir) -> ResidueAtomTemplates:
    """Read off one featurization: residues 1-20 of ``G + TOKENS + G`` (mid-chain) and
    the last residue of each ``G + aa`` (C-terminal). ``ref_pos`` is centred over the
    residue (`refresh_binder_atoms` applies the random pose)."""
    from jol1.reference import reference_feature_dict

    sequences = ["G" + TOKENS + "G", *("G" + aa for aa in TOKENS)]
    entry = {
        "name": "templates",
        "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in sequences],
    }
    feats, atom_array = reference_feature_dict(entry, use_msa=False, data_dir=data_dir)
    pack = build_pack(feats, atom_array)
    atoms, atom37_idx, rep_atom = jax.tree.map(
        np.asarray, (pack.features.atoms, pack.atom37_idx, pack.features.tokens.rep_atom)
    )
    restype = np.asarray(feats["restype"]).argmax(-1)

    def residue(token: int, aa: int, c_terminal: bool) -> ResidueAtomTemplates:
        idx = np.flatnonzero(atoms.atom_to_token == token)
        names = [str(n) for n in atom_array.atom_name[idx]]
        assert restype[token] == aa, (TOKENS[aa], restype[token])
        assert names[:4] == ["N", "CA", "C", "O"], names
        assert (names[-1] == "OXT") == c_terminal, names

        def pad(x):
            return np.pad(x[idx], [(0, MAX_ATOMS_PER_RESIDUE - len(idx))] + [(0, 0)] * (x.ndim - 1))

        centred = eqx.tree_at(lambda a: a.ref_pos, atoms, atoms.ref_pos - atoms.ref_pos[idx].mean(0))
        return ResidueAtomTemplates(
            atoms=jax.tree.map(pad, centred),
            atom37_idx=pad(atom37_idx),
            rep_offset=np.int32(rep_atom[token] - idx[0]),
        )

    mid_chain = [residue(1 + aa, aa, False) for aa in range(20)]
    c_terminal = [residue(len(sequences[0]) + 2 * aa + 1, aa, True) for aa in range(20)]
    return jax.tree.map(
        lambda *xs: jnp.asarray(np.stack(xs).reshape(2, 20, *np.shape(xs[0]))),
        *mid_chain,
        *c_terminal,
    )


def refresh_binder_atoms(
    pack: Mol1FeaturePack, pssm: Float[Array, "L 20"], key: jax.Array
) -> Mol1FeaturePack:
    """Repack the binder's atoms as ``argmax(pssm)`` (see the comment above): every
    atom-axis feature and atom index of the pack, rebuilt for the new layout.

    ``ref_pos`` gets the featurizer's augmentation: per residue, a random translation in
    [-1, 1]^3 and then a uniform random rotation (``key``)."""
    L, tmpl, atoms = pack.binder_length, pack.templates, pack.features.atoms
    assert pssm.shape == (L, 20), (pssm.shape, L)
    N = atoms.atom_to_token.shape[0]
    allocated = binder_atom_budget(L)
    ids = jnp.argmax(pssm, axis=-1)
    variant = (jnp.arange(L) == L - 1).astype(jnp.int32)  # C-terminal residue has OXT

    def gather(table):  # [2, 20, ...] -> [L, ...]
        return table[variant, ids]

    binder = jax.tree.map(gather, tmpl.atoms)  # [L, A, ...]
    k = binder.atom_mask.sum(-1)
    offset = jnp.cumsum(k) - k  # first atom of each binder residue
    shift = jnp.sum(k) - allocated  # target atoms move by this much
    slot = jnp.arange(MAX_ATOMS_PER_RESIDUE)
    dest = jnp.where(slot < k[:, None], offset[:, None] + slot, N).reshape(-1)  # N: dropped
    target_dest = jnp.arange(allocated, N) + shift

    def repack(binder, field, fill=0):
        """Binder atoms packed tightly, then the target atoms; the freed tail is ``fill``."""
        binder = binder.reshape(-1, *field.shape[1:]).astype(field.dtype)
        packed = jnp.full_like(field, fill).at[dest].set(binder, mode="drop")
        return packed.at[target_dest].set(field[allocated:])

    def token_atoms(binder, field):
        """Per-token atom indices: binder from the new offsets, target shifted."""
        return jnp.concatenate([binder, field[L:] + shift]).astype(field.dtype)

    kt, kr = jax.random.split(key)
    translation = jax.random.uniform(kt, (L, 1, 3), minval=-1.0, maxval=1.0)
    rotation = jax.vmap(random_rotation)(jax.random.split(kr, L))
    ref_pos = jnp.einsum("lac,ldc->lad", binder.ref_pos + translation, rotation)
    # Binder residues own tokens and reference-space uids 0..L-1. The zero-filled tail
    # of `atom_mask` marks the padding.
    token = jnp.broadcast_to(jnp.arange(L)[:, None], (L, MAX_ATOMS_PER_RESIDUE))
    binder = eqx.tree_at(
        lambda a: (a.ref_pos, a.ref_space_uid, a.atom_to_token), binder, (ref_pos, token, token)
    )
    features = eqx.tree_at(
        lambda f: (f.atoms, f.tokens.rep_atom),
        pack.features,
        (
            jax.tree.map(repack, binder, atoms),
            token_atoms(offset + gather(tmpl.rep_offset), pack.features.tokens.rep_atom),
        ),
    )
    return eqx.tree_at(
        lambda p: (p.features, p.atom37_idx, p.plddt_atom_idx),
        pack,
        (
            features,
            repack(gather(tmpl.atom37_idx), pack.atom37_idx, fill=-1),
            token_atoms(offset + 1, pack.plddt_atom_idx),  # CA
        ),
    )

_RELEASE_URL = "https://mol-1-release.oss-cn-shanghai.aliyuncs.com"
_MSA_SERVER = "https://api.colabfold.com"
_MSA_ROWS = 1024
# `mol pred` defaults.
_UPSTREAM_SAMPLING_STEPS = 200
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


class Mol1(DiffusionStructurePredictionModel[Mol1FeaturePack, jol1.InputEmbedding]):
    """Mol-1 (``mol-1-base``). Construct with `Mol1Base()`."""

    mol1: jol1.Mol1
    templates: ResidueAtomTemplates
    data_dir: str = eqx.field(static=True)
    # Leading MSA rows (query first) every trunk pass uses; `None`: all.
    msa_rows: int | None = eqx.field(static=True)
    upstream_sampling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    design_recycling_steps: int = eqx.field(static=True, default=1)
    backprop_all_passes: bool = eqx.field(static=True, default=False)

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

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        if num_steps is None:
            num_steps = self.upstream_sampling_steps
        return mol1_sampler(self.mol1, num_steps, schedule=schedule)

    # -- the structure-model contract -----------------------------------------

    def set_sequence(
        self, features: Mol1FeaturePack, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray
    ) -> Mol1FeaturePack:
        """Set the binder's sequence channels; a design pack also repacks the binder's
        atoms from ``argmax(sequence)``."""
        pack = features
        if pack.templates is not None:
            pack = refresh_binder_atoms(pack, sequence, key)
        return eqx.tree_at(lambda p: p.features, pack, set_binder_sequence(sequence, pack.features))

    def embed(self, features: Mol1FeaturePack, *, key: PRNGKeyArray) -> jol1.InputEmbedding:
        del key
        return self.mol1.embed(features.features)

    def init_state(
        self, features: Mol1FeaturePack, inputs: jol1.InputEmbedding, *, key: PRNGKeyArray
    ) -> TrunkState:
        del features, key
        return TrunkState(pair=jnp.zeros_like(inputs.z_init), single=jnp.zeros_like(inputs.s_init))

    def step(
        self,
        features: Mol1FeaturePack,
        inputs: jol1.InputEmbedding,
        state: TrunkState,
        *,
        key: PRNGKeyArray,
    ) -> TrunkState:
        """One pass over the first `msa_rows` MSA rows (no random subsampling)."""
        del key
        feats = features.features
        msa = jax.tree.map(lambda x: x[: self.msa_rows], feats.msa)
        feats = eqx.tree_at(lambda f: f.msa, feats, msa)
        rows = jnp.ones(msa.msa.shape[0], bool)
        out = self.mol1.cycle(feats, inputs, _jol1_state(state), rows)
        return TrunkState(pair=out.z, single=out.s)

    def structure_denoiser(self, trunk: Trunk) -> tuple[Denoise, Float[Array, " A"]]:
        return mol1_denoiser(
            self.mol1, trunk.features.features, trunk.inputs, _jol1_state(trunk.state)
        )

    def distogram(self, trunk: Trunk) -> Distogram:
        return Distogram(
            logits=self.mol1.distogram_logits(_jol1_state(trunk.state)),
            bins=jnp.array(DISTOGRAM_BINS),
        )

    def confidence(self, trunk: Trunk, coords: Float[Array, "A 3"]) -> Confidence:
        out = self.mol1.confidence(
            trunk.features.features, trunk.inputs, _jol1_state(trunk.state), coords
        )
        plddt_logits = out.plddt_logits[trunk.features.plddt_atom_idx]
        return Confidence(
            plddt=(jax.nn.softmax(plddt_logits) * PLDDT_BINS).sum(-1),
            pae_logits=out.pae_logits,
            pae_bins=jnp.array(PAE_BINS, jnp.float32),
        )

    def tokens(self, features: Mol1FeaturePack) -> TokenInfo:
        tokens = features.features.tokens
        return TokenInfo(
            sequence=tokens.restype[:, :20],
            asym_id=tokens.asym_id,
            residue_index=tokens.residue_index,
            mask=jnp.ones(tokens.asym_id.shape, bool),
        )

    def atom37(self, features: Mol1FeaturePack, coords: Float[Array, "A 3"]) -> Atom37:
        feats = features.features
        atom37_coords, atom37_mask = scatter_atom37(
            atom_coords=coords,
            atom_to_token=feats.atoms.atom_to_token,
            atom37_idx=features.atom37_idx,
            n_token=feats.num_tokens,
        )
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(
        self,
        features: Mol1FeaturePack,
        writer,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """A design pack is written from the atom37 view (protein only); otherwise
        from its ``AtomArray`` (``writer``)."""
        del sequence
        if features.templates is not None:
            return output.to_structure()
        return _to_gemmi(writer, np.asarray(output.structure_coordinates))


def _jol1_state(state: TrunkState) -> jol1.TrunkState:
    return jol1.TrunkState(s=state.single, z=state.pair)


def Mol1Base() -> Mol1:
    data_dir = release_dir()
    return Mol1(
        mol1=load_model(data_dir),
        templates=residue_atom_templates(data_dir),
        data_dir=str(data_dir),
        msa_rows=_MSA_ROWS,
        upstream_sampling_steps=_UPSTREAM_SAMPLING_STEPS,
        inference_recycling_steps=_INFERENCE_RECYCLING_STEPS,
    )
