
from __future__ import annotations

import dataclasses
import tempfile
from copy import deepcopy

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from jpromera.config import DIFFUSION, NTOKS
from jpromera.diffusion import AtomDiffusion
from jpromera.feats import Feats
from jpromera.features import build_msas, copy_coords, featurize
from jpromera.model import InitialEmbedding, JPromera
from jpromera.model import JPromera as _JPromera
from jpromera.model import TrunkState as PromeraTrunkState
from jpromera.serialize import load_model as load_promera_model
from jpromera.trunk import TrunkOutput

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput
from mosaic.sampling import Denoise, Sampler, schedules
from mosaic.sampling.core import mean_centering, rigid_align_stop_gradient
from mosaic.sampling.schedules import Schedule
from mosaic.structure_prediction import (
    Atom37,
    Confidence,
    DiffusionStructurePredictionModel,
    Distogram,
    PolymerType,
    TargetChain,
    TokenInfo,
    TrunkState,
)


def _denoise(structure_module: AtomDiffusion, conditioning: PyTree, x, sigma):
    return structure_module.preconditioned_network_forward(
        x[None], sigma, conditioning
    )["denoised_atom_coords"][0]


def promera_denoiser(
    model: JPromera, feats: Feats, out
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk output (`JPromera.fold`), hoisting the step-invariant cache.

    Returns the denoiser and the atom mask.
    """
    cache = model.structure_module.score_model.precompute_conditioning(
        s_trunk=out.s,
        z_trunk=out.z,
        relative_position_encoding=out.relpos,
        feats=feats,
    )
    conditioning = {
        "s_inputs": out.s_inputs,
        "s_trunk": out.s,
        "z_trunk": out.z,
        "relative_position_encoding": out.relpos,
        "feats": feats,
        "multiplicity": 1,
        "cache": cache,
    }
    denoise = eqx.Partial(_denoise, model.structure_module, conditioning)
    return denoise, feats.atom_pad_mask[0].astype(jnp.float32)


def promera_sampler(
    num_steps: int,
    diffusion_cfg: dict = DIFFUSION,
    *,
    schedule: schedules.Schedule | None = None,
) -> Sampler:
    """The `Sampler` reproducing `JPromera.sample`.

    `schedule` defaults to the model's own `Karras`."""
    cfg = diffusion_cfg
    if schedule is None:
        schedule = schedules.Karras(rho=cfg["rho"])
    sigmas = schedule(
        np.linspace(0.0, 1.0, num_steps + 1),
        sigma_data=cfg["sigma_data"],
        s_max=cfg["sigma_max"],
        s_min=cfg["sigma_min"],
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=cfg["gamma_0"],
        gamma_min=cfg["gamma_min"],
        noise_scale=cfg["noise_scale"],
        step_scale=cfg["step_scale"],
        center=mean_centering,
        augment_after_noise=True,
        align=rigid_align_stop_gradient,
    )

_AA0 = 2
_UNK = 22
DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
_BB_SLOTS = jnp.array([0, 1, 2, 4])


def _pad_pssm(pssm: Float[Array, "Nb 20"]) -> Float[Array, "Nb NTOKS"]:
    return jnp.pad(pssm, ((0, 0), (_AA0, NTOKS - _AA0 - 20)))


def set_binder_sequence(pssm: Float[Array, "Nb 20"], feats: Feats) -> Feats:
    binder_len = pssm.shape[0]
    soft = _pad_pssm(pssm)

    restype = jax.nn.one_hot(feats.restype, NTOKS).at[0, :binder_len].set(soft)
    msa = jax.nn.one_hot(feats.msa, NTOKS).at[0, 0, :binder_len].set(soft)
    profile = feats.profile.at[0, :binder_len].set(soft)

    return eqx.tree_at(
        lambda f: (f.restype, f.msa, f.profile), feats, (restype, msa, profile)
    )


def apply_binder_sequence(
    pssm, features: "MosaicFeatures", *, mask_fraction=0.0, key=None
):
    pack = features.pack
    assert pack is not None, (
        "apply_binder_sequence needs binder-design features built by "
        "binder_features(); got target-only features (pack is None)."
    )
    feats = set_binder_sequence(pssm, features.feats)
    argmax = jnp.argmax(jax.lax.stop_gradient(pssm), axis=-1)

    m = _binder_mask(mask_fraction, jax.random.fold_in(key, 1), argmax.shape[0])
    feats = _mask_binder_tokens(feats, m)
    argmax = jnp.where(m, UNK_ROW, argmax)

    packed = pack_from_argmax(argmax, pack.target, pack.tables, pack.m_max)
    atom37_idx = packed.pop("atom37_idx")
    t2r = packed.pop("token_to_rep_atom")

    def setter(f):
        return (
            f.ref_pos,
            f.ref_element,
            f.ref_charge,
            f.ref_hydrogens,
            f.ref_atom_name_chars,
            f.ref_space_uid,
            f.atom_to_token,
            f.atom_pad_mask,
            f.atom_is_std,
            f.token_to_rep_atom,
        )

    vals = (
        packed["ref_pos"][None],
        packed["ref_element"][None],
        packed["ref_charge"][None],
        packed["ref_hydrogens"][None],
        packed["ref_atom_name_chars"][None],
        packed["ref_space_uid"][None],
        packed["atom_to_token"][None],
        packed["atom_pad_mask"][None],
        packed["atom_is_std"][None],
        t2r[None],
    )
    feats = eqx.tree_at(setter, feats, vals)
    return feats, atom37_idx


def _binder_mask(fraction: float, key, n_binder: int) -> Array:
    return jax.random.uniform(key, (n_binder,)) < fraction


def _mask_binder_tokens(feats: Feats, m: Array) -> Feats:
    nb = m.shape[0]
    unk = jax.nn.one_hot(_UNK, NTOKS, dtype=feats.restype.dtype)
    mm = m[:, None]
    restype = feats.restype.at[0, :nb].set(jnp.where(mm, unk, feats.restype[0, :nb]))
    msa = feats.msa.at[0, 0, :nb].set(jnp.where(mm, unk, feats.msa[0, 0, :nb]))
    profile = feats.profile.at[0, :nb].set(jnp.where(mm, unk, feats.profile[0, :nb]))
    return eqx.tree_at(
        lambda f: (f.restype, f.msa, f.profile), feats, (restype, msa, profile)
    )


def atom37_index(feats: Feats) -> np.ndarray:
    chars = np.asarray(feats.ref_atom_name_chars[0])
    pad = np.asarray(feats.atom_pad_mask[0]) > 0
    idx = np.full(chars.shape[0], -1, dtype=np.int32)
    for a in range(chars.shape[0]):
        if not pad[a]:
            continue
        name = "".join(chr(int(c) + 32) for c in chars[a] if c != 0)
        idx[a] = ATOM37_INDEX.get(name, -1)
    return idx


class PackCtx(eqx.Module):
    target: object
    tables: object
    binder_len: int = eqx.field(static=True)
    m_max: int = eqx.field(static=True)


class MosaicFeatures(eqx.Module):
    feats: Feats
    atom37_idx: Array
    pack: PackCtx | None = None

    @staticmethod
    def of(feats: Feats, *, binder_len: int | None = None) -> "MosaicFeatures":
        atom37_idx = jnp.asarray(atom37_index(feats))
        pack = None
        if binder_len is not None:
            target = extract_target_block(feats, binder_len, np.asarray(atom37_idx))
            pack = PackCtx(
                target=target,
                tables=tables_to_jnp(),
                binder_len=binder_len,
                m_max=m_max_for(binder_len, int(target.n_atoms)),
            )
        return MosaicFeatures(feats=feats, atom37_idx=atom37_idx, pack=pack)


TINYPROT_RES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_LETTERS = "ARNDCQEGHILKMFPSTWYV"
NUM_AA = 20
UNK_RES = "UNK"
UNK_ROW = NUM_AA
NUM_ROWS = NUM_AA + 1
ALL_RES = TINYPROT_RES + [UNK_RES]
MAX_ATOMS = 14


def _name_chars(name: str) -> list[int]:
    return [ord(c) - 32 for c in name] + [0] * (4 - len(name))


class AATables:
    __slots__ = (
        "ref_pos",
        "ref_element",
        "ref_charge",
        "ref_hydrogens",
        "ref_atom_name_chars",
        "atom_is_std",
        "n_atoms",
        "rep_slot",
        "atom37_slot",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


def build_aa_tables() -> AATables:
    from tinyprot.ccd import _get_attrs, _get_conformer

    from mosaic.losses.atom37 import ATOM37_INDEX

    ref_pos = np.zeros((NUM_ROWS, MAX_ATOMS, 3), dtype=np.float32)
    ref_element = np.zeros((NUM_ROWS, MAX_ATOMS), dtype=np.int64)
    ref_charge = np.zeros((NUM_ROWS, MAX_ATOMS), dtype=np.int64)
    ref_hydrogens = np.zeros((NUM_ROWS, MAX_ATOMS), dtype=np.int64)
    name_chars = np.zeros((NUM_ROWS, MAX_ATOMS, 4), dtype=np.int64)
    atom_is_std = np.zeros((NUM_ROWS, MAX_ATOMS), dtype=bool)
    n_atoms = np.zeros((NUM_ROWS,), dtype=np.int64)
    rep_slot = np.zeros((NUM_ROWS,), dtype=np.int64)
    atom37_slot = np.full((NUM_ROWS, MAX_ATOMS), -1, dtype=np.int64)

    for r, res in enumerate(ALL_RES):
        names = _get_attrs(res, lambda at: at.GetProp("name"), leaving=False)
        elems = _get_attrs(res, lambda at: at.GetAtomicNum(), leaving=False)
        charges = _get_attrs(res, lambda at: at.GetFormalCharge(), leaving=False)
        hyd = _get_attrs(
            res, lambda at: at.GetTotalNumHs(includeNeighbors=True), leaving=False
        )
        pos = _get_conformer(res, leaving=False, jitter=0.0, randrot=False)

        n = len(names)
        assert n <= MAX_ATOMS, f"{res} has {n} > {MAX_ATOMS} atoms"
        n_atoms[r] = n
        ref_pos[r, :n] = pos
        ref_element[r, :n] = elems
        ref_charge[r, :n] = charges
        ref_hydrogens[r, :n] = hyd
        atom_is_std[r, :n] = True
        for j, nm in enumerate(names):
            name_chars[r, j] = _name_chars(nm)
            atom37_slot[r, j] = ATOM37_INDEX.get(nm, -1)
        rep_slot[r] = names.index("CB") if "CB" in names else names.index("CA")

    return AATables(
        ref_pos=ref_pos,
        ref_element=ref_element,
        ref_charge=ref_charge,
        ref_hydrogens=ref_hydrogens,
        ref_atom_name_chars=name_chars,
        atom_is_std=atom_is_std,
        n_atoms=n_atoms,
        rep_slot=rep_slot,
        atom37_slot=atom37_slot,
    )


def round32(n: int) -> int:
    return ((int(n) + 31) // 32) * 32


def m_max_for(binder_len: int, target_n_atoms: int) -> int:
    return round32(binder_len * MAX_ATOMS + target_n_atoms)


def _exclusive_cumsum(x):
    return jnp.cumsum(x) - x


class TargetBlock(eqx.Module):
    ref_pos: jax.Array
    ref_element: jax.Array
    ref_charge: jax.Array
    ref_hydrogens: jax.Array
    ref_atom_name_chars: jax.Array
    atom_is_std: jax.Array
    a2t_local: jax.Array
    atom37: jax.Array
    n_atoms_per_token: jax.Array
    rep_local: jax.Array
    n_atoms: int = eqx.field(static=True)
    n_tokens: int = eqx.field(static=True)


def extract_target_block(feats, binder_len: int, atom37_idx) -> TargetBlock:
    a2t = np.asarray(feats.atom_to_token[0])
    pad = np.asarray(feats.atom_pad_mask[0]) > 0
    t2r = np.asarray(feats.token_to_rep_atom[0])
    atom37_idx = np.asarray(atom37_idx)

    n_real = int(pad.sum())
    N = int(feats.restype.shape[1])
    is_target_atom = (a2t >= binder_len) & (np.arange(len(a2t)) < n_real)
    sel = np.where(is_target_atom)[0]
    if sel.size == 0:
        s0 = s1 = n_real
    else:
        s0 = int(sel.min())
        s1 = int(sel.max()) + 1
        assert np.array_equal(sel, np.arange(s0, s1)), "target atoms not contiguous"
        uid = np.asarray(feats.ref_space_uid[0])
        assert np.array_equal(uid[s0:s1], a2t[s0:s1]), (
            "packed path needs ref_space_uid == atom_to_token (standard polymer target)"
        )

    n_per = np.bincount(a2t[:n_real], minlength=N)
    off = np.concatenate([[0], np.cumsum(n_per)[:-1]])
    tgt_tokens = np.arange(binder_len, N)
    n_atoms_per_token = n_per[tgt_tokens]
    rep_local = t2r[tgt_tokens] - off[tgt_tokens]

    f = feats
    return TargetBlock(
        ref_pos=jnp.asarray(np.asarray(f.ref_pos[0])[s0:s1]),
        ref_element=jnp.asarray(np.asarray(f.ref_element[0])[s0:s1]),
        ref_charge=jnp.asarray(np.asarray(f.ref_charge[0])[s0:s1]),
        ref_hydrogens=jnp.asarray(np.asarray(f.ref_hydrogens[0])[s0:s1]),
        ref_atom_name_chars=jnp.asarray(np.asarray(f.ref_atom_name_chars[0])[s0:s1]),
        atom_is_std=jnp.asarray(np.asarray(f.atom_is_std[0])[s0:s1]),
        a2t_local=jnp.asarray((a2t[s0:s1] - binder_len).astype(np.int32)),
        atom37=jnp.asarray(atom37_idx[s0:s1].astype(np.int32)),
        n_atoms_per_token=jnp.asarray(n_atoms_per_token.astype(np.int32)),
        rep_local=jnp.asarray(rep_local.astype(np.int32)),
        n_atoms=int(s1 - s0),
        n_tokens=int(N - binder_len),
    )


class JnpAATables(eqx.Module):
    ref_pos: jax.Array
    ref_element: jax.Array
    ref_charge: jax.Array
    ref_hydrogens: jax.Array
    ref_atom_name_chars: jax.Array
    atom_is_std: jax.Array
    n_atoms: jax.Array
    rep_slot: jax.Array
    atom37_slot: jax.Array


def tables_to_jnp(tables: AATables | None = None) -> JnpAATables:
    t = tables if tables is not None else build_aa_tables()
    return JnpAATables(
        ref_pos=jnp.asarray(t.ref_pos),
        ref_element=jnp.asarray(t.ref_element.astype(np.int32)),
        ref_charge=jnp.asarray(t.ref_charge.astype(np.int32)),
        ref_hydrogens=jnp.asarray(t.ref_hydrogens.astype(np.int32)),
        ref_atom_name_chars=jnp.asarray(t.ref_atom_name_chars.astype(np.int32)),
        atom_is_std=jnp.asarray(t.atom_is_std),
        n_atoms=jnp.asarray(t.n_atoms.astype(np.int32)),
        rep_slot=jnp.asarray(t.rep_slot.astype(np.int32)),
        atom37_slot=jnp.asarray(t.atom37_slot.astype(np.int32)),
    )


def pack_from_argmax(
    argmax_idx, target: TargetBlock, tables: JnpAATables, m_max: int
) -> dict:
    Nb = argmax_idx.shape[0]
    J = MAX_ATOMS
    Ta = target.n_atoms

    nb_atoms = tables.n_atoms[argmax_idx]
    n_t = jnp.concatenate([nb_atoms, target.n_atoms_per_token])
    off = _exclusive_cumsum(n_t)
    off_target = jnp.sum(nb_atoms)

    r_idx = jnp.repeat(jnp.arange(Nb), J)
    j_idx = jnp.tile(jnp.arange(J), Nb)
    valid = j_idx < nb_atoms[r_idx]
    binder_dest = jnp.where(valid, off[r_idx] + j_idx, m_max)

    target_dest = off_target + jnp.arange(Ta)

    def scatter(binder_flat, target_vals, fill, dtype):
        shape = (m_max + 1,) + binder_flat.shape[1:]
        out = jnp.full(shape, fill, dtype=dtype)
        out = out.at[binder_dest].set(binder_flat.astype(dtype))
        out = out.at[target_dest].set(target_vals.astype(dtype))
        return out[:m_max]

    def gather_binder(table_field):
        g = table_field[argmax_idx]
        return g.reshape(Nb * J, *table_field.shape[2:])

    ref_pos = scatter(gather_binder(tables.ref_pos), target.ref_pos, 0.0, jnp.float32)
    ref_element = scatter(
        gather_binder(tables.ref_element), target.ref_element, 0, jnp.int32
    )
    ref_charge = scatter(
        gather_binder(tables.ref_charge), target.ref_charge, 0, jnp.int32
    )
    ref_hydrogens = scatter(
        gather_binder(tables.ref_hydrogens), target.ref_hydrogens, 0, jnp.int32
    )
    ref_name_chars = scatter(
        gather_binder(tables.ref_atom_name_chars),
        target.ref_atom_name_chars,
        0,
        jnp.int32,
    )
    atom_is_std = scatter(
        gather_binder(tables.atom_is_std), target.atom_is_std, False, jnp.bool_
    )
    atom37_b = gather_binder(tables.atom37_slot)
    atom37_idx = scatter(atom37_b, target.atom37, -1, jnp.int32)

    binder_tok = r_idx
    target_tok = Nb + target.a2t_local
    atom_to_token = scatter(binder_tok, target_tok, 0, jnp.int32)

    ref_space_uid = atom_to_token

    n_real = off_target + Ta
    atom_pad_mask = (jnp.arange(m_max) < n_real).astype(jnp.int32)

    rep_slot_all = jnp.concatenate([tables.rep_slot[argmax_idx], target.rep_local])
    token_to_rep_atom = off + rep_slot_all

    return dict(
        ref_pos=ref_pos,
        ref_element=ref_element,
        ref_charge=ref_charge,
        ref_hydrogens=ref_hydrogens,
        ref_atom_name_chars=ref_name_chars,
        atom_is_std=atom_is_std,
        atom_to_token=atom_to_token,
        ref_space_uid=ref_space_uid,
        atom_pad_mask=atom_pad_mask,
        token_to_rep_atom=token_to_rep_atom,
        atom37_idx=atom37_idx,
    )

# `model_output` and the loss builders.
_DESIGN_RECYCLING_STEPS = 3
# `predict`: upstream's `inference/cofolding.yaml`.
_UPSTREAM_SAMPLING_STEPS = 200
_INFERENCE_RECYCLING_STEPS = 4

_POLY = {
    PolymerType.PROTEIN: "protein",
    PolymerType.RNA: "rna",
    PolymerType.DNA: "dna",
}


class _StructureWriter:
    def __init__(self, struct, schema):
        self.struct = struct
        self.schema = schema

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        struct = self.struct
        if binder_pssm is not None:
            schema = deepcopy(self.schema)
            binder_id = next(iter(schema))
            binder_sequence = "".join(
                TOKENS[index] for index in np.asarray(binder_pssm).argmax(-1)
            )
            schema[binder_id]["sequence"] = binder_sequence
            _features, struct = featurize(schema, build_msa=False)
        s = copy_coords(struct, np.asarray(coords))
        with tempfile.NamedTemporaryFile(suffix=".cif") as f:
            s.to_mmcif(f.name, metadata=False)
            return gemmi.read_structure(f.name)


def _schema(chains: list[TargetChain]) -> dict:
    return {
        cid: {"type": _POLY[c.polymer_type], "sequence": c.sequence}
        for cid, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    }


def load_pretrained():
    from huggingface_hub import hf_hub_download

    cache = cache_dir() / "promera"
    stem = hf_hub_download(
        repo_id="escalante-bio/jpromera",
        filename="promera_2606.eqx",
        cache_dir=cache,
    )[: -len(".eqx")]
    hf_hub_download(
        repo_id="escalante-bio/jpromera",
        filename="promera_2606.skeleton.pkl",
        cache_dir=cache,
    )
    return load_promera_model(stem)


class JPromeraModel(DiffusionStructurePredictionModel[MosaicFeatures, InitialEmbedding]):
    """ProMERA. `recycling_steps` counts trunk passes, as upstream's does."""

    model: _JPromera
    subsample: int = 1024
    # Fraction of binder tokens masked to UNK each step (design losses only).
    mask_fraction: float | Array = 0.0

    design_recycling_steps: int = eqx.field(static=True, default=_DESIGN_RECYCLING_STEPS)
    inference_recycling_steps: int = eqx.field(static=True, default=_INFERENCE_RECYCLING_STEPS)
    backprop_all_passes: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        model: _JPromera | None = None,
        subsample: int = 1024,
        mask_fraction=0.0,
        *,
        design_recycling_steps: int = _DESIGN_RECYCLING_STEPS,
        inference_recycling_steps: int = _INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
    ):
        self.model = model if model is not None else load_pretrained()
        self.subsample = subsample
        self.mask_fraction = mask_fraction
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    def _features(self, chains: list[TargetChain], *, binder_len=None):
        from tinyprot.structure import Structure

        schema = _schema(chains)
        struct0 = Structure.from_schema(schema)
        want = {
            cid: struct0.chains[cid]
            for cid, c in zip(struct0.chains, chains)
            if c.use_msa
        }
        if want:
            build_msas(want)
        feats, struct = featurize(schema, build_msa=False)
        mf = MosaicFeatures.of(feats, binder_len=binder_len)
        return mf, _StructureWriter(struct, schema)

    def target_only_features(self, chains: list[TargetChain]):
        return self._features(list(chains))

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        binder = TargetChain(sequence="X" * binder_length, use_msa=False)
        return self._features([binder] + list(chains), binder_len=binder_length)

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return promera_sampler(num_steps or _UPSTREAM_SAMPLING_STEPS, DIFFUSION, schedule=schedule)

    # -- the structure-model contract -----------------------------------------

    def set_sequence(
        self, features: MosaicFeatures, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ) -> MosaicFeatures:
        """Set the binder's sequence channels and repack its atoms from
        `argmax(sequence)`, masking `mask_fraction` of it to UNK."""
        feats, atom37_idx = apply_binder_sequence(
            sequence, features, mask_fraction=self.mask_fraction, key=key
        )
        return dataclasses.replace(features, feats=feats, atom37_idx=atom37_idx)

    def embed(self, features: MosaicFeatures, *, key: PRNGKeyArray) -> InitialEmbedding:
        del key
        return self.model.embed_inputs(features.feats)

    def init_state(self, features: MosaicFeatures, inputs: InitialEmbedding, *, key: PRNGKeyArray) -> TrunkState:
        del features, key
        state = self.model.init_state(inputs)
        return TrunkState(pair=state.z[0], single=state.s[0])

    def step(self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        """One pass on a random subset of `subsample` MSA rows."""
        out = self.model.step(
            features.feats, inputs, _promera_state(state), key=key, subsample=self.subsample
        )
        return TrunkState(pair=out.z[0], single=out.s[0])

    def _trunk_output(self, inputs: InitialEmbedding, state: TrunkState) -> TrunkOutput:
        z = state.pair[None]
        return TrunkOutput(
            s=state.single[None], z=z, s_inputs=inputs.s_inputs, s_init=inputs.s_init,
            z_init=inputs.z_init, relpos=inputs.relpos, pdistogram=self.model.distogram_module(z),
        )

    def structure_denoiser(
        self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState
    ) -> tuple[Denoise, Float[Array, " A"]]:
        return promera_denoiser(self.model, features.feats, self._trunk_output(inputs, state))

    def distogram(self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState) -> Distogram:
        del features, inputs
        return Distogram(
            logits=self.model.distogram_module(state.pair[None])[0],
            bins=jnp.asarray(DISTOGRAM_BINS),
        )

    def confidence(
        self, features: MosaicFeatures, inputs: InitialEmbedding, state: TrunkState, coords: Float[Array, "A 3"]
    ) -> Confidence:
        out = self.model.sm_confidence_module(
            features.feats, self._trunk_output(inputs, state), coords[None], multiplicity=1
        )
        return Confidence(plddt=out.plddt[0], pae_logits=out.pae_logits[0], pae_bins=jnp.asarray(PAE_BINS))

    def tokens(self, features: MosaicFeatures) -> TokenInfo:
        feats = features.feats
        restype = feats.restype if feats.restype.ndim == 3 else jax.nn.one_hot(feats.restype, NTOKS)
        return TokenInfo(
            sequence=restype[0][:, _AA0 : _AA0 + 20],
            asym_id=feats.asym_id[0],
            residue_index=feats.residue_index[0],
            mask=feats.token_pad_mask[0].astype(bool),
        )

    def atom37(self, features: MosaicFeatures, coords: Float[Array, "A 3"]) -> Atom37:
        feats = features.feats
        atom37_coords, atom37_mask = scatter_atom37(
            coords, feats.atom_to_token[0], features.atom37_idx, feats.restype.shape[1]
        )
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(self, features, writer, output: StructureModelOutput, sequence) -> gemmi.Structure:
        del features
        return writer(output.structure_coordinates, binder_pssm=sequence)


def _promera_state(state: TrunkState) -> PromeraTrunkState:
    return PromeraTrunkState(s=state.single[None], z=state.pair[None])
