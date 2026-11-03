
import copy
import dataclasses

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import protenix.backend as protenix_backend
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from protenix.data.constants import GAP, PRO_STD_RESIDUES, RES_ATOMS_DICT
from protenix.data.template import ChainInput, featurize
from protenix.protenij import DiffusionModel, InitialEmbedding, TrunkEmbedding
from protenix.protenij import Protenix as Protenij

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
    TrunkState,
)


def _denoise(diffusion_module: DiffusionModel, conditioning: dict, x, sigma):
    return diffusion_module(x_noisy=x[None], t_hat_noise_level=sigma[None], **conditioning)[0]


def protenix_denoiser(
    model: Protenij, features: PyTree, initial_embedding, trunk_state
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (`Protenix.embed` and `Protenix.step`).

    Returns the denoiser and the atom mask.
    """
    conditioning = {
        "input_feature_dict": features,
        "s_inputs": initial_embedding.s_inputs,
        "s_trunk": trunk_state.s,
        "z_trunk": trunk_state.z,
    }
    denoise = eqx.Partial(_denoise, model.diffusion_module, conditioning)
    num_atoms = features["atom_to_token_idx"].shape[-1]
    # Design packs pad the atom axis after refreshing the binder (`atom_pad_mask`).
    return denoise, jnp.asarray(features.get("atom_pad_mask", jnp.ones(num_atoms)), jnp.float32)


def protenix_sampler(
    model: Protenij, num_steps: int, *, schedule: schedules.Schedule | None = None
) -> Sampler:
    """The `Sampler` of upstream Protenix's `sample_diffusion`, with the
    schedule and churn parameters read off the model.

    Upstream re-centres and randomly rotates and translates every step; the JAX
    port only re-centres. `schedule` defaults to the model's own `Karras`.
    """
    scheduler = model.inference_noise_scheduler
    if schedule is None:
        schedule = schedules.Karras(rho=scheduler.rho)
    sigmas = schedules.zero_last(
        schedule(
            np.linspace(0.0, 1.0, num_steps + 1),
            sigma_data=scheduler.sigma_data,
            s_max=scheduler.s_max,
            s_min=scheduler.s_min,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=model.gamma0,
        gamma_min=model.gamma_min,
        noise_scale=model.noise_scale_lambda,
        step_scale=model.step_scale_eta,
        center=masked_centering,
        augment=True,
        align=no_alignment,
    )


def biotite_atom_to_gemmi_atom(atom):
    ga = gemmi.Atom()
    ga.pos = gemmi.Position(*atom.coord)
    ga.element = gemmi.Element(atom.element)
    ga.name = atom.atom_name
    return ga


def new_gemmi_residue(atom):
    r = gemmi.Residue()
    r.name = atom.res_name
    r.seqid = gemmi.SeqId(atom.res_id, " ")
    r.entity_type = gemmi.EntityType.Polymer
    return r


def biotite_array_to_gemmi_struct(atom_array, pred_coord=None, per_atom_plddt=None):
    if pred_coord is not None:
        atom_array = copy.deepcopy(atom_array)
        atom_array.coord = pred_coord
    structure = gemmi.Structure()
    model = gemmi.Model("0")
    chains = {}
    for atom_idx, atom in enumerate(atom_array):
        chain = chains.setdefault(atom.chain_id, {})
        residue = chain.setdefault(int(atom.res_id), new_gemmi_residue(atom))
        gemmi_atom = biotite_atom_to_gemmi_atom(atom)
        if per_atom_plddt is not None:
            gemmi_atom.b_iso = per_atom_plddt[atom_idx]
        residue.add_atom(gemmi_atom)
    for k in chains:
        chain = gemmi.Chain(k)
        chain.append_residues(list(chains[k].values()))
        model.add_chain(chain)
    structure.add_model(model)
    return structure


def _build_boltz_to_protenix_matrix():
    T = np.zeros((len(TOKENS), 32))
    for i, tok in enumerate(TOKENS):
        protenix_idx = PRO_STD_RESIDUES[
            gemmi.expand_one_letter(tok, gemmi.ResidueKind.AA)
        ]
        T[i, protenix_idx] = 1
    return T

BOLTZ_TO_PROTENIX = _build_boltz_to_protenix_matrix()


def _build_protenix_atom37_table() -> np.ndarray:
    """`tokatom_to_atom37[protenix_idx, tokatom_idx]` → atom37 slot, -1 if
    the atom isn't in atom37 (e.g. OXT) or the residue isn't a standard AA.
    """
    n_protenix = 32
    max_tokatom = 1 + max(
        max(atoms.values()) for atoms in RES_ATOMS_DICT.values() if atoms
    )
    tokatom_to_atom37 = -np.ones((n_protenix, max_tokatom), dtype=np.int32)
    for three_letter, protenix_idx in PRO_STD_RESIDUES.items():
        for atom_name, tokatom_idx in RES_ATOMS_DICT.get(three_letter, {}).items():
            if atom_name in ATOM37_INDEX:
                tokatom_to_atom37[protenix_idx, tokatom_idx] = ATOM37_INDEX[atom_name]
    return tokatom_to_atom37


_TOKATOM_TO_ATOM37 = _build_protenix_atom37_table()


def _set_binder_identity(new_sequence: Float[Array, "N 20"], features: PyTree):
    """Set the binder's (first `N` tokens) `restype`, `profile` and MSA query row
    to `new_sequence`; atom features are left as featurized.

    Protenix takes `profile` per chain, as the mean over that chain's own MSA
    rows: the rows with a residue in its columns (unpaired MSAs are merged
    row-aligned, gap-padded). Swapping the query row is then a rank-one update
    by 1 / that depth, which keeps the rest of a binder MSA's statistics; a
    binder without an MSA has depth 1, so its profile becomes the sequence.
    Exact unless the featurizer cropped the binder's MSA rows below the depth
    its profile was taken over (more than 16384 rows)."""
    binder_len = new_sequence.shape[0]
    protenix_sequence = new_sequence @ BOLTZ_TO_PROTENIX
    msa = features["msa"]
    depth = (msa[:, :binder_len] != GAP["-"]).any(-1).sum()
    old_query = features["restype"][:binder_len]  # one-hot of msa[0] at the binder positions
    return features | {
        "restype": features["restype"].at[:binder_len].set(protenix_sequence),
        "profile": features["profile"].at[:binder_len].add((protenix_sequence - old_query) / depth),
        "msa": msa.at[0, :binder_len].set(protenix_sequence.argmax(-1).astype(msa.dtype)),
    }


# ---------------------------------------------------------------------------
# Binder atom refresh
# ---------------------------------------------------------------------------
# `binder_features` featurizes a poly-Trp binder (Trp is the largest residue), so
# the binder's atom block fits any sequence without a recompile. Protenix's
# atom-level features are those of the featurized residues, so
# `refresh_binder_atoms` rebuilds every atom-axis feature (and the token
# features that index atoms) of the binder from `argmax(pssm)`: binder atoms
# are tight-packed in native order, the target atoms follow, and the freed tail
# is padding (`atom_pad_mask`, which Protenij masks throughout). The real-atom
# prefix then equals Protenix's own featurization of the designed sequence, up
# to the random per-residue rigid augmentation of `ref_pos`.

MAX_ATOMS_PER_RES = 15  # C-terminal Trp: 14 heavy atoms + OXT
# Read from per-residue templates.
_TEMPLATE_ATOM_FIELDS = (
    "ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
    "atom_to_tokatom_idx", "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
    "distogram_rep_atom_mask",
)
# Constant over a residue: copied from the placeholder residue at the same position.
_RESIDUE_ATOM_FIELDS = (
    "ref_space_uid", "is_protein", "is_ligand", "is_dna", "is_rna", "mol_id",
    "entity_mol_id", "modified_res_mask",
)
# Rebuilt from the packing (`bond_mask` is all-zero for protein-only complexes).
_DERIVED_ATOM_FIELDS = ("atom_to_token_idx", "mol_atom_index", "bond_mask", "atom_pad_mask")


class ProtenixAtomTemplates(eqx.Module):
    """Protenix atom features of each residue type (mosaic `TOKENS` order) in two
    contexts: 0 = internal (or N-terminal), 1 = C-terminal (carries OXT).
    Atom slots are in native order, zero past `n_atoms`; `ref_pos` is the
    reference conformer centred over the residue (pre-augmentation)."""

    atoms: dict  # name -> [2, 20, MAX_ATOMS_PER_RES, ...]
    n_atoms: Int[Array, "2 20"]
    frame: Int[Array, "2 20 3"]  # local N, CA, C (`frame_atom_index`)
    rep: Int[Array, "2 20"]  # local representative atom (`atom_rep_atom_idx`: CB, Gly CA)


def build_atom_templates(features: dict, ref_pos, tokens) -> ProtenixAtomTemplates:
    """Read templates off a featurization: `tokens[context, aa]` is the token
    holding residue `TOKENS[aa]` in that context; `ref_pos` is the unaugmented
    conformer (`atom_array.ref_pos`)."""
    a2t = np.asarray(features["atom_to_token_idx"])
    atoms = {
        name: np.zeros((2, 20, MAX_ATOMS_PER_RES, *np.shape(features[name])[1:]), np.asarray(features[name]).dtype)
        for name in _TEMPLATE_ATOM_FIELDS
    }
    n_atoms, frame, rep = np.zeros((2, 20), np.int32), np.zeros((2, 20, 3), np.int32), np.zeros((2, 20), np.int32)
    for c, i in np.ndindex(2, 20):
        token = tokens[c][i]
        idx = np.flatnonzero(a2t == token)
        assert len(idx) <= MAX_ATOMS_PER_RES and np.all(np.diff(idx) == 1)
        assert features["has_frame"][token] == 1
        for name in _TEMPLATE_ATOM_FIELDS:
            atoms[name][c, i, : len(idx)] = features[name][idx]
        atoms["ref_pos"][c, i, : len(idx)] = ref_pos[idx] - ref_pos[idx].mean(0)
        n_atoms[c, i] = len(idx)
        frame[c, i] = features["frame_atom_index"][token] - idx[0]
        rep[c, i] = features["atom_rep_atom_idx"][token] - idx[0]
    return ProtenixAtomTemplates(
        atoms={k: jnp.asarray(v) for k, v in atoms.items()},
        n_atoms=jnp.asarray(n_atoms),
        frame=jnp.asarray(frame),
        rep=jnp.asarray(rep),
    )


class ProtenixDesignFeatures(eqx.Module):
    """A design pack from `Protenix.binder_features`: placeholder features plus
    what `set_binder_sequence` needs to rebuild the binder's atoms for any
    sequence of `binder_length`."""

    features: dict
    atom_templates: ProtenixAtomTemplates
    binder_length: int = eqx.field(static=True)
    binder_atom_alloc: int = eqx.field(static=True)


def binder_design_features(
    features: dict, binder_length: int, templates: ProtenixAtomTemplates
) -> ProtenixDesignFeatures:
    """Wrap a featurization whose first chain is a `binder_length` placeholder
    binder with enough atoms for any sequence (poly-Trp)."""
    from protenix.atom_padding import ATOM_FEATURE_AXES

    a2t = np.asarray(features["atom_to_token_idx"])
    asym = np.asarray(features["asym_id"])
    alloc = int((a2t < binder_length).sum())
    assert (asym == asym[0]).sum() == binder_length, "the binder must be the first chain"
    assert alloc >= (MAX_ATOMS_PER_RES - 1) * binder_length + 1, "placeholder binder too small (use poly-Trp)"
    assert not np.any(features["bond_mask"]), "bond_mask must be empty (protein-only complex)"
    handled = {*_TEMPLATE_ATOM_FIELDS, *_RESIDUE_ATOM_FIELDS, *_DERIVED_ATOM_FIELDS}
    assert not (unhandled := (ATOM_FEATURE_AXES.keys() & features.keys()) - handled), unhandled
    features = features | {"atom_pad_mask": np.ones(len(a2t), dtype=bool)}
    return ProtenixDesignFeatures(
        features=jax.tree.map(jnp.asarray, features),
        atom_templates=templates,
        binder_length=binder_length,
        binder_atom_alloc=alloc,
    )


def refresh_binder_atoms(pack: ProtenixDesignFeatures, pssm: Float[Array, "L 20"], key: jax.Array) -> dict:
    """Rebuild the binder's atom-axis features (and `frame_atom_index`,
    `atom_rep_atom_idx`) for `argmax(pssm)`.

    `ref_pos` gets the featurizer's augmentation: per residue, a random
    translation in [-1, 1]^3 and then a uniform random rotation (`key`)."""
    features, templates = pack.features, pack.atom_templates
    L, alloc = pack.binder_length, pack.binder_atom_alloc
    n_atoms = features["atom_to_token_idx"].shape[0]
    M = MAX_ATOMS_PER_RES
    ids = pssm.argmax(-1)
    context = (jnp.arange(L) == L - 1).astype(jnp.int32)

    def gather(table):  # [2, 20, ...] -> [L, ...]
        return table[context, ids]

    k = gather(templates.n_atoms)
    start = jnp.cumsum(k) - k
    shift = k.sum() - alloc  # target atoms move left by the binder's unused slots
    slot = jnp.arange(M)
    dest = jnp.where(slot < k[:, None], start[:, None] + slot, n_atoms).reshape(-1)  # n_atoms: dropped
    target = jnp.arange(alloc, n_atoms) + shift

    def pack(name, binder):
        old = features[name]
        new = jnp.zeros_like(old).at[dest].set(binder.reshape(L * M, *old.shape[1:]).astype(old.dtype), mode="drop")
        return new.at[target].set(old[alloc:])

    kt, kr = jax.random.split(key)
    t = jax.random.uniform(kt, (L, 1, 3), minval=-1.0, maxval=1.0)
    ref_pos = jnp.einsum("lac,ldc->lad", gather(templates.atoms["ref_pos"]) + t, jax.vmap(random_rotation)(jax.random.split(kr, L)))
    first = jnp.searchsorted(features["atom_to_token_idx"][:alloc], jnp.arange(L))  # placeholder residue starts
    binder = {name: gather(table) for name, table in templates.atoms.items()} | {
        name: jnp.broadcast_to(features[name][first][:, None], (L, M)) for name in _RESIDUE_ATOM_FIELDS if name in features
    } | {
        "ref_pos": ref_pos,
        "atom_to_token_idx": jnp.broadcast_to(jnp.arange(L)[:, None], (L, M)),
        "mol_atom_index": start[:, None] + slot,
    }
    frame, rep = features["frame_atom_index"], features["atom_rep_atom_idx"]
    return features | {name: pack(name, values) for name, values in binder.items()} | {
        "atom_pad_mask": jnp.arange(n_atoms) < n_atoms + shift,
        "frame_atom_index": jnp.concatenate([start[:, None] + gather(templates.frame), frame[L:] + shift]).astype(frame.dtype),
        "atom_rep_atom_idx": jnp.concatenate([start + gather(templates.rep), rep[L:] + shift]).astype(rep.dtype),
    }


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree | ProtenixDesignFeatures,
    key: jax.Array,
):
    """Set the binder (the first `N` tokens) to `new_sequence`.

    `restype`, `profile` and the MSA query row are set as Protenix would
    featurize the sequence (`_set_binder_identity`). A `ProtenixDesignFeatures`
    pack (from `binder_features`) also gets the binder's atoms and frames
    rebuilt from `argmax(PSSM)` (`key` drives the `ref_pos` augmentation). A
    plain feature dict (e.g. `target_only_features`) keeps the atoms of the
    residues it was featurized with."""
    if not isinstance(features, ProtenixDesignFeatures):
        return _set_binder_identity(new_sequence, features)
    assert new_sequence.shape == (features.binder_length, 20), new_sequence.shape
    # A separate stream for the `ref_pos` augmentation, leaving `key` to the fold.
    feat = refresh_binder_atoms(features, new_sequence, jax.random.fold_in(key, 0x70A7))
    return _set_binder_identity(new_sequence, feat)


PROTENIX_DISTOGRAM_BINS = np.linspace(start=2.3125, stop=21.6875, num=64)

_MSA_AUTH_FIX_INSTALLED = False


def install_protenix_msa_auth_fix():
    global _MSA_AUTH_FIX_INSTALLED
    if _MSA_AUTH_FIX_INSTALLED:
        return
    try:
        import protenix.web_service.colab_request_utils as _cru
        _cru.HTTPBasicAuth = lambda *a, **kw: None
    except Exception:
        return
    _MSA_AUTH_FIX_INSTALLED = True

install_protenix_msa_auth_fix()

def atom_templates():
    """Per-residue atom templates read off one featurization: residues 1-20 of
    `G + TOKENS + G` (internal) and the last residue of each `G + aa` (C-terminal)."""
    sequences = ["G" + TOKENS + "G", *("G" + aa for aa in TOKENS)]
    features, atom_array, _ = featurize([ChainInput(s, compute_msa=False) for s in sequences])
    tokens = [range(1, 21), [len(sequences[0]) + 2 * i + 1 for i in range(20)]]
    return build_atom_templates(features, atom_array.ref_pos, tokens)


def _chain_input(chain: TargetChain | ChainInput) -> ChainInput:
    """Protenix chains pass through (e.g. with `precomputed_msa_dir`)."""
    if isinstance(chain, ChainInput):
        return chain
    assert chain.polymer_type == PolymerType.PROTEIN, (
        "Protenix interface only supports Protein chains. Manually build features for more complex targets. "
    )
    return ChainInput(sequence=chain.sequence, compute_msa=chain.use_msa, template=chain.template_chain)


def load_model(name="protenix_mini_default_v0.5.0"):
    protenix_backend._CACHE_DIR = str(cache_dir() / "protenix")
    return protenix_backend.load_model(name)


class Protenix(DiffusionStructurePredictionModel[dict | ProtenixDesignFeatures, InitialEmbedding]):
    """`recycling_steps` counts trunk passes (upstream's `N_cycle`)."""

    protenix: eqx.Module
    # Upstream's per-model inference settings (`configs_model_type.py`).
    upstream_sampling_steps: int
    inference_recycling_steps: int
    # Initial trunk state; zeros if None.
    initial_recycling_state: TrunkEmbedding | None = None
    design_recycling_steps: int = eqx.field(static=True, default=1)
    backprop_all_passes: bool = eqx.field(static=True, default=False)

    def target_only_features(self, chains: list[TargetChain | ChainInput]):
        features_dict, atom_array, _ = featurize([_chain_input(c) for c in chains])
        return features_dict, atom_array

    def binder_features(self, binder_length, chains: list[TargetChain | ChainInput]):
        """A poly-Trp binder followed by `chains`, as a `ProtenixDesignFeatures`
        pack: a PSSM passed with it rebuilds the binder's atoms (Protenix's own
        featurization of `argmax(PSSM)`, padded to the poly-Trp atom budget).
        The writer is the placeholder's; `predict` writes the atom37 view instead."""
        binder = TargetChain(sequence="W" * binder_length, use_msa=False)
        features, atom_array = self.target_only_features([binder, *chains])
        return binder_design_features(features, binder_length, atom_templates()), atom_array

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return protenix_sampler(
            self.protenix, num_steps or self.upstream_sampling_steps, schedule=schedule
        )

    # -- the structure-model contract -----------------------------------------

    def set_sequence(self, features, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray):
        """Set the binder's sequence channels; a design pack also rebuilds the
        binder's atoms from `argmax(sequence)`."""
        refreshed = set_binder_sequence(sequence, features, key)
        if isinstance(features, ProtenixDesignFeatures):
            return dataclasses.replace(features, features=refreshed)
        return refreshed

    def embed(self, features, *, key: PRNGKeyArray) -> InitialEmbedding:
        del key
        return self.protenix.embed_inputs(input_feature_dict=_feat(features))

    def init_state(self, features, inputs: InitialEmbedding, *, key: PRNGKeyArray) -> TrunkState:
        """Zeros, or `initial_recycling_state` when set."""
        del features, key
        state = self.initial_recycling_state or self.protenix.init_state(initial_embedding=inputs)
        return TrunkState(pair=state.z, single=state.s)

    def step(self, features, inputs: InitialEmbedding, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        out = self.protenix.step(
            initial_embedding=inputs,
            input_feature_dict=_feat(features),
            state=_protenij_state(state),
            key=key,
        )
        return TrunkState(pair=out.z, single=out.s)

    def structure_denoiser(
        self, features, inputs: InitialEmbedding, state: TrunkState
    ) -> tuple[Denoise, Float[Array, " A"]]:
        return protenix_denoiser(self.protenix, _feat(features), inputs, _protenij_state(state))

    def distogram(self, features, inputs: InitialEmbedding, state: TrunkState) -> Distogram:
        del features, inputs
        return Distogram(
            logits=self.protenix.distogram_head(state.pair),
            bins=jnp.array(PROTENIX_DISTOGRAM_BINS),
        )

    def confidence(
        self, features, inputs: InitialEmbedding, state: TrunkState, coords: Float[Array, "A 3"]
    ) -> Confidence:
        feats = _feat(features)
        out = self.protenix.confidence_metrics(
            initial_embedding=inputs,
            trunk_embedding=_protenij_state(state),
            input_feature_dict=feats,
            coordinates=coords[None],
            key=jax.random.key(0),  # inference: no dropout
        )
        # pLDDT normalized to [0, 1]
        plddt = (
            jax.nn.softmax(out.plddt_logits[0][feats["atom_rep_atom_idx"]])
            * jnp.linspace(0, 1, 50)[None, :]
        ).sum(-1)
        return Confidence(plddt=plddt, pae_logits=out.pae_logits[0], pae_bins=jnp.array(PAE_BINS))

    def tokens(self, features) -> TokenInfo:
        feats = _feat(features)
        n = feats["restype"].shape[0]
        return TokenInfo(
            sequence=feats["restype"] @ BOLTZ_TO_PROTENIX.T,
            asym_id=feats["asym_id"],
            residue_index=feats["residue_index"],
            mask=feats.get("token_pad_mask", jnp.ones(n, bool)),
        )

    def atom37(self, features, coords: Float[Array, "A 3"]) -> Atom37:
        feats = _feat(features)
        a2t = feats["atom_to_token_idx"]
        atom37_idx = jnp.asarray(_TOKATOM_TO_ATOM37)[feats["restype"].argmax(-1)[a2t], feats["atom_to_tokatom_idx"]]
        atom37_idx = jnp.where(feats["ref_mask"] > 0.5, atom37_idx, jnp.int32(-1))
        atom37_coords, atom37_mask = scatter_atom37(coords, a2t, atom37_idx, feats["restype"].shape[0])
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(self, features, writer, output: StructureModelOutput, sequence) -> gemmi.Structure:
        """A design pack is written from the atom37 view; otherwise with `writer`."""
        del sequence
        if isinstance(features, ProtenixDesignFeatures):
            return output.to_structure()
        return biotite_array_to_gemmi_struct(writer, np.array(output.structure_coordinates))


def _feat(features) -> dict:
    return features.features if isinstance(features, ProtenixDesignFeatures) else features


def _protenij_state(state: TrunkState) -> TrunkEmbedding:
    return TrunkEmbedding(s=state.single, z=state.pair)


def _protenix(name: str, *, mini: bool) -> Protenix:
    return Protenix(
        load_model(name=name),
        upstream_sampling_steps=5 if mini else 200,
        inference_recycling_steps=4 if mini else 10,
    )


def ProtenixMini():
    return _protenix("protenix_mini_default_v0.5.0", mini=True)


def ProtenixTiny():
    return _protenix("protenix_tiny_default_v0.5.0", mini=True)


def ProtenixBase():
    return _protenix("protenix_base_default_v1.0.0", mini=False)


def Protenix2025():
    return _protenix("protenix_base_20250630_v1.0.0", mini=False)


def ProtenixV2():
    return _protenix("protenix-v2", mini=False)
