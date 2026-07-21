"""Mosaic `StructurePredictionModel` wrapper for OpenDDE (the `jopendde` JAX
port of Aureka's OpenDDE — an AF3-style all-atom co-folding model).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import tempfile
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from jopendde.features import Features
from jopendde.model import OpenDDE as JaxOpenDDE

from mosaic.common import LinearCombination, LossTerm
from mosaic.cache import cache_dir
from mosaic.losses.opendde import (
    MAX_ATOMS_PER_RES,
    MAX_STRUCT_PER_RES,
    MultiSampleOpenDDELoss,
    OpenDDEDesignFeatures,
    OpenDDEAtomTemplates,
    build_opendde_atom_templates,
    set_binder_sequence,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)

_DEFAULT_NUM_CYCLES = 4
_DEFAULT_NUM_SAMPLING_STEPS = 20

# `opendde_v1` relative-position-encoding parameters.
_RELP_R_MAX = 32
_RELP_S_MAX = 2

_MSA_SERVER = "https://api.colabfold.com"


def _target_a3m(sequence: str) -> str:
    """Fetch (and cache) an unpaired a3m for `sequence` from the ColabFold MMseqs2
    server, returning the a3m path. Cached by sequence hash under
    `MOSAIC_CACHE_DIR/msa`. Only called for chains with `use_msa=True`."""
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    msa_cache = cache_dir() / "msa"
    msa_cache.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    a3m_path = msa_cache / f"{seq_hash}.a3m"
    if not a3m_path.exists():
        result = run_mmseqs2(
            [sequence],
            prefix=str(msa_cache / seq_hash),
            use_env=True,
            use_filter=True,
            use_pairing=False,
            host_url=_MSA_SERVER,
        )
        # run_mmseqs2 returns list[str] or a tuple of lists depending on version.
        a3m_strings = result[0] if isinstance(result, tuple) else result
        a3m_path.write_text(a3m_strings[0])
    return str(a3m_path)


def _set_asset_cache_dir(configs, cache_root: Path):
    root = cache_root / "opendde"
    configs.load_checkpoint_dir = str(root / "checkpoint")
    configs.data.ccd_components_file = str(root / "common" / "components.cif")
    configs.data.ccd_components_rdkit_mol_file = str(
        root / "common" / "components.cif.rdkit_mol.pkl"
    )


def _build_dense_atom_to_atom37() -> np.ndarray:
    """`[restype(32), tokatom_idx] -> atom37 slot`, -1 where invalid.

    `atom_to_tokatom_idx` indexes each atom into its residue's dense atom list
    (`ATOM14_PADDED` order); map each dense atom name to its atom37 slot. Only the
    20 protein rows are populated (UNK/nucleic/gap -> -1). OpenDDE's atom37
    ordering matches mosaic's, so the slots line up directly.
    """
    from opendde.data.constants import (
        ATOM14_PADDED,
        ATOM37_ORDER,
        PROTEIN_COMMON_ONE_TO_THREE,
        PROTEIN_TYPES_ONE_LETTER,
    )

    max_dense = max(len(v) for v in ATOM14_PADDED.values())
    table = -np.ones((32, max_dense), dtype=np.int32)
    for rt, letter in enumerate(PROTEIN_TYPES_ONE_LETTER):  # rt 0..19
        names = ATOM14_PADDED[PROTEIN_COMMON_ONE_TO_THREE[letter]]
        for j, name in enumerate(names):
            if name and name in ATOM37_ORDER:
                table[rt, j] = ATOM37_ORDER[name]
    return table


def _build_spec(chains: list[TargetChain], *, name: str = "design", seed: int = 0):
    """OpenDDE/AF3 input spec (chains in order; binder first for a design run)."""
    sequences = []
    for c in chains:
        if c.polymer_type != PolymerType.PROTEIN:
            raise NotImplementedError(
                f"OpenDDE wrapper supports protein chains only; got {c.polymer_type}"
            )
        chain = {"sequence": c.sequence, "count": 1}
        if c.use_msa:
            chain["unpairedMsaPath"] = _target_a3m(c.sequence)
        sequences.append({"proteinChain": chain})
    return [{"name": name, "modelSeeds": [seed], "sequences": sequences}]


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cached_featurize_config(cache_root: str):
    """Build the immutable base config for an architecture and cache root."""
    from opendde.config.inference import (
        build_inference_config,
        update_gpu_compatible_configs,
    )
    from opendde.utils.download import download_inference_cache

    cfg = build_inference_config(
        model_name="opendde_v1", fill_required_with_null=True
    )
    _set_asset_cache_dir(cfg, Path(cache_root))
    cfg.use_msa = cfg.use_template = cfg.use_rna_msa = False
    cfg.triangle_multiplicative = cfg.triangle_attention = "torch"
    cfg.enable_diffusion_shared_vars_cache = False
    cfg.enable_efficient_fusion = False
    cfg = update_gpu_compatible_configs(cfg)
    download_inference_cache(cfg)
    return cfg


def _featurize_configs():
    """Return an isolated `opendde_v1` featurization config."""
    return copy.deepcopy(_cached_featurize_config(str(cache_dir())))


def _relp(feat_dict: dict) -> dict:
    """Add the `opendde_v1` relative-position feature to a raw feature dict."""
    import types

    from opendde.model.opendde import RelativePositionEncoding

    ns = types.SimpleNamespace(r_max=_RELP_R_MAX, s_max=_RELP_S_MAX)
    return RelativePositionEncoding.generate_relp(ns, feat_dict, lazy=False)


def _target_template_features(
    chains: list[TargetChain], *, max_templates: int = 4
) -> dict[str, np.ndarray] | None:
    """Convert Mosaic's in-memory Gemmi chains to OpenDDE template features."""
    if not any(c.template_chain is not None for c in chains):
        return None

    from opendde.data.constants import ATOM14_PADDED, STD_RESIDUES_WITH_GAP
    from opendde.data.template.template_featurizer import Templates

    n_res = sum(len(c.sequence) for c in chains)
    gap = STD_RESIDUES_WITH_GAP["-"]
    # OpenDDE represents an empty template row with GAP and pads additional rows
    # with ALA. Padded identities still affect embeddings despite zero masks.
    aatype = np.zeros((max_templates, n_res), dtype=np.int32)
    aatype[0] = gap
    positions = np.zeros((max_templates, n_res, 24, 3), dtype=np.float32)
    mask = np.zeros((max_templates, n_res, 24), dtype=bool)

    offset = 0
    for chain in chains:
        template_chain = chain.template_chain
        if template_chain is not None:
            polymer = template_chain.get_polymer()
            template_sequence = gemmi.one_letter_code([res.name for res in polymer])
            if template_sequence != chain.sequence:
                raise ValueError(
                    "OpenDDE template sequence does not match its target sequence: "
                    f"{template_sequence!r} != {chain.sequence!r}"
                )
            for i, residue in enumerate(polymer):
                restype = STD_RESIDUES_WITH_GAP.get(
                    residue.name, STD_RESIDUES_WITH_GAP["UNK"]
                )
                aatype[0, offset + i] = restype
                atom_slots = {
                    name: j
                    for j, name in enumerate(ATOM14_PADDED.get(residue.name, ()))
                    if name
                }
                for atom in residue:
                    atom_name = atom.name.upper()
                    if atom_name == "SE" and residue.name == "MSE":
                        atom_name = "SD"
                    if atom_name in atom_slots:
                        j = atom_slots[atom_name]
                        positions[0, offset + i, j] = atom.pos.tolist()
                        mask[0, offset + i, j] = True
        offset += len(chain.sequence)

    return Templates(
        aatype=aatype, atom_positions=positions, atom_mask=mask
    ).as_opendde_dict()


def _featurize(chains: list[TargetChain]):
    """Featurize protein chains without loading model weights.

    Returns JOpenDDE features and the Biotite atom array for native output.
    """
    import torch

    from opendde.data.inference.infer_dataloader import get_inference_dataloader
    from opendde.model.opendde import update_input_feature_dict

    from jopendde.features import Features
    from jopendde.inference import _spec_tempfile, _to_numpy

    cfg = _featurize_configs()
    # Per-chain a3m presence (set in `_build_spec`) controls MSA depth; the
    # featurizer gate is on when any chain supplies one.
    cfg.use_msa = any(c.use_msa for c in chains)
    spec = _build_spec(chains)
    seed = int(spec[0]["modelSeeds"][0])
    with _spec_tempfile(spec) as json_path:
        cfg.input_json_path = json_path
        cfg.seeds = [seed]
        data, atom_array, err = next(iter(get_inference_dataloader(configs=cfg)))[0]
        if err:
            raise RuntimeError(f"OpenDDE featurization failed: {err}")
    raw = copy.deepcopy(data["input_feature_dict"])
    raw["inference_seed"] = torch.tensor(seed, dtype=torch.long)
    with torch.no_grad():
        raw = _relp(raw)
    raw = update_input_feature_dict(raw)
    raw_np = _to_numpy(raw)
    template_features = _target_template_features(chains)
    if template_features is not None:
        raw_np.update(template_features)
    return Features.from_dict(raw_np), atom_array


class OpenDDEModel(StructurePredictionModel):
    """OpenDDE wrapped behind mosaic's structure-prediction interface."""

    model: JaxOpenDDE
    dense_atom_to_atom37: Int[Array, "32 Adense"]
    # Architecture-level residue atom/layout templates used to refresh design
    # geometry. These are not target-chain structural templates.
    atom_templates: OpenDDEAtomTemplates
    # (min_bin, max_bin, no_bins) for the confidence heads, read from the
    # checkpoint config at construction (static -> baked into the loss).
    pae_bin_params: tuple = eqx.field(static=True, default=(0.0, 32.0, 64))
    plddt_bin_params: tuple = eqx.field(static=True, default=(0.0, 1.0, 50))
    default_sampling_steps: int = eqx.field(
        static=True, default=_DEFAULT_NUM_SAMPLING_STEPS
    )

    # ------------------------------------------------------------------ features
    def target_only_features(self, chains: list[TargetChain]):
        """Featurize real protein chains. Returns `(feat, atom_array)` where `feat`
        is the jopendde `Features` and `atom_array` is the biotite writer for
        native structure output. Model weights are not loaded by this method."""
        return _featurize(chains)

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        """Featurize a poly-Trp binder followed by target chains for design.

        Returns an `OpenDDEDesignFeatures` bundle whose static binder extents and
        atom templates allow geometry to be refreshed from each candidate PSSM.
        """
        if binder_length < 1:
            raise ValueError("OpenDDE binder_length must be positive")
        binder = TargetChain(sequence="W" * binder_length, use_msa=False)
        feat, atom_array = self.target_only_features([binder, *chains])
        observed_length, binder_atom_alloc = _binder_extents(feat)
        if observed_length != binder_length:
            raise ValueError(
                "OpenDDE featurization changed the binder length: "
                f"expected {binder_length}, got {observed_length}"
            )
        design_features = OpenDDEDesignFeatures(
            features=feat,
            atom_templates=self.atom_templates,
            binder_length=observed_length,
            binder_atom_alloc=binder_atom_alloc,
        )
        return design_features, atom_array

    # ------------------------------------------------------------------ loss
    def build_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: OpenDDEDesignFeatures,
        recycling_steps: int = _DEFAULT_NUM_CYCLES,
        sampling_steps: int | None = None,
        stop_grad_conf_coords: bool = False,
    ) -> LossTerm:
        return self.build_multisample_loss(
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            num_samples=1,
            stop_grad_conf_coords=stop_grad_conf_coords,
        )

    def build_multisample_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: OpenDDEDesignFeatures,
        recycling_steps: int = _DEFAULT_NUM_CYCLES,
        sampling_steps: int | None = None,
        num_samples: int = 1,
        reduction=jnp.mean,
        stop_grad_conf_coords: bool = False,
    ) -> MultiSampleOpenDDELoss:
        if sampling_steps is None:
            sampling_steps = self.default_sampling_steps
        if not isinstance(features, OpenDDEDesignFeatures):
            raise TypeError(
                "OpenDDE design losses require OpenDDEDesignFeatures from "
                f"binder_features(); got {type(features).__name__}"
            )
        return MultiSampleOpenDDELoss(
            model=self.model,
            features=features,
            loss=loss,
            dense_atom_to_atom37=self.dense_atom_to_atom37,
            num_cycles=recycling_steps,
            n_step=sampling_steps,
            num_samples=num_samples,
            pae_bin_params=self.pae_bin_params,
            plddt_bin_params=self.plddt_bin_params,
            reduction=reduction,
            stop_grad_conf_coords=stop_grad_conf_coords,
        )

    # ------------------------------------------------------------------ forward
    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: Features | OpenDDEDesignFeatures,
        recycling_steps: int = _DEFAULT_NUM_CYCLES,
        sampling_steps: int | None = None,
        key,
    ):
        from mosaic.losses.opendde import opendde_forward_from_trunk

        if sampling_steps is None:
            sampling_steps = self.default_sampling_steps
        if PSSM is not None:
            # `features` is a design bundle; setting the binder sequence also
            # refreshes its atom + structural-token geometry from the PSSM.
            key, geom_key = jax.random.split(key)
            feat = set_binder_sequence(PSSM, features, geom_key)
        else:
            # No PSSM: fold as featurized (a bare Features, or an OpenDDEDesignFeatures'
            # untouched poly-Trp placeholder geometry).
            feat = (
                features.features if isinstance(features, OpenDDEDesignFeatures) else features
            )
        s_inputs, s, z = self.model.get_pairformer_output(feat, recycling_steps)
        return opendde_forward_from_trunk(
            self.model,
            feat,
            s_inputs,
            s,
            z,
            key,
            n_step=sampling_steps,
            dense_atom_to_atom37=self.dense_atom_to_atom37,
            pae_bin_params=self.pae_bin_params,
            plddt_bin_params=self.plddt_bin_params,
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features,
        writer=None,
        recycling_steps: int = _DEFAULT_NUM_CYCLES,
        sampling_steps: int | None = None,
        key,
    ) -> StructurePrediction:
        """Fold the (binder+)target complex and return scores + a structure,
        written from the canonical atom37 view (`StructureModelOutput.to_structure`).

        `features` is an `OpenDDEDesignFeatures` bundle (from `binder_features`) when
        `PSSM` is given -- the binder geometry is refreshed from the PSSM before
        folding -- or a bare `target_only_features` `Features` for a target-only fold."""
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=key,
        )
        output = jax.tree.map(np.asarray, output)
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
        return StructurePrediction(
            st=output.to_structure(),
            plddt=output.plddt,
            pae=output.pae,
            iptm=iptm,
            model_output=output,
        )


def _binder_extents(features) -> tuple[int, int]:
    """Static (host) binder token count + atom budget of a design featurization: chain-0
    (asym_id 0) token count, and the number of atoms belonging to those tokens.
    Used to parameterize the in-loop geometry refresh."""
    asym = np.asarray(features.asym_id)
    binder_length = int((asym == asym[0]).sum())
    a2t = np.asarray(features.atom_to_token_idx)
    binder_atom_alloc = int((a2t < binder_length).sum())
    return binder_length, binder_atom_alloc


# Atom templates are architecture data derived from OpenDDE CCD features.
_ATOM_TEMPLATE_ARCH = "opendde_v1"
_ATOM_TEMPLATE_SCHEMA_VERSION = 1
_ATOM_TEMPLATE_CACHE: dict[tuple[str, str, str, int], OpenDDEAtomTemplates] = {}


def _jopendde_build_id() -> str:
    """Return the installed JOpenDDE package revision or version."""
    try:
        dist = distribution("jopendde")
    except PackageNotFoundError:
        return "unknown"
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except json.JSONDecodeError:
            commit = None
        if commit:
            return commit
    return dist.version


def _atom_template_fields() -> list[str]:
    return [
        field.name
        for field in dataclasses.fields(OpenDDEAtomTemplates)
        if field.name not in ("max_atoms", "max_struct")
    ]


def _atom_template_shapes() -> dict[str, tuple[int, ...]]:
    atom = MAX_ATOMS_PER_RES
    struct = MAX_STRUCT_PER_RES
    return {
        "ref_pos": (2, 20, atom, 3),
        "ref_element": (2, 20, atom, 128),
        "ref_charge": (2, 20, atom),
        "ref_atom_name_chars": (2, 20, atom, 4, 64),
        "n_atoms": (2, 20),
        "disto_off": (2, 20),
        "pae_off": (2, 20),
        "a_struct_tok": (2, 20, atom),
        "a_struct_tokatom": (2, 20, atom),
        "s_disto_off": (2, 20, struct),
        "s_pae_off": (2, 20, struct),
        "s_frame_off": (2, 20, struct, 3),
        "s_valid": (2, 20, struct),
    }


def _validate_atom_template_cache(data, build_id: str) -> dict[str, np.ndarray]:
    metadata = {
        "_schema_version": str(_ATOM_TEMPLATE_SCHEMA_VERSION),
        "_architecture": _ATOM_TEMPLATE_ARCH,
        "_jopendde_build": build_id,
    }
    for name, expected in metadata.items():
        if name not in data or str(np.asarray(data[name]).item()) != expected:
            raise ValueError(f"invalid OpenDDE atom-template metadata: {name}")

    arrays = {}
    float_fields = {"ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars"}
    for name, shape in _atom_template_shapes().items():
        if name not in data:
            raise ValueError(f"missing OpenDDE atom-template field: {name}")
        value = np.asarray(data[name])
        if value.shape != shape:
            raise ValueError(
                f"invalid OpenDDE atom-template shape for {name}: "
                f"expected {shape}, got {value.shape}"
            )
        expected_kind = "f" if name in float_fields else "i"
        if value.dtype.kind != expected_kind:
            raise ValueError(
                f"invalid OpenDDE atom-template dtype for {name}: {value.dtype}"
            )
        arrays[name] = value
    return arrays


def _write_atom_template_cache(
    path: Path, templates: OpenDDEAtomTemplates, build_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: np.asarray(getattr(templates, name))
        for name in _atom_template_fields()
    }
    payload.update(
        {
            "_schema_version": np.asarray(str(_ATOM_TEMPLATE_SCHEMA_VERSION)),
            "_architecture": np.asarray(_ATOM_TEMPLATE_ARCH),
            "_jopendde_build": np.asarray(build_id),
        }
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_atom_template_cache(
    path: Path, build_id: str
) -> OpenDDEAtomTemplates:
    with np.load(path, allow_pickle=False) as data:
        arrays = _validate_atom_template_cache(data, build_id)
    return OpenDDEAtomTemplates(
        **{name: jnp.asarray(value) for name, value in arrays.items()}
    )


def _build_atom_template_cache(
    path: Path, build_id: str
) -> OpenDDEAtomTemplates:
    def featurize_one(sequence: str):
        features, _ = _featurize(
            [TargetChain(sequence=sequence, use_msa=False)]
        )
        return features

    templates = build_opendde_atom_templates(featurize_one)
    _write_atom_template_cache(path, templates, build_id)
    return templates


def _get_atom_templates() -> OpenDDEAtomTemplates:
    root = cache_dir()
    build_id = _jopendde_build_id()
    cache_key = (
        _ATOM_TEMPLATE_ARCH,
        str(root),
        build_id,
        _ATOM_TEMPLATE_SCHEMA_VERSION,
    )
    if cache_key in _ATOM_TEMPLATE_CACHE:
        return _ATOM_TEMPLATE_CACHE[cache_key]

    filename = (
        f"{_ATOM_TEMPLATE_ARCH}-v{_ATOM_TEMPLATE_SCHEMA_VERSION}-"
        f"{build_id[:12]}.npz"
    )
    path = root / "opendde" / "atom_templates" / filename
    if path.exists():
        try:
            templates = _load_atom_template_cache(path, build_id)
        except (OSError, ValueError, KeyError):
            path.unlink(missing_ok=True)
            templates = _build_atom_template_cache(path, build_id)
    else:
        templates = _build_atom_template_cache(path, build_id)

    _ATOM_TEMPLATE_CACHE[cache_key] = templates
    return templates


def _build_model(model_name: str, checkpoint_file: str | None = None) -> OpenDDEModel:
    from jopendde.inference import Predictor

    predictor = Predictor.from_checkpoint(
        model_name,
        checkpoint_file=checkpoint_file,
        asset_cache_dir=cache_dir() / "opendde",
    )
    sp = predictor.summary_params
    return OpenDDEModel(
        model=predictor.model,
        dense_atom_to_atom37=jnp.array(_build_dense_atom_to_atom37()),
        atom_templates=_get_atom_templates(),
        pae_bin_params=tuple(sp.pae_bins),
        plddt_bin_params=tuple(sp.plddt_bins),
    )


def OpenDDEModelV1() -> OpenDDEModel:
    return _build_model(
        "opendde_v1",
        checkpoint_file="opendde.pt",
    )


def OpenDDEModelAbag() -> OpenDDEModel:
    """Load the ABAG-optimized weights (`opendde_abag.pt`) for antibody-antigen
    complexes."""
    return _build_model(
        "opendde_v1",
        checkpoint_file="opendde_abag.pt",
    )
