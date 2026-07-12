"""Mosaic `StructurePredictionModel` wrapper for OpenDDE (the `jopendde` JAX
port of Aureka's OpenDDE — an AF3-style all-atom co-folding model).

`OpenDDEModelV1()` loads + converts the released checkpoint via
`jopendde.Predictor` (only to build the JAX model's weights) and exposes mosaic's
standard interface. This mirrors `models/of3.py`. Featurization is a separate,
weight-free step (`_featurize`): it runs OpenDDE's torch data pipeline as plain
function calls -- no model object, no checkpoint -- so the integration
environment needs `torch` + the `opendde` package to featurize, but only the
checkpoint to run the forward. The design loop itself is pure JAX. See
`mosaic.losses.opendde` for the soft-sequence plumbing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from jopendde.model import OpenDDE as JaxOpenDDE

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.opendde import (
    MultiSampleOpenDDELoss,
    OpenDDEFeatures,
    OpenDDEResidueTemplates,
    build_opendde_templates,
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

# opendde_v1 relative-position-encoding constants (config
# `model.relative_position_encoding`). `generate_relp` is weight-free one-hot
# index math, so featurization needs only these two ints -- never the checkpoint.
# They're architecture-level: identical for OpenDDEModelV1 and OpenDDEModelAbag.
_RELP_R_MAX = 32
_RELP_S_MAX = 2

# Cache the torch-bound Predictor (checkpoint weights) so a repeated model build
# doesn't reload it. Keyed by (model_name, checkpoint_file). Used ONLY to build
# the JAX model's weights at construction; featurization is weight-free and needs
# no Predictor (see `_featurize`).
_PREDICTOR_CACHE: dict[tuple[str, str | None], object] = {}

_MSA_SERVER = "https://api.colabfold.com"
_MSA_CACHE = Path(os.environ.get("MOSAIC_MSA_CACHE", "~/.cache/mosaic/msa")).expanduser()


def _target_a3m(sequence: str) -> str:
    """Fetch (and cache) an unpaired a3m for `sequence` from the ColabFold MMseqs2
    server, returning the a3m path. Cached by sequence hash under
    `MOSAIC_MSA_CACHE`. Only called for chains with `use_msa=True`."""
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    _MSA_CACHE.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    a3m_path = _MSA_CACHE / f"{seq_hash}.a3m"
    if not a3m_path.exists():
        result = run_mmseqs2(
            [sequence],
            prefix=str(_MSA_CACHE / seq_hash),
            use_env=True,
            use_filter=True,
            use_pairing=False,
            host_url=_MSA_SERVER,
        )
        # run_mmseqs2 returns list[str] or a tuple of lists depending on version.
        a3m_strings = result[0] if isinstance(result, tuple) else result
        a3m_path.write_text(a3m_strings[0])
    return str(a3m_path)


def _get_predictor(model_name: str, checkpoint_file: str | None = None):
    key = (model_name, checkpoint_file)
    if key not in _PREDICTOR_CACHE:
        from jopendde.inference import Predictor

        _PREDICTOR_CACHE[key] = Predictor.from_checkpoint(
            model_name, checkpoint_file=checkpoint_file
        )
    return _PREDICTOR_CACHE[key]


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
                "OpenDDE wrapper supports protein chains only; got "
                f"{c.polymer_type}"
            )
        if c.template_chain is not None:
            raise NotImplementedError("OpenDDE wrapper does not support templates.")
        chain = {"sequence": c.sequence, "count": 1}
        if c.use_msa:
            chain["unpairedMsaPath"] = _target_a3m(c.sequence)
        sequences.append({"proteinChain": chain})
    return [{"name": name, "modelSeeds": [seed], "sequences": sequences}]


# ---------------------------------------------------------------------------
# Weight-free featurization
# ---------------------------------------------------------------------------
# Featurization is a *setup*-time operation (build the `Features` once; the JAX
# design loop never re-featurizes), and it needs no model weights. Upstream
# `Predictor.featurize` only reaches into the torch model for
# `relative_position_encoding.generate_relp` -- a `torch.no_grad` block of pure
# one-hot index math with no learned params. So we run featurization as plain
# function calls (the torch data pipeline + `generate_relp` + the
# `update_input_feature_dict` helper), and the wrapper holds no torch object.

_FEATURIZE_CONFIGS = None  # opendde_v1 inference config; built once, holds no weights.


def _featurize_configs():
    """The `opendde_v1` inference config that drives the featurization data
    pipeline. Built once per process from `build_inference_config` (no checkpoint
    load) and reused. It's architecture-level -- identical for V1 and Abag (only
    the checkpoint weights differ) -- so featurization needs no per-model input."""
    global _FEATURIZE_CONFIGS
    if _FEATURIZE_CONFIGS is None:
        from opendde.config.inference import (
            build_inference_config,
            update_gpu_compatible_configs,
        )
        from opendde.utils.download import download_inference_cache

        cfg = build_inference_config(model_name="opendde_v1", fill_required_with_null=True)
        cfg.use_msa = cfg.use_template = cfg.use_rna_msa = False
        cfg.triangle_multiplicative = cfg.triangle_attention = "torch"
        cfg.enable_diffusion_shared_vars_cache = False
        cfg.enable_efficient_fusion = False
        cfg = update_gpu_compatible_configs(cfg)
        download_inference_cache(cfg)  # CCD reference data on disk (no weights loaded)
        _FEATURIZE_CONFIGS = cfg
    return _FEATURIZE_CONFIGS


def _relp(feat_dict: dict) -> dict:
    """Add the weight-free relative-position feature `relp` to a raw feature dict.

    `RelativePositionEncoding.generate_relp` reads only `self.r_max` / `self.s_max`
    (pure one-hot index math, no learned params), so we invoke it as a function
    with the opendde_v1 constants instead of constructing the torch module."""
    import types

    from opendde.model.opendde import RelativePositionEncoding

    ns = types.SimpleNamespace(r_max=_RELP_R_MAX, s_max=_RELP_S_MAX)
    return RelativePositionEncoding.generate_relp(ns, feat_dict, lazy=False)


def _featurize(chains: list[TargetChain]):
    """Featurize protein `chains` into a jopendde `Features`, weight-free.

    Mirrors `Predictor.featurize` / `_load_batch` minus the model touches: run the
    torch data pipeline (`get_inference_dataloader`, driven by the shared config),
    add `relp` (`_relp`), rebuild the windowed atom-pair features
    (`update_input_feature_dict`), and adapt to `Features`. Returns
    `(feat, atom_array)`; `atom_array` is the biotite writer for native output."""
    import copy

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
        assert not err, err
    raw = copy.deepcopy(data["input_feature_dict"])
    raw["inference_seed"] = torch.tensor(seed, dtype=torch.long)
    with torch.no_grad():
        raw = _relp(raw)
    raw = update_input_feature_dict(raw)
    return Features.from_dict(_to_numpy(raw)), atom_array


class OpenDDEModel(StructurePredictionModel):
    """OpenDDE wrapped behind mosaic's structure-prediction interface."""

    model: JaxOpenDDE
    dense_atom_to_atom37: Int[Array, "32 Adense"]
    # Per-residue atom/structural templates for the in-loop binder geometry
    # refresh. Built once at construction; None disables the refresh (binder
    # keeps its poly-Trp placeholder geometry).
    templates: OpenDDEResidueTemplates | None = None
    model_name: str = eqx.field(static=True, default="opendde_v1")
    # Overriding weight file (e.g. "opendde_abag.pt"); None uses the default for
    # model_name. Only affects which cached Predictor featurization reloads.
    checkpoint_file: str | None = eqx.field(static=True, default=None)
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
        native structure output. Weight-free -- runs the data pipeline directly
        (`_featurize`), no Predictor / checkpoint needed."""
        return _featurize(chains)

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        """Design features: a poly-Trp binder as chain 0, then the targets. The
        binder occupies token positions `[0, binder_length)` / asym_id 0. Trp is
        the largest residue, so this allocates the maximum atom budget per
        position; the in-loop `refresh_binder_geometry` then tight-packs the
        designed residues' atoms into it each step without a recompile (any
        shorter sequence fits). Pin a partially-fixed framework with
        `SetPositions` during optimization.

        Returns `OpenDDEFeatures` (not a bare jopendde `Features`): it bundles the
        poly-Trp features with the per-residue templates + binder extents, so
        `set_binder_sequence` always refreshes the binder geometry from the
        designed PSSM (see `OpenDDEFeatures`)."""
        binder = TargetChain(sequence="W" * binder_length, use_msa=False)
        feat, atom_array = self.target_only_features([binder, *chains])
        return self._as_opendde_features(feat), atom_array

    def _as_opendde_features(self, features) -> OpenDDEFeatures:
        """Bundle a poly-Trp jopendde `Features` (or pass existing `OpenDDEFeatures`
        through) into `OpenDDEFeatures`, attaching the model's templates + the
        (static, host-computed) binder token/atom extents the in-loop refresh needs."""
        if isinstance(features, OpenDDEFeatures):
            return features
        if self.templates is None:
            raise ValueError(
                "OpenDDE design needs per-residue templates for the in-loop "
                "geometry refresh, but none were built for this model."
            )
        binder_length, binder_atom_alloc = _binder_extents(features)
        return OpenDDEFeatures(
            features=features,
            templates=self.templates,
            binder_length=binder_length,
            binder_atom_alloc=binder_atom_alloc,
        )

    # ------------------------------------------------------------------ loss
    def build_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features,
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
        features,
        recycling_steps: int = _DEFAULT_NUM_CYCLES,
        sampling_steps: int | None = None,
        num_samples: int = 1,
        reduction=jnp.mean,
        stop_grad_conf_coords: bool = False,
    ) -> MultiSampleOpenDDELoss:
        if sampling_steps is None:
            sampling_steps = self.default_sampling_steps
        # `features` is a design bundle (poly-Trp binder as chain 0, from
        # `binder_features`); `_as_opendde_features` also wraps a bare `Features`,
        # attaching templates + the (static, host-computed) binder extents. The
        # in-loop geometry refresh is always on -- `set_binder_sequence` refreshes
        # the binder side chains every step.
        return MultiSampleOpenDDELoss(
            model=self.model,
            features=self._as_opendde_features(features),
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
        features,
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
            # No PSSM: fold as featurized (a bare Features, or an OpenDDEFeatures'
            # untouched poly-Trp placeholder geometry).
            feat = features.features if isinstance(features, OpenDDEFeatures) else features
        s_inputs, s, z = self.model.get_pairformer_output(feat, recycling_steps)
        return opendde_forward_from_trunk(
            self.model, feat, s_inputs, s, z, key,
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

        `features` is an `OpenDDEFeatures` bundle (from `binder_features`) when
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


# Residue templates are CCD conformers + structural-token layout read off the
# (weight-free, architecture-level) featurizer -- they do NOT depend on the
# checkpoint, so a single set is shared across all `opendde_v1` checkpoints (V1,
# Abag, ...). Keyed by the featurization architecture only, matching `_featurize`'s
# hardcoded config. Backed by a small on-disk npz so a fresh process skips the
# ~40-featurization host build.
_TEMPLATE_ARCH = "opendde_v1"
_TEMPLATE_CACHE: OpenDDEResidueTemplates | None = None
_TEMPLATE_DISK_CACHE = Path(
    os.environ.get("MOSAIC_OPENDDE_TEMPLATE_CACHE", "~/.cache/mosaic/opendde_templates")
).expanduser()


def _template_fields():
    return [
        f.name for f in dataclasses.fields(OpenDDEResidueTemplates)
        if f.name not in ("max_atoms", "max_struct")
    ]


def _get_templates() -> OpenDDEResidueTemplates:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is not None:
        return _TEMPLATE_CACHE

    disk = _TEMPLATE_DISK_CACHE / f"{_TEMPLATE_ARCH}.npz"
    if disk.exists():
        data = np.load(disk)
        tmpl = OpenDDEResidueTemplates(**{k: jnp.asarray(data[k]) for k in _template_fields()})
    else:
        # Weight-free: the templates are CCD conformers read off the featurization
        # data pipeline, so building them never loads the checkpoint.
        def featurize_one(seq: str):
            feat, _ = _featurize([TargetChain(sequence=seq, use_msa=False)])
            return feat

        tmpl = build_opendde_templates(featurize_one)
        disk.parent.mkdir(parents=True, exist_ok=True)
        np.savez(disk, **{k: np.asarray(getattr(tmpl, k)) for k in _template_fields()})

    _TEMPLATE_CACHE = tmpl
    return tmpl


def _build_model(model_name: str, checkpoint_file: str | None = None) -> OpenDDEModel:
    predictor = _get_predictor(model_name, checkpoint_file)
    sp = predictor.summary_params
    return OpenDDEModel(
        model=predictor.model,
        dense_atom_to_atom37=jnp.array(_build_dense_atom_to_atom37()),
        templates=_get_templates(),
        model_name=model_name,
        checkpoint_file=checkpoint_file,
        pae_bin_params=tuple(sp.pae_bins),
        plddt_bin_params=tuple(sp.plddt_bins),
    )


def OpenDDEModelV1() -> OpenDDEModel:
    """Load the released `opendde_v1` checkpoint (single-sequence, no MSA/template)
    as a mosaic backend."""
    return _build_model("opendde_v1")


def OpenDDEModelAbag() -> OpenDDEModel:
    """Load the ABAG-optimized weights (`opendde_abag.pt`) for antibody-antigen
    complexes. Same `opendde_v1` architecture, different checkpoint."""
    return _build_model("opendde_v1", checkpoint_file="opendde_abag.pt")
