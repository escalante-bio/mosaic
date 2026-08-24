"""Configuration-driven structure-model registry for binder design.

The binder pipeline has two independent structure-prediction slots:

``hallucination``
    The differentiable model whose confidence/geometry objective shapes a new
    binder backbone.

``folding``
    The model used to refold and score finished binder sequences (including the
    evaluation-only workflow).

Keeping construction here gives both workflows one spelling and one validation
path for model names.  Imports are deliberately lazy: importing every backend
would initialize several large dependency stacks even when only AF2 is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AVAILABLE_MODELS",
    "ModelSelection",
    "create_structure_model",
    "model_selection",
]


AVAILABLE_MODELS = ("af2", "boltz2", "protenix-v2", "esmfold2")

_ALIASES = {
    "af2": "af2",
    "alphafold2": "af2",
    "alpha-fold2": "af2",
    "boltz2": "boltz2",
    "boltz-2": "boltz2",
    "protenix2": "protenix-v2",
    "protenix-v2": "protenix-v2",
    "protenix_v2": "protenix-v2",
    "esmfold2": "esmfold2",
    "esm-fold2": "esmfold2",
    "esmfold-2": "esmfold2",
}

_STAGE_KEYS = {
    "hallucination": ("hallucination_model", "design_model"),
    "folding": ("folding_model", "validation_model"),
}


def _canonical_name(value: Any) -> str:
    name = str(value or "af2").strip().lower().replace(" ", "-")
    try:
        return _ALIASES[name]
    except KeyError:
        choices = ", ".join(AVAILABLE_MODELS)
        raise ValueError(
            f"Unknown structure model '{value}'. Choose one of: {choices}"
        ) from None


@dataclass(frozen=True)
class ModelSelection:
    """One configured model slot and its model-specific options."""

    stage: str
    name: str
    options: dict[str, Any]

    @property
    def sampling_steps(self) -> int | None:
        value = self.options.get("sampling_steps")
        return None if value is None else int(value)

    @property
    def members(self) -> tuple[int, ...] | None:
        configured = self.options.get("members")
        if configured is not None:
            if isinstance(configured, (str, bytes)):
                raise TypeError(
                    f"{self.stage}_model_options.members must be a JSON array"
                )
            members = tuple(int(member) for member in configured)
            if not members:
                raise ValueError(f"{self.stage}_model_options.members cannot be empty")
            if len(members) > 5 or len(set(members)) != len(members):
                raise ValueError(
                    f"{self.stage}_model_options.members must contain 1-5 "
                    "distinct member indices"
                )
            if any(member < 0 or member > 4 for member in members):
                raise ValueError(
                    f"{self.stage}_model_options.members uses zero-based indices "
                    "from 0 through 4"
                )
            return members
        samples = self.options.get("num_samples")
        if samples is None:
            return None
        count = int(samples)
        if not 1 <= count <= 5:
            raise ValueError(
                f"{self.stage}_model_options.num_samples must be between 1 and 5"
            )
        return tuple(range(count))


def model_selection(settings: Mapping[str, Any], stage: str) -> ModelSelection:
    """Parse a hallucination/folding model selection from advanced settings.

    The concise form is a string::

        "hallucination_model": "boltz2"

    A mapping may keep the name and options together::

        "folding_model": {"name": "esmfold2", "sampling_steps": 20}

    Separate ``<stage>_model_options`` mappings are merged on top.  The legacy
    aliases ``design_model`` and ``validation_model`` are accepted so early
    experimental configs continue to load.
    """

    if stage not in _STAGE_KEYS:
        raise ValueError(f"Unknown model stage '{stage}'")

    raw: Any = None
    for key in _STAGE_KEYS[stage]:
        if settings.get(key) is not None:
            raw = settings[key]
            break
    if raw is None:
        raw = "af2"

    options: dict[str, Any]
    if isinstance(raw, Mapping):
        options = dict(raw)
        raw_name = options.pop("name", options.pop("model", "af2"))
    else:
        raw_name = raw
        options = {}

    extra = settings.get(f"{stage}_model_options", {}) or {}
    if not isinstance(extra, Mapping):
        raise TypeError(f"{stage}_model_options must be a JSON object")
    options.update(extra)
    return ModelSelection(stage=stage, name=_canonical_name(raw_name), options=options)


def create_structure_model(
    selection: ModelSelection,
    settings: Mapping[str, Any],
    *,
    multimer: bool | None = None,
    model_numbers: tuple[int, ...] | None = None,
):
    """Construct one registered model without importing unused backends."""

    options = selection.options
    if selection.name == "af2":
        from mosaic.models.af2 import AlphaFold2

        if selection.sampling_steps is not None:
            raise ValueError("AF2 does not support sampling_steps")

        data_dir = options.get("data_dir", options.get("params_dir"))
        if data_dir is None:
            data_dir = settings.get("af_params_dir")
        if multimer is None:
            setting = (
                "use_multimer_design"
                if selection.stage == "hallucination"
                else "use_multimer_validation"
            )
            multimer = bool(settings.get(setting, True))
        configured_numbers = options.get("model_numbers")
        if configured_numbers is not None:
            model_numbers = tuple(int(number) for number in configured_numbers)
            if (
                not model_numbers
                or len(set(model_numbers)) != len(model_numbers)
                or any(number < 1 or number > 5 for number in model_numbers)
            ):
                raise ValueError(
                    "AF2 model_numbers must contain distinct one-based values 1-5"
                )
        return AlphaFold2(
            data_dir=data_dir,
            multimer=bool(multimer),
            model_numbers=model_numbers,
        )

    if selection.name == "boltz2":
        from mosaic.models.boltz2 import Boltz2

        checkpoint = options.get("checkpoint_path", options.get("cache_path"))
        return Boltz2(cache_path=Path(checkpoint).expanduser() if checkpoint else None)

    if selection.name == "protenix-v2":
        from mosaic.models.protenix import ProtenixV2

        return ProtenixV2()

    if selection.name == "esmfold2":
        from mosaic.models.esmfold2 import ESMFold2Fast, ESMFold2Full

        variant = str(options.get("variant", "fast")).strip().lower()
        if variant == "fast":
            return ESMFold2Fast()
        if variant == "full":
            return ESMFold2Full()
        raise ValueError("esmfold2 variant must be 'fast' or 'full'")

    raise AssertionError(f"unhandled registered model: {selection.name}")
