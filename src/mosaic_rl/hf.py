"""Helpers for training HuggingFace causal LMs with Mosaic-style phases."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .optimizers import hf_grpo_optimizer, hf_dataset_grpo_optimizer


def _ensure_numpy_metadata() -> None:
    """Ensure importlib can resolve numpy version; do not swallow failures."""
    import importlib.metadata as metadata
    # If resolution fails, let it raise instead of silently patching.
    _ = metadata.version("numpy")


def _as_callable(obj, loader_factory: Callable[[str], Callable[[str | None], object]]):
    if callable(obj):
        return obj
    if isinstance(obj, str):
        return loader_factory(obj)
    raise TypeError("expected string checkpoint name or callable loader")


def _make_model_loader(model_id: str) -> Callable[[str | None], object]:
    def loader(checkpoint: str | None):
        _ensure_numpy_metadata()
        from transformers import AutoModelForCausalLM

        target = checkpoint or model_id
        return AutoModelForCausalLM.from_pretrained(target)

    return loader


def _make_tokenizer_loader(tokenizer_id: str) -> Callable[[str | None], object]:
    def loader(checkpoint: str | None):
        _ensure_numpy_metadata()
        from transformers import AutoTokenizer

        target = tokenizer_id or checkpoint  # prefer explicit tokenizer id over checkpoint
        tokenizer = AutoTokenizer.from_pretrained(target)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    return loader


def build_hf_grpo_phase(
    *,
    name: str,
    model: str | Callable[[str | None], object],
    tokenizer: str | Callable[[str | None], object] | None = None,
    prompts: Sequence[str],
    scorer: Callable[[Sequence[str], Sequence[str]], Sequence[float]],
    steps: int,
    generations: int = 4,
    max_new_tokens: int = 32,
    results_dir: str | Path | None = None,
    schedule: Mapping | Callable[[int, int], Mapping] | None = None,
    transforms: Mapping | None = None,
    analyzers: Sequence[Callable[[Mapping], Mapping]] | None = None,
) -> dict:
    """Return a phase dict that trains a HF causal LM with REINFORCE-style updates."""

    model_loader = _as_callable(model, _make_model_loader)
    if tokenizer is None:
        if not isinstance(model, str):
            raise ValueError("tokenizer must be provided when model is a callable loader")
        tokenizer_loader = _make_tokenizer_loader(model)
    else:
        tokenizer_loader = _as_callable(tokenizer, _make_tokenizer_loader)

    resources = {
        "kind": "hf_grpo",
        "model_loader": model_loader,
        "tokenizer_loader": tokenizer_loader,
        "initial_checkpoint": model if isinstance(model, str) else None,
        "prompts": list(prompts),
        "scorer": scorer,
        "generations": int(generations),
        "max_new_tokens": int(max_new_tokens),
        "results_dir": Path(results_dir) if results_dir is not None else None,
        "run_name": name,
    }

    def build_loss():
        return resources

    return {
        "name": name,
        "build_loss": build_loss,
        "optimizer": hf_grpo_optimizer,
        "steps": int(steps),
        "schedule": schedule,
        "transforms": transforms or {},
        "analyzers": analyzers or [],
    }


__all__ = ["build_hf_grpo_phase"]


def build_hf_dataset_phase(
    *,
    name: str,
    model: str | Callable[[str | None], object],
    tokenizer: str | Callable[[str | None], object] | None = None,
    train_dataset,
    eval_dataset,
    training_args,
    results_dir: str | Path | None = None,
    schedule: Mapping | Callable[[int, int], Mapping] | None = None,
    transforms: Mapping | None = None,
    analyzers: Sequence[Callable[[Mapping], Mapping]] | None = None,
    step_index: int = 0,
    optimizer_builder: Callable[[object], tuple[object, object]] | None = None,
) -> dict:
    """Phase that trains a HF causal LM from a precomputed reward dataset."""

    model_loader = _as_callable(model, _make_model_loader)
    if tokenizer is None:
        if not isinstance(model, str):
            raise ValueError("tokenizer must be provided when model is a callable loader")
        tokenizer_loader = _make_tokenizer_loader(model)
    else:
        tokenizer_loader = _as_callable(tokenizer, _make_tokenizer_loader)

    resources = {
        "kind": "hf_dataset_grpo",
        "model_loader": model_loader,
        "tokenizer_loader": tokenizer_loader,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "training_args": training_args,
        "results_dir": Path(results_dir) if results_dir is not None else None,
        "run_name": name,
        "step_index": step_index,
        "initial_checkpoint": model if isinstance(model, str) else None,
        "optimizer_builder": optimizer_builder,
    }

    def build_loss():
        return resources

    return {
        "name": name,
        "build_loss": build_loss,
        "optimizer": hf_dataset_grpo_optimizer,
        "steps": 1,
        "schedule": schedule,
        "transforms": transforms or {},
        "analyzers": analyzers or [],
    }


__all__.append("build_hf_dataset_phase")
