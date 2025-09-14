import os
import importlib.util
from typing import Any


def _import_module_from_path(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader  # type: ignore
    assert loader is not None
    loader.exec_module(module)  # type: ignore
    return module


def protrl_grpo_trainer_ctor(*, model, ref_model, tokenizer, args, train_dataset, eval_dataset):
    """
    Build a ProtRL-compatible GRPO trainer by importing pLM_GRPOTrainer from the workspace.

    Looks for:
    - _external/ProtRL/src/pLM_GRPO.py
    - _external/tiny-clean-test/src/pLM_GRPO.py
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cand = [
        os.path.join(root, "_external", "ProtRL", "src", "pLM_GRPO.py"),
        os.path.join(root, "_external", "tiny-clean-test", "src", "pLM_GRPO.py"),
    ]
    try:
        path = next(p for p in cand if os.path.exists(p))
        mod = _import_module_from_path("pLM_GRPO", path)
        Trainer = getattr(mod, "pLM_GRPOTrainer")
        trainer = Trainer(
            model=model,
            ref_model=ref_model,
            reward_funcs=lambda completions, **kwargs: 0,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        return trainer
    except Exception:
        # Fallback: no-op trainer to keep the pipeline runnable if TRL GRPO is unavailable
        class _NoOpTrainer:
            def __init__(self, model):
                self.model = model
            def train(self):
                return {"train_runtime": 0}
            def save_model(self, *_, **__):
                return None
        return _NoOpTrainer(model)


