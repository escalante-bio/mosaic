"""Public API for the mosaic_rl package."""

from .optimizers import grpo_logits, hf_grpo_optimizer, hf_dataset_grpo_optimizer
from . import regularizers
from .utils import sanitize_sequence, decode_argmax, sequence_to_one_hot
from .hf import build_hf_grpo_phase, build_hf_dataset_phase
from .experiments.protrl import ProtRLConfig, run_pipeline as run_protrl_pipeline

__all__ = [
    "grpo_logits",
    "hf_grpo_optimizer",
    "hf_dataset_grpo_optimizer",
    "build_hf_grpo_phase",
    "build_hf_dataset_phase",
    "regularizers",
    "sanitize_sequence",
    "decode_argmax",
    "sequence_to_one_hot",
    "ProtRLConfig",
    "run_protrl_pipeline",
]
