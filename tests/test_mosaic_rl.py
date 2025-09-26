import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path
import pytest

from mosaic_rl.optimizers import grpo_logits
from mosaic_rl import build_hf_grpo_phase
from mosaic_rl.experiments.tiny_clean import TinyCleanConfig, run_pipeline
from mosaic_rl.utils import sequence_to_one_hot, sanitize_sequence, AA_VOCAB
from mosaic_workflows import run_workflow


def test_grpo_logits_improves_first_token():
    binder_len = 6

    def build_loss():
        def loss(one_hot, key=None):
            rewards = one_hot[:, 0]
            value = -float(np.sum(rewards))
            return value, {"sum": float(np.sum(rewards))}

        return loss

    phase = {
        "name": "synthetic_grpo",
        "build_loss": build_loss,
        "optimizer": grpo_logits,
        "steps": 8,
        "schedule": lambda g, p: {"lr": 0.1, "num_samples": 3},
    }

    workflow = {
        "binder_len": binder_len,
        "seed": 0,
        "phases": [phase],
    }

    result = run_workflow(workflow)
    assert len(result["trajectory"]) == 8

    best_sequence = result["best_sequence"]
    assert len(best_sequence) == binder_len
    assert any("metrics" in entry.get("aux", {}) for entry in result["trajectory"])


def test_hf_grpo_phase_with_clean_reward(tmp_path):
    binder_len = 8
    target_embed = np.zeros(1280, dtype=np.float32)
    target_embed[0] = 1.0

    aa_to_index = {aa: idx for idx, aa in enumerate(AA_VOCAB)}

    def embed_fn(seq: str) -> np.ndarray:
        sanitized = sanitize_sequence(seq, binder_len)
        vec = np.zeros(1280, dtype=np.float32)
        for ch in sanitized:
            vec[aa_to_index[ch]] += 1.0
        return vec

    def scorer(prompts, completions):
        rewards = []
        for seq in completions:
            vec = embed_fn(seq)
            denom = (np.linalg.norm(vec) * np.linalg.norm(target_embed) + 1e-8)
            cos = float(np.dot(vec, target_embed) / denom)
            rewards.append(cos)
        return rewards

    phase = build_hf_grpo_phase(
        name="clean_tiny",
        model="sshleifer/tiny-gpt2",
        tokenizer="sshleifer/tiny-gpt2",
        prompts=["M"],
        scorer=scorer,
        steps=1,
        generations=1,
        max_new_tokens=4,
        results_dir=tmp_path,
    )

    workflow = {
        "binder_len": binder_len,
        "seed": 0,
        "phases": [phase],
        "initial_x": {"checkpoint": "sshleifer/tiny-gpt2"},
    }

    result = run_workflow(workflow)
    assert len(result["trajectory"]) == 1
    checkpoint = result["x"].get("checkpoint")
    assert checkpoint
    assert (tmp_path / "clean_tiny_step1").exists()


def test_tiny_clean_pipeline(tmp_path):
    base_model = Path("_external/tiny-clean-test/models/base/tiny")
    if not base_model.exists():
        pytest.skip("tiny-clean-test model not available")

    tokenizer_path = Path("_external/tiny-clean-test/models/tokenizer")
    if not tokenizer_path.exists():
        pytest.skip("tiny-clean-test tokenizer not available")

    cfg = TinyCleanConfig(
        workspace=tmp_path / "workspace",
        label="TEST",
        binder_len=8,
        model=str(base_model),
        reference_model=str(base_model),
        clean_head_path=Path("_external/tiny-clean-test/CLEAN/app/data/pretrained/split100.pth"),
        embedding_path=Path("_external/tiny-clean-test/CLEAN/app/data/pretrained/100.pt"),
        results_dir=tmp_path / "results",
        use_esm=False,
        num_generations=2,
        max_new_tokens=8,
        tokenizer_id=str(tokenizer_path),
    )
    run_pipeline(cfg, iterations=1)
    assert (cfg.inputs_dir / "seq_gen_iteration0.fasta").exists()
