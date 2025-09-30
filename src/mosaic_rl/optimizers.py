"""RL optimisers that adhere to the Mosaic-Workflows optimiser contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .sampling import sample_categorical_sequences
# Regularizers should be passed as callables via schedule; avoid registries per Mosaic philosophy.

Array = jnp.ndarray

from dataclasses import dataclass


def _ensure_key(key):
    if key is None:
        return jax.random.key(np.random.randint(0, 1_000_000))
    return key


def _chain(functions: Sequence[Any], arr, ctx):
    if not functions:
        return arr
    for fn in functions:
        arr = fn(arr, ctx)
    return arr


def _apply_transforms(kind: str, transforms: Mapping[str, Any] | None, arr, ctx):
    if not transforms:
        return arr
    chain = transforms.get(kind)
    if isinstance(chain, Iterable) and not isinstance(chain, dict):
        return _chain(list(chain), arr, ctx)  # type: ignore[arg-type]
    return arr


def _schedule_dict(schedule, global_step: int, phase_step: int) -> Dict[str, Any]:
    if schedule is None:
        return {}
    if callable(schedule):
        return dict(schedule(global_step, phase_step))
    return dict(schedule)


def _reward_metrics(rewards: jnp.ndarray) -> Dict[str, float]:
    rewards_np = np.asarray(rewards)
    return {
        "reward/mean": float(rewards_np.mean()),
        "reward/std": float(rewards_np.std(ddof=0)) if rewards_np.size > 1 else 0.0,
        "reward/max": float(rewards_np.max()),
        "reward/min": float(rewards_np.min()),
    }


def _policy_entropy(probs: jnp.ndarray) -> float:
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8)) / probs.shape[0]
    return float(entropy)


def _regulariser_gradients(
    logits: Array,
    schedule: Mapping[str, Any],
    aux_context: Mapping[str, Any] | None,
) -> tuple[Array, Dict[str, float]]:
    regulariser_specs = schedule.get("regularizers") or []
    if not regulariser_specs:
        return jnp.zeros_like(logits), {}

    reference_logits = None
    if aux_context is not None:
        reference_logits = aux_context.get("reference_logits")
        if reference_logits is not None:
            reference_logits = jnp.asarray(reference_logits)

    grad_accum = jnp.zeros_like(logits)
    metrics: Dict[str, float] = {}
    total = jnp.array(0.0, dtype=logits.dtype)

    for entry in regulariser_specs:
        # Expect (callable, weight)
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError("regularizers must be a list of (callable, weight)")
        fn, weight = entry
        if not callable(fn):
            raise TypeError("regularizer spec must pass callable, not string name; avoid registries")
        penalty, grad, diagnostics = fn(logits, reference_logits, float(weight), dict(schedule))
        grad_accum = grad_accum + grad
        total = total + penalty
        metrics.update(diagnostics)

    metrics.setdefault("regularizer/total", float(total))
    return grad_accum, metrics


def grpo_logits(
    *,
    loss_function,
    x,
    n_steps,
    key=None,
    schedule=None,
    transforms=None,
    trajectory_fn=None,
    aux_context=None,
    update_loss_state: bool = False,
    **kwargs,
):
    """Policy-gradient optimiser that treats the supplied loss as ``-reward``.

    The implementation follows REINFORCE/GRPO style updates with an optional
    KL regulariser. It obeys the standard Mosaic-Workflows optimiser contract.
    """

    logits = jnp.asarray(x)
    binder_len, vocab = logits.shape
    best_logits = logits
    best_reward = -jnp.inf

    key = _ensure_key(key)
    aux_context = aux_context or {}

    global_step = int(aux_context.get("global_step", 0))

    for step in range(n_steps):
        sched = _schedule_dict(schedule, global_step + step, step)
        lr = float(sched.get("lr", sched.get("learning_rate", 1e-2)))
        temperature = float(sched.get("temperature", 1.0))
        num_samples = int(sched.get("num_samples", 4))

        ctx = {"schedule": sched, **aux_context}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)

        key, sample_key = jax.random.split(key)
        indices, probs = sample_categorical_sequences(logits, sample_key, num_samples=num_samples, temperature=temperature)

        # Evaluate rewards (negative losses) for each sample independently.
        reward_keys = jax.random.split(key, num_samples + 1)
        key = reward_keys[0]
        rewards_list: list[float] = []
        rows = []
        for i, sample_indices in enumerate(indices):
            sample_one_hot = jax.nn.one_hot(sample_indices, vocab, dtype=jnp.float32)
            value, aux = loss_function(sample_one_hot, key=reward_keys[i + 1])
            reward = -float(value)
            rewards_list.append(reward)
            rows.append({
                "sequence": sample_indices.tolist(),
                "reward": reward,
                "aux": aux,
            })

        rewards = jnp.asarray(rewards_list, dtype=jnp.float32)
        baseline = jnp.mean(rewards)
        advantages = rewards - baseline

        # Compute policy gradient.
        one_hots = jax.nn.one_hot(indices, vocab, dtype=jnp.float32)
        centered = one_hots - probs
        grad = jnp.mean(advantages[:, None, None] * centered, axis=0) / max(temperature, 1e-6)

        reg_grad, reg_metrics = _regulariser_gradients(logits, sched, aux_context)
        logits = logits + lr * (grad - reg_grad)
        logits = _apply_transforms("post_logits", transforms, logits, ctx)

        probs = jax.nn.softmax(logits, axis=-1)
        avg_reward = float(jnp.mean(rewards))
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_logits = logits

        metrics = _reward_metrics(rewards)
        metrics["policy/entropy"] = _policy_entropy(probs)
        metrics.update(reg_metrics)

        aux_payload = {
            "loss": -avg_reward,
            "metrics": metrics,
            "rewards": [float(r) for r in rewards],
            "rows": rows,
        }
        if trajectory_fn is not None:
            trajectory_fn(aux_payload, probs)

    return np.array(logits), np.array(best_logits), None


__all__ = ["grpo_logits"]


class _DatasetGRPOTrainer:
    """Lightweight wrapper around HF's GRPOTrainer for reward datasets."""

    def __init__(self, *, model, tokenizer, training_args, train_dataset, eval_dataset, optimizers=None):
        from trl import GRPOTrainer  # type: ignore

        # Provide a dummy reward function; dataset already stores rewards
        def reward_fn(completions, **unused):
            return [0.0 for _ in completions]

        kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "reward_funcs": reward_fn,
        }
        if optimizers is not None:
            kwargs["optimizers"] = optimizers

        self._trainer = GRPOTrainer(**kwargs)

    def train(self):
        self._trainer.train()

    def save_model(self, path: str) -> None:
        self._trainer.save_model(path)

    @property
    def model(self):
        return self._trainer.model


def hf_dataset_grpo_optimizer(
    *,
    loss_function,
    x,
    n_steps,
    key=None,
    schedule=None,
    transforms=None,
    trajectory_fn=None,
    aux_context=None,
    update_loss_state: bool = False,
    **kwargs,
):
    """Optimiser that trains a HF causal LM using precomputed reward datasets."""

    resources = loss_function
    if not isinstance(resources, dict) or resources.get("kind") != "hf_dataset_grpo":
        raise TypeError("hf_dataset_grpo_optimizer expects resources built by build_hf_dataset_phase")

    import torch
    from datasets import Dataset, load_from_disk  # type: ignore

    current_checkpoint = None
    if isinstance(x, dict):
        current_checkpoint = x.get("checkpoint")
    if current_checkpoint is None:
        current_checkpoint = resources.get("initial_checkpoint")

    model_loader = resources["model_loader"]
    tokenizer_loader = resources["tokenizer_loader"]

    model = model_loader(current_checkpoint)
    tokenizer = tokenizer_loader(current_checkpoint)

    device = torch.device(resources.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    training_args = resources["training_args"]
    optimizer_builder = resources.get("optimizer_builder")

    train_dataset_ref = resources["train_dataset"]
    eval_dataset_ref = resources["eval_dataset"]

    if isinstance(train_dataset_ref, (str, Path)):
        train_dataset = load_from_disk(str(train_dataset_ref))
    else:
        train_dataset = train_dataset_ref

    if isinstance(eval_dataset_ref, (str, Path)):
        eval_dataset = load_from_disk(str(eval_dataset_ref))
    else:
        eval_dataset = eval_dataset_ref

    optimizers = None
    if callable(optimizer_builder):
        optimizers = optimizer_builder(model)

    trainer = _DatasetGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=optimizers,
    )

    trainer.train()

    results_dir = resources.get("results_dir") or Path.cwd()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_name = resources.get("run_name", "hf_dataset_rl")
    latest_dir = results_dir / f"{run_name}_step{resources.get('step_index', 0)}"
    latest_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(latest_dir))
    tokenizer.save_pretrained(latest_dir)

    if trajectory_fn is not None:
        aux_payload = {
            "loss": 0.0,
            "metrics": resources.get("metrics", {}),
            "rows": [],
        }
        trajectory_fn(aux_payload, None)

    return {"checkpoint": str(latest_dir)}, {"checkpoint": str(latest_dir)}, None


__all__.append("hf_dataset_grpo_optimizer")


def _tokenize_pair(tokenizer, prompt: str, completion: str, *, device) -> tuple["torch.Tensor", "torch.Tensor", int]:  # type: ignore[name-defined]
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if completion_ids.numel() == 0:
        return None, None, 0  # type: ignore[return-value]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return input_ids, attention_mask, completion_ids.shape[1]


def hf_grpo_optimizer(
    *,
    loss_function,
    x,
    n_steps,
    key=None,
    schedule=None,
    transforms=None,
    trajectory_fn=None,
    aux_context=None,
    update_loss_state: bool = False,
    **kwargs,
):
    """Train a HF causal LM with a simple REINFORCE objective.

    The ``loss_function`` is expected to be the resource dictionary emitted by
    :func:`mosaic_rl.hf.build_hf_grpo_phase`.
    """

    resources = loss_function
    if not isinstance(resources, dict) or resources.get("kind") != "hf_grpo":
        raise TypeError("hf_grpo_optimizer expects resources built by build_hf_grpo_phase")

    import torch

    # Initial checkpoint or path
    current_checkpoint = None
    if isinstance(x, dict):
        current_checkpoint = x.get("checkpoint")
    if current_checkpoint is None:
        current_checkpoint = resources.get("initial_checkpoint")

    model_loader = resources["model_loader"]
    tokenizer_loader = resources["tokenizer_loader"]
    model = model_loader(current_checkpoint)
    tokenizer = tokenizer_loader(current_checkpoint)
    model.train()

    results_dir = resources.get("results_dir") or Path.cwd()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = resources.get("run_name", "hf_rl")

    global_step = int(aux_context.get("global_step", 0)) if aux_context else 0

    initial_sched = _schedule_dict(schedule, global_step, 0)
    device = torch.device(initial_sched.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    lr = float(initial_sched.get("lr", initial_sched.get("learning_rate", 1e-5)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_reward = -float("inf")
    best_dir: Path | None = None

    prompts_base: Sequence[str] = resources["prompts"]
    generations_default = int(resources.get("generations", 4))
    max_tokens_default = int(resources.get("max_new_tokens", 32))

    for step in range(n_steps):
        sched = _schedule_dict(schedule, global_step + step, step)
        lr = float(sched.get("lr", sched.get("learning_rate", lr)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        device = torch.device(sched.get("device", device))
        model.to(device)

        num_generations = int(sched.get("num_generations", generations_default))
        max_new_tokens = int(sched.get("max_new_tokens", max_tokens_default))
        temperature = float(sched.get("temperature", 1.0))
        top_k = int(sched.get("top_k", 0))

        prompts_batch: list[str] = []
        completions: list[str] = []

        model.eval()
        with torch.no_grad():
            for prompt in prompts_base:
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": max(1, num_generations),
                    "do_sample": True,
                    "temperature": max(temperature, 1e-6),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if top_k > 0:
                    gen_kwargs["top_k"] = top_k

                outputs = model.generate(**gen_kwargs)
                prompt_len = input_ids.shape[1]
                for row in outputs:
                    completion_ids = row[prompt_len:]
                    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    prompts_batch.append(prompt)
                    completions.append(completion)

        if not completions:
            continue

        rewards_raw = resources["scorer"](prompts_batch, completions)
        rewards = [float(r) for r in rewards_raw]

        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        valid = 0

        for prompt, completion, reward in zip(prompts_batch, completions, rewards):
            tensors = _tokenize_pair(tokenizer, prompt, completion, device=device)
            if tensors[0] is None:
                continue
            input_ids, attention_mask, comp_len = tensors
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            # Only completion tokens contribute because labels for the prompt are masked to -100.
            policy_loss = float(reward) * outputs.loss * comp_len
            total_loss = total_loss + policy_loss
            valid += 1

        if valid > 0:
            total_loss = total_loss / valid
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        rewards_np = np.asarray(rewards, dtype=np.float32)
        avg_reward = float(rewards_np.mean()) if rewards_np.size > 0 else 0.0
        metrics = _reward_metrics(rewards_np)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_dir = results_dir / f"{run_name}_best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.eval()
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            model.train()

        aux_payload = {
            "loss": float(total_loss.detach().cpu()) if valid else 0.0,
            "metrics": metrics,
            "rewards": rewards,
            "rows": [
                {"prompt": p, "completion": c, "reward": r}
                for p, c, r in zip(prompts_batch, completions, rewards)
            ],
        }
        if trajectory_fn is not None:
            trajectory_fn(aux_payload, completions)

    latest_dir = results_dir / f"{run_name}_step{global_step + n_steps}"
    latest_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(latest_dir)
    tokenizer.save_pretrained(latest_dir)

    if best_dir is None:
        best_dir = latest_dir

    return {"checkpoint": str(latest_dir)}, {"checkpoint": str(best_dir)}, None


__all__.append("hf_grpo_optimizer")
