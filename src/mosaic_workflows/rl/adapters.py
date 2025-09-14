import numpy as np
from typing import Callable, Any

from datasets import Dataset

from .utils import (
    sanitize_sequence,
    sequence_to_one_hot,
    one_hot_to_crisp_logits,
    build_reward_fn_from_loss,
)


def _filter_and_trim(seqs, binder_len: int):
    out = []
    for s in seqs:
        t = sanitize_sequence(s)
        if len(t) == binder_len:
            out.append(t)
    return out


def rl_custom_adapter(
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
    init_model_and_tokenizer: Callable[[], tuple[Any, Any]],
    generate_fn: Callable[[Any, Any, str, int, dict], list[str]],
    update_fn: Callable[[Any, Any, list[str], list[float], dict], tuple[Any, dict]],
    prompt: str,
    binder_len: int,
    num_return: int = 8,
    gen_cfg: dict | None = None,
    step_cfg: dict | None = None,
):
    model, tokenizer = init_model_and_tokenizer()
    reward_fn = build_reward_fn_from_loss(loss_function, binder_len=binder_len)

    best_seq = None
    best_reward = -1e9

    for step in range(int(n_steps)):
        seqs = generate_fn(model, tokenizer, prompt, int(num_return), gen_cfg or {})
        seqs = _filter_and_trim(seqs, binder_len)
        rewards = reward_fn(seqs)
        model, metrics = update_fn(model, tokenizer, seqs, rewards, step_cfg or {})

        if seqs:
            i = int(np.argmax(rewards))
            if float(rewards[i]) > best_reward:
                best_reward = float(rewards[i])
                best_seq = seqs[i]

        if trajectory_fn is not None:
            aux = {"rl_step": step, "avg_reward": float(np.mean(rewards) if rewards else 0.0), "metrics": metrics}
            traj_x = None
            try:
                if best_seq is not None:
                    oh = sequence_to_one_hot(best_seq, binder_len)
                    traj_x = oh
            except Exception:
                traj_x = None
            try:
                trajectory_fn(aux, traj_x)
            except Exception:
                pass

    if best_seq is None:
        return x, x, None

    oh = sequence_to_one_hot(best_seq, binder_len)
    logits = one_hot_to_crisp_logits(oh)
    return logits, logits, None


def rl_trl_adapter(
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
    init_model_and_tokenizer: Callable[[], tuple[Any, Any]],
    trainer_ctor: Callable[..., Any],
    prompt: str,
    binder_len: int,
    num_return: int = 8,
    gen_cfg: dict | None = None,
    trainer_args: dict | None = None,
):
    from transformers import AutoModelForCausalLM

    model, tokenizer = init_model_and_tokenizer()
    reward_fn = build_reward_fn_from_loss(loss_function, binder_len=binder_len)

    best_seq = None
    best_reward = -1e9

    for step in range(int(n_steps)):
        # Simple on-policy data: generate with current model via HF generate
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        gen = model.generate(
            input_ids,
            top_k=int((gen_cfg or {}).get("top_k", 16)),
            temperature=float((gen_cfg or {}).get("temperature", 0.9)),
            max_length=int((gen_cfg or {}).get("max_length", binder_len + input_ids.shape[-1] + 4)),
            do_sample=True,
            num_return_sequences=int(num_return),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        )
        prompt_len = input_ids.shape[-1]
        seqs = [tokenizer.decode(ids[prompt_len:], skip_special_tokens=True) for ids in gen]
        seqs = _filter_and_trim(seqs, binder_len)

        rewards = reward_fn(seqs)

        rows = [{"prompt": prompt, "completion": s, "reward": float(r)} for s, r in zip(seqs, rewards)]
        ds = Dataset.from_list(rows)
        split = ds.train_test_split(test_size=0.2, seed=42, shuffle=True)
        tr, ev = split["train"], split["test"]

        trainer = trainer_ctor(
            model=model,
            ref_model=model,  # pass same by default; ctor may build its own ref
            tokenizer=tokenizer,
            args=(trainer_args or {}),
            train_dataset=tr,
            eval_dataset=ev,
        )
        trainer.train()
        try:
            model = trainer.model
        except Exception:
            pass

        if seqs:
            i = int(np.argmax(rewards))
            if float(rewards[i]) > best_reward:
                best_reward = float(rewards[i])
                best_seq = seqs[i]

        if trajectory_fn is not None:
            aux = {"rl_step": step, "avg_reward": float(np.mean(rewards) if rewards else 0.0)}
            try:
                trajectory_fn(aux, None)
            except Exception:
                pass

    if best_seq is None:
        return x, x, None

    oh = sequence_to_one_hot(best_seq, binder_len)
    logits = one_hot_to_crisp_logits(oh)
    return logits, logits, None


def rl_grpo_adapter(**kwargs):
    return rl_trl_adapter(**kwargs)


def rl_wdpo_adapter(**kwargs):
    return rl_trl_adapter(**kwargs)


