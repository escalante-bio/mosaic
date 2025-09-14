import os
import json
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from mosaic_workflows.design import run_workflow
from mosaic_workflows.rl import rl_grpo_adapter, protrl_grpo_trainer_ctor


def init_tiny_llama(root: str):
    # Resolve tiny-clean-test assets from provided root or baked /repo
    cand_roots = [root, "/repo", "/workspace"]
    base = next((r for r in cand_roots if os.path.exists(os.path.join(r, "_external", "tiny-clean-test"))), "/repo")
    tokenizer_dir = os.path.join(base, "_external", "tiny-clean-test", "models", "tokenizer")
    config_dir = os.path.join(base, "_external", "tiny-clean-test", "models", "size_config", "tiny")
    config_file = os.path.join(config_dir, "llama_config.json")
    with open(config_file, "r") as f:
        cfg = json.load(f)
    model = LlamaForCausalLM(LlamaConfig(**cfg))
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, add_eos_token=False, add_bos_token=True, use_fast=True)
    return model, tok


def build_loss():
    # Minimal toy loss: prefer alanine 'A' everywhere (as a proxy for testing)
    import equinox as eqx
    import jax.numpy as jnp
    class ToyLoss(eqx.Module):
        def __call__(self, probs, *, key):
            vocab = "ARNDCQEGHILKMFPSTWYV"
            a_idx = vocab.index("A")
            # negative mean prob of A (maximize A)
            val = -jnp.mean(probs[:, a_idx])
            return val, {"toy": {"pA": float(jnp.mean(probs[:, a_idx]))}}
    return ToyLoss()


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    binder_len = 50
    prompt = "M"

    def optimizer(**kw):
        return rl_grpo_adapter(
            **kw,
            init_model_and_tokenizer=lambda: init_tiny_llama(root),
            trainer_ctor=protrl_grpo_trainer_ctor,
            prompt=prompt,
            binder_len=binder_len,
            num_return=8,
            gen_cfg={"top_k": 8, "temperature": 0.9, "max_length": binder_len + 8},
            trainer_args={"logging_steps": 10, "num_train_epochs": 1},
        )

    phase = {
        "name": "rl",
        "build_loss": build_loss,
        "optimizer": optimizer,
        "steps": 2,
        "schedule": lambda g,p: {},
    }

    wf = {"phases": [phase], "binder_len": binder_len, "seed": 0}
    res = run_workflow(wf)
    print(res["best_sequence"])


if __name__ == "__main__":
    main()


