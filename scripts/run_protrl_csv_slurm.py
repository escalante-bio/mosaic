#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _add_paths(workspace: Path) -> None:
    src = workspace / "src"
    if src.exists():
        import sys
        sys.path.insert(0, str(src))


def _parse_weights(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in str(spec).split(','):
        if not item.strip():
            continue
        k, v = item.split('=')
        out[k.strip()] = float(v)
    return out


def _build_yaml(binder_len: int, ligand_smiles: str | None) -> str:
    lines = ["version: 1", "sequences:"]
    lines.append(f"  - protein:\n      id: A\n      sequence: {'X'*binder_len}\n      msa: empty")
    if ligand_smiles is not None and len(str(ligand_smiles)) > 0:
        lines.append(f"  - ligand:\n      id: L\n      smiles: '{ligand_smiles}'")
    return "\n".join(lines)


def _init_worker(device_id: int, xla_cuda_dir: str | None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["JAX_PLATFORMS"] = "cuda"
    # Route XLA to libdevice if not globally set
    if xla_cuda_dir:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_cuda_data_dir={xla_cuda_dir}"
    # Delay heavy imports until after env is set
    global _jax, _jnp, _TOKENS, _load_b2, _b2_load, _b2_set, _B2Out
    import jax as _jax  # type: ignore
    import jax.numpy as _jnp  # type: ignore
    from mosaic.common import TOKENS as _TOKENS  # type: ignore
    from mosaic.losses.boltz2 import (  # type: ignore
        load_boltz2 as _load_b2,
        load_features_and_structure_writer as _b2_load,
        set_binder_sequence as _b2_set,
        Boltz2Output as _B2Out,
    )


def _worker_score_one(args: Tuple[str, Dict[str, float], str | None, str | None, str | None]) -> Dict[str, Any]:
    seq, weights, ligand_smiles, cache_dir, xla_cuda_dir = args
    # Ensure worker is initialized
    try:
        _ = _jax  # type: ignore[name-defined]
    except NameError:
        _init_worker(0, xla_cuda_dir)

    binder_len = len(seq)
    vocab = _TOKENS['order']  # type: ignore[name-defined]
    onehot = _jnp.zeros((binder_len, 20), dtype=_jnp.float32)  # type: ignore[name-defined]
    idx_map = {aa: i for i, aa in enumerate(vocab)}
    for i, ch in enumerate(seq):
        if ch in idx_map:
            onehot = onehot.at[i, idx_map[ch]].set(1.0)  # type: ignore[attr-defined]

    yaml_str = _build_yaml(binder_len, ligand_smiles)
    features, _ = _b2_load(  # type: ignore[name-defined]
        yaml_str,
        cache=Path(str(cache_dir or os.environ.get("BOLTZ_CACHE", "/root/.boltz"))).expanduser(),
    )
    features = _b2_set(onehot, features)  # type: ignore[name-defined]
    model = _load_b2()  # type: ignore[name-defined]
    out = _B2Out(joltz2=model, features=features, deterministic=True, key=_jax.random.PRNGKey(0))  # type: ignore[name-defined]

    # Metrics
    plddt = float(out.plddt[:binder_len].mean())
    iptm = float(out.iptm)

    return {"boltz2_plddt": plddt, "boltz2_iptm": iptm}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--model-id", type=str, default="AI4PD/ZymCTRL")
    p.add_argument("--ref-model", type=str, default=None)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--designs", type=int, default=2)
    p.add_argument("--reward", type=str, default="boltz2_plddt=1.0,boltz2_iptm=1.0,length100=-1.0")
    # Optional CSV predictors / CLEAN
    p.add_argument("--csv-predictors", action="store_true")
    p.add_argument("--csv-cols", type=str, default="total_score,efield_score,ncaa_interface_score")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--clean-ec-label", type=str, default=None)
    p.add_argument("--clean-head-path", type=str, default=None)
    p.add_argument("--clean-embedding-path", type=str, default=None)
    p.add_argument("--clean-labels-path", type=str, default=None)
    p.add_argument("--esm-model-id", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--ligand-smiles", type=str, default=None)
    p.add_argument("--reward-devices", type=str, default="0,1,2,3")
    p.add_argument("--xla-cuda-dir", type=str, default=os.environ.get("XLA_CUDA_DIR", "/usr/local/cuda"))
    p.add_argument("--max-new-tokens", type=int, default=600)
    args = p.parse_args()

    # Paths
    workspace = Path(__file__).resolve().parents[1]
    _add_paths(workspace)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Env
    os.environ.setdefault("BOLTZ_CACHE", str(out_dir / "boltz_cache"))
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # HF/TRL
    from datasets import Dataset  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    from trl import GRPOConfig as _GRPOConfig, GRPOTrainer as _GRPOTrainer  # type: ignore

    # Minimal dataset of prompts (prompt text unused by reward; model learns to produce sequences)
    import pandas as _pd
    df = _pd.read_csv(args.csv)
    prompts = [str(x) for x in df.get("prompt", ["Design a protein sequence:"] * len(df))]
    dataset = Dataset.from_dict({"prompt": prompts})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to("cuda")

    weights = _parse_weights(args.reward)
    devices = [int(x.strip()) for x in str(args.reward_devices).split(',') if x.strip()]

    # Create a per-GPU worker pool for Boltz2
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    # Round-robin assignment handled by starmap input
    worker_args: List[Tuple[str, Dict[str, float], str | None, str | None, str | None]] = []

    # Build reward via mosaic_rl helpers
    from mosaic_rl.rewards import compose_reward
    reward_opts = {
        "weights": weights,
        "csv_path": args.csv if args.csv_predictors or args.clean else None,
        "csv_pred_cols": tuple([c.strip() for c in str(args.csv_cols).split(',') if c.strip()]),
        "use_csv_predictors": bool(args.csv_predictors),
        "use_clean": bool(args.clean),
        "clean_ec_label": args.clean_ec_label,
        "clean_head_path": args.clean_head_path,
        "clean_embedding_path": args.clean_embedding_path,
        "clean_labels_path": args.clean_labels_path,
        "esm_model_id": args.esm_model_id,
        "boltz_ligand_smiles": args.ligand_smiles,
        "xla_cuda_dir": args.xla_cuda_dir,
    }
    reward_func_base = compose_reward(reward_opts)

    # Wrap Boltz2 terms with a simple per-call multi-GPU pool if >1 device
    def reward_func(prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        if (len(devices) <= 1) or ("boltz2_plddt" not in weights and "boltz2_iptm" not in weights):
            return reward_func_base(prompts, completions)
        # Evaluate only the Boltz part in parallel, then add non-Boltz part from base with zero Boltz weights
        no_boltz_weights = {k: v for k, v in weights.items() if k not in ("boltz2_plddt", "boltz2_iptm")}
        from mosaic_rl.rewards import build_boltz2_scorer
        shard_scores: List[Dict[str, Any]] = []
        jobs: List[Tuple[str, Dict[str, float], str | None, str | None, str | None]] = []
        for seq in completions:
            jobs.append((seq, weights, args.ligand_smiles, os.environ.get("BOLTZ_CACHE"), args.xla_cuda_dir))
        outs: List[Dict[str, Any]] = []
        # shard to devices
        shards: List[List[Tuple[str, Dict[str, float], str | None, str | None, str | None]]] = [[] for _ in devices]
        for i, job in enumerate(jobs):
            shards[i % len(devices)].append(job)
        with ctx.Pool(processes=len(devices)) as pool:
            mapped: List[List[Dict[str, Any]]] = []
            for dev, shard in zip(devices, shards):
                if not shard:
                    mapped.append([])
                    continue
                mapped.append(pool.map(_worker_score_one, shard))
        for lst in mapped:
            outs.extend(lst)
        base_no_boltz = compose_reward({**reward_opts, "weights": no_boltz_weights})
        base_vals = base_no_boltz(prompts, completions)
        results: List[float] = []
        for i, (seq, b) in enumerate(zip(completions, outs)):
            total = float(base_vals[i])
            if "boltz2_plddt" in weights:
                total += weights["boltz2_plddt"] * float(b.get("boltz2_plddt", 0.0))
            if "boltz2_iptm" in weights:
                total += weights["boltz2_iptm"] * float(b.get("boltz2_iptm", 0.0))
            results.append(total)
        return results

    args_trl = _GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        logging_steps=50,
        num_train_epochs=1,
        learning_rate=2e-5,
        save_strategy="no",
        per_device_train_batch_size=1,
        max_prompt_length=64,
        max_completion_length=int(args.max_new_tokens),
        num_generations=int(args.designs),
        generation_batch_size=int(args.designs),
        report_to=[],
    )

    trainer = _GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args_trl,
        reward_funcs=[reward_func],
        train_dataset=dataset,
    )

    # Loop iterations
    for it in range(int(args.iterations)):
        trainer.train()
        # Simple artifact to mark progress
        (out_dir / f"iteration_{it}.done").write_text("ok")

    print({"results_dir": str(out_dir)})


if __name__ == "__main__":
    main()


