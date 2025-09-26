import os
import sys
from pathlib import Path

import modal


# --- Modal image: minimal deps for Tiny-CLEAN RL ---
image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git")
    .env({
        "HF_DATASETS_CACHE": "/mosaic/hf",
        "TRANSFORMERS_CACHE": "/mosaic/hf",
        "HF_HOME": "/mosaic/hf",
        "TORCH_HOME": "/mosaic/torch",
        "JAX_PLATFORMS": "cuda",
        "BOLTZ_CACHE": "/root/.boltz",
        # Point directly to NVCC-provided CUDA dir for libdevice
        "XLA_FLAGS": "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_cuda_data_dir=/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
    })
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        # Torch (CUDA wheels) - use cu124 channel to obtain >=2.6
        "python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch --upgrade && "
        # Ensure cuDNN matches jaxlib build by overriding PyTorch's pinned cudnn
        "python -m pip uninstall -y nvidia-cudnn-cu12 || true && "
        "python -m pip install nvidia-cudnn-cu12==9.8.0.87 --no-deps && "
        # Uninstall mismatched JAX stack and pin to boltz-compatible versions
        "python -m pip uninstall -y jax jaxlib jax-cuda12-plugin || true && "
        "python -m pip install --no-cache-dir jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        # NumPy 1.26.x for numba/boltz
        "python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4 numba==0.61.0 && "
        # Transformers/TRL/Datasets/Accelerate
        "python -m pip install transformers datasets trl accelerate && "
        # Equinox for Joltz
        "python -m pip install equinox && "
        # CUDA NVCC (provides libdevice required by XLA)
        "python -m pip install nvidia-cuda-nvcc-cu12 && "
        # CUDA runtime libs for JAX plugin (ensure availability)
        "python -m pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 && "
        # ESM for embeddings, Gemmi for PDB parsing
        "python -m pip install fair-esm gemmi && "
        # Joltz + Boltz (from git)
        "python -m pip install --no-cache-dir git+https://github.com/adaptyvbio/joltz.git && "
        "python -m pip install --no-cache-dir git+https://github.com/jwohlwend/boltz.git && "
        # Plotting
        "python -m pip install matplotlib seaborn && "
        # FINAL: enforce cuDNN version compatible with jaxlib
        "python -m pip uninstall -y nvidia-cudnn-cu12 || true && "
        "python -m pip install nvidia-cudnn-cu12==9.8.0.87 --no-deps"
    )
    # Add just our source tree (small) so we can import mosaic_rl and workflows
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")
    .add_local_file(
        "/Users/tudorcotet/Downloads/6QZ4.pdb",
        "/seed/6QZ4.pdb",
    )
    # Optional: seed a local CSV for dataset-based ProtRL runs (user-provided path)
    .add_local_file(
        "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/_external/selected_designs_getbest_Chai1.csv",
        "/seed/protrl_dataset.csv",
    )
)


app = modal.App("mosaic-rl-tiny-clean", image=image)

# Single persisted volume for all Mosaic RL outputs and caches
mosaic_vol = modal.Volume.from_name("mosaic-rl", create_if_missing=True)
boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)


def _add_paths() -> None:
    # Prefer mounted local source (latest edits)
    local_src = Path("/workspace/src")
    if local_src.exists():
        sys.path.insert(0, str(local_src))


# (Removed legacy Tiny-CLEAN helpers and function)


@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/mosaic": mosaic_vol},
)
def run_protrl(
    *,
    ec_label: str = "3.1.1.102",
    iterations: int = 3,
    designs: int = 20,
    model_id: str = "AI4PD/ZymCTRL",
) -> str:
    _add_paths()

    from mosaic_rl.experiments.protrl import ProtRLConfig, run_pipeline

    # Build run directory
    import time as _time, uuid as _uuid, json as _json
    cache_root = Path("/mosaic")
    run_id = f"protrl_{int(_time.time())}_{str(_uuid.uuid4())[:8]}"
    run_dir = cache_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    workspace = run_dir / "workspace"
    results_dir = run_dir / "checkpoints"

    cfg = ProtRLConfig(
        workspace=workspace,
        label=ec_label,
        model=model_id,
        reference_model=model_id,
        results_dir=results_dir,
        tokenizer_id=model_id,
        max_new_tokens=1024,
        top_k=9,
        num_generations=int(designs),
        schedule=(2e-5, 1e-5, 5e-6),
        device="cuda",
    )

    # Save metadata/config
    meta = {
        "run_id": run_id,
        "created_at": int(_time.time()),
        "ec_label": ec_label,
        "iterations": int(iterations),
        "designs_per_iteration": int(designs),
        "workspace": str(workspace),
        "results_dir": str(results_dir),
        "model": model_id,
        "tokenizer": model_id,
        "schedule": list(cfg.schedule),
        "device": cfg.device,
    }
    (run_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))

    # Ensure dirs and run
    cfg.inputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(cfg, iterations=int(iterations))

    return str(run_dir)


@app.function(
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={"/mosaic": mosaic_vol, "/root/.boltz": boltz_cache},
)
def run_protrl_csv(
    *,
    ec_label: str = "3.1.1.102",
    model_id: str = "AI4PD/ZymCTRL",
    reward: str = "total_score=1.0,efield_score=1.0,ncaa_interface_score=1.0,boltz2_plddt=1.0,boltz2_iptm=1.0,motif_pos=1.0,motif_rmsd=-1.0,length100=-1.0",
    eos_token: str = "<|endoftext|>",
    iterations: int = 5,
    designs: int = 20,
    max_new_tokens: int = 600,
    ligand_smiles: str | None = "[C@H](O)(c1ccc(C(=O)O)cc1)Oc1ccc(cc1)N(=O)=O",
    cat_positions: str = "10,20,30",
    cat_identities: str = "S,D,H",
    motif_pdb_path: str | None = "/seed/6QZ4.pdb",
    motif_chain_id: str | None = "A",
    motif_resnums: str | None = "225,492,528",
) -> str:
    _add_paths()
    # Use GPU for Boltz2 scoring (JAX CUDA stack pinned in image)
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_DISABLE_JAX_PLUGIN_DISCOVERY"] = "0"
    os.environ["WANDB_DISABLED"] = "true"
    # Ensure Boltz cache directory exists
    os.environ.setdefault("BOLTZ_CACHE", "/root/.boltz")
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)

    import time as _time, uuid as _uuid, json as _json
    from datasets import Dataset as _Dataset
    from transformers import AutoTokenizer as _AutoTokenizer
    from trl import GRPOConfig as _GRPOConfig, GRPOTrainer as _GRPOTrainer

    cache_root = Path("/mosaic")
    run_id = f"protrl_csv_{int(_time.time())}_{str(_uuid.uuid4())[:8]}"
    run_dir = cache_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy CSV from seed into run folder
    csv_src = Path("/seed/protrl_dataset.csv")
    csv_dst = run_dir / "dataset.csv"
    if not csv_src.exists():
        raise FileNotFoundError("Seed CSV not found at /seed/protrl_dataset.csv")
    csv_dst.write_bytes(csv_src.read_bytes())

    # Helper: parse reward weights from CLI string

    def _parse_weights(spec: str) -> dict[str, float]:
        weights_: dict[str, float] = {}
        for part in spec.split(','):
            if not part.strip():
                continue
            k, v = part.split('=')
            weights_[k.strip()] = float(v.strip())
        return weights_

    weights = _parse_weights(reward)

    # Build prompt-only dataset for GRPO
    dataset = _Dataset.from_list([{"prompt": str(ec_label)} for _ in range(max(1, int(designs)))])

    results_dir = run_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)

    # For generation config only
    from mosaic_rl.experiments.protrl import ProtRLConfig as _ProtRLConfig

    current_checkpoint = model_id
    metrics_log = []

    for it in range(int(iterations)):
        # Tokenizer for generation side-effects and EOS handling
        tok = _AutoTokenizer.from_pretrained(current_checkpoint)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        step_dir = results_dir / f"iter_{it}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Centralized reward via mosaic_rl.rewards.compose_reward (plain dict config)
        from mosaic_rl.rewards import compose_reward as _compose, build_reward_scorers as _build_scorers
        _reward_core = _compose({
            "weights": weights,
            "csv_path": str(csv_dst),
            "use_csv_predictors": True,
            "boltz_ligand_smiles": ligand_smiles,
            "xla_cuda_dir": "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc",
            "eos_token": eos_token,
            "motif_positions": tuple(int(x.strip()) for x in str(cat_positions).split(',') if x.strip()) if cat_positions else None,
            "motif_identities": tuple(s.strip().upper() for s in str(cat_identities).split(',') if s.strip()) if cat_identities else None,
        })
        # Components scorer for logging
        _score_total, _score_components = _build_scorers({
            "weights": weights,
            "csv_path": str(csv_dst),
            "use_csv_predictors": True,
            "boltz_ligand_smiles": ligand_smiles,
            "xla_cuda_dir": "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc",
            "eos_token": eos_token,
            "motif_positions": tuple(int(x.strip()) for x in str(cat_positions).split(',') if x.strip()) if cat_positions else None,
            "motif_identities": tuple(s.strip().upper() for s in str(cat_identities).split(',') if s.strip()) if cat_identities else None,
        })

        # Capture per-iteration sequence rows from the same sequences used for reward
        rows_iter: list[dict] = []

        def _reward_fn(prompts, completions, **_):
            seqs = ["".join(str(c).split()) for c in completions]
            totals = _reward_core(prompts, seqs)
            comps = _score_components(seqs)
            # record rows for logging
            for s, t, comp in zip(seqs, totals, comps):
                row = {"iteration": int(it), "sequence": s, "reward": float(t), "length": len(s)}
                row.update({k: float(v) for k, v in comp.items()})
                rows_iter.append(row)
            return totals

        args = _GRPOConfig(
            output_dir=str(step_dir),
            logging_steps=50,
            num_train_epochs=1,
            learning_rate=2e-5,
            save_strategy="no",
            fp16=False,
            bf16=False,
            per_device_train_batch_size=1,
            max_prompt_length=64,
            max_completion_length=int(max_new_tokens),
            num_generations=2,
            generation_batch_size=2,
        )

        trainer = _GRPOTrainer(
            model=current_checkpoint,
            reward_funcs=_reward_fn,
            args=args,
            train_dataset=dataset,
        )

        trainer.train()
        # Save checkpoint for this iteration
        trainer.save_model(str(step_dir))
        tok.save_pretrained(step_dir)

        # Log metrics for this iteration, using rows captured during reward calls
        history = getattr(trainer.state, "log_history", [])
        iter_metrics: dict[str, float] = {"iteration": float(it)}
        for record in history:
            for k, v in record.items():
                if isinstance(v, (int, float)) and (k.startswith("train") or k.startswith("eval")):
                    iter_metrics[k] = float(v)
        # Append to JSONL and write CSV snapshot (rows captured during reward_fn)
        seq_path = run_dir / "sequences.jsonl"
        with open(seq_path, "a") as _fjsonl:
            for rec in rows_iter:
                _fjsonl.write(_json.dumps(rec) + "\n")
        import pandas as _pd
        _pd.DataFrame(rows_iter).to_csv(run_dir / f"iteration_{it}_scores.csv", index=False)
        # Compute per-iteration reward statistics
        if rows_iter:
            import numpy as _np
            vals = _np.asarray([r["reward"] for r in rows_iter], dtype=_np.float32)
            iter_metrics["reward_mean"] = float(vals.mean())
            iter_metrics["reward_std"] = float(vals.std(ddof=0))
        # reset for next iteration
        rows_iter = []
        metrics_log.append(iter_metrics)
        (run_dir / "metrics.jsonl").write_text("\n".join(_json.dumps(m) for m in metrics_log))

        # End iteration loop body

    # Make a simple reward-over-iterations plot with uncertainty
    try:
        import pandas as _pd2
        import matplotlib.pyplot as _plt
        data = _pd2.DataFrame(metrics_log)
        if not data.empty and {"iteration", "reward_mean"}.issubset(data.columns):
            xs = data["iteration"].values
            ys = data["reward_mean"].values
            yerr = data.get("reward_std", _pd2.Series([0.0] * len(xs))).values
            fig, ax = _plt.subplots(figsize=(6, 3))
            ax.plot(xs, ys, marker="o", label="reward_mean")
            ax.fill_between(xs, ys - yerr, ys + yerr, alpha=0.2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Reward")
            fig.tight_layout()
            fig.savefig(run_dir / "iterations_reward.png", dpi=150)
            _plt.close(fig)
    except Exception:
        pass

    meta = {
        "run_id": run_id,
        "created_at": int(_time.time()),
        "ec_label": ec_label,
        "model": model_id,
        "reward_weights": weights,
        "eos_token": eos_token,
        "results_dir": str(results_dir),
        "csv_source": str(csv_src),
        "iterations": int(iterations),
        "designs_per_iteration": int(designs),
    }
    (run_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))
    # Create a final JSON snapshot for convenience
    import time as _t3
    final_payload = {
        "run_id": run_id,
        "created_at": meta["created_at"],
        "completed_at": int(_t3.time()),
        "ec_label": ec_label,
        "model": model_id,
        "iterations": int(iterations),
        "designs_per_iteration": int(designs),
        "results_dir": str(results_dir),
        "csv_source": str(csv_src),
    }
    (run_dir / "final.json").write_text(_json.dumps(final_payload, indent=2))
    return str(run_dir)


@app.function(volumes={"/mosaic": mosaic_vol})
def list_cached_results(base: str = "/mosaic/runs") -> None:
    from pathlib import Path as _Path
    import os as _os
    root = _Path(base)
    if not root.exists():
        print({"exists": False, "path": str(root)})
        return
    out = []
    for dirpath, dirnames, filenames in _os.walk(root):
        rel = str(_Path(dirpath))
        files = sorted(filenames)
        out.append({"dir": rel, "files": files[:10], "num_files": len(files)})
        if len(out) >= 20:
            break
    print({"exists": True, "path": str(root), "sample": out})


@app.local_entrypoint()
def main(
    ec_label: str = "3.1.1.102",
    iterations: int = 5,
    binder_len: int = 50,
    designs: int = 4,
    workflow: str = "run",
    reward: str = "boltz2_plddt=1.0,boltz2_iptm=1.0,motif_rmsd=-1.0",
    max_new_tokens: int = 600,
    ligand_smiles: str | None = "[C@H](O)(c1ccc(C(=O)O)cc1)Oc1ccc(cc1)N(=O)=O",
    cat_positions: str = "10,20,30",
    cat_identities: str = "S,D,H",
    motif_pdb_path: str | None = None,
    motif_chain_id: str | None = None,
    motif_resnums: str | None = None,
) -> None:
    if workflow == "list":
        list_cached_results.remote()
    elif workflow == "protrl":
        out = run_protrl.remote(ec_label=ec_label, iterations=int(iterations), designs=int(designs))
        print({"results_dir": out})
    elif workflow == "protrl_csv":
        out = run_protrl_csv.remote(
            ec_label=ec_label,
            reward=reward,
            iterations=int(iterations),
            designs=int(designs),
            max_new_tokens=int(max_new_tokens),
            ligand_smiles=ligand_smiles,
            cat_positions=cat_positions,
            cat_identities=cat_identities,
            motif_pdb_path=motif_pdb_path,
            motif_chain_id=motif_chain_id,
            motif_resnums=motif_resnums,
        )
        print({"results_dir": out})
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


