import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, Dict, List
from loguru import logger
from mosaic_workflows.analyzers import flatten_aux


def _default_schedule(global_step: int, phase_step: int) -> dict:
    return {}


def _apply_analyzers(analyzers, aux_or_ctx: dict) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for fn in analyzers or []:
        m = fn(aux_or_ctx) or {}
        if isinstance(m, dict):
            metrics |= m
    return metrics
    


def _run_phase(*, phase: dict, x: np.ndarray, key, global_step: int, callbacks, trajectory_path: str | None):
    name = phase["name"]
    build_loss = phase["build_loss"]
    optimizer = phase["optimizer"]
    steps = int(phase["steps"])
    schedule = phase.get("schedule") or _default_schedule
    transforms = phase.get("transforms") or {}
    analyzers = phase.get("analyzers") or []
    analyze_every = int(phase.get("analyze_every", 0))

    # Cache built loss per phase to avoid repeated JIT compiles
    if "_built_loss_cached" in phase:
        loss_built = phase["_built_loss_cached"]
    else:
        loss_built = build_loss()
        phase["_built_loss_cached"] = loss_built
    # Support binder_games two-player minmax losses by dispatching to two-player optimizers when present.
    if isinstance(loss_built, dict) and "two_player" in loss_built:
        two_player_loss = loss_built["two_player"]
        # Wrap two-player loss to match optimizer expectation directly
        def loss_function_two_player(x_probs, y_probs, key=None):
            return two_player_loss(x_probs, y_probs, key)
        # prefer provided optimizer (should be a two-player optimizer)
        loss_function = loss_function_two_player
    else:
        loss_function = loss_built

    trajectory: List[Dict[str, Any]] = []

    def trajectory_fn(aux, x_arr):
        nonlocal trajectory
        rec = {"step": len(trajectory), "aux": aux, "x": x_arr}
        # Surface metrics provided by optimizer
        aux_metrics = aux.get("metrics", {}) if isinstance(aux, dict) else {}
        if isinstance(aux_metrics, dict) and aux_metrics:
            rec["metrics"] = dict(aux_metrics)
        # Optionally append analyzer-derived metrics at configured cadence
        # Always run analyzers for side-effects (e.g., logging)
        aux_for_analyzers = dict(aux) if isinstance(aux, dict) else {"loss": aux}
        aux_for_analyzers["step"] = len(trajectory)
        analyzer_metrics_all = _apply_analyzers(analyzers, aux_for_analyzers)
        if analyze_every and (len(trajectory) % analyze_every == 0):
            if analyzer_metrics_all:
                rec.setdefault("metrics", {}).update(analyzer_metrics_all)
        trajectory.append(rec)
        # Per-iteration logging and streaming JSONL write
        step_idx = rec["step"]
        if trajectory_path:
            os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
            with open(trajectory_path, "a") as f:
                f.write(json.dumps({"step": step_idx, "aux": aux}, default=lambda o: float(o) if hasattr(o, "item") else None) + "\n")
        return rec

    # Sanitize input logits to prevent NaN propagation across phase boundaries
    x_to_use = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if getattr(x_to_use, "ndim", 0) == 2:
        denom = x_to_use.sum(axis=-1, keepdims=True)
        x_to_use = jnp.where(denom > 0, x_to_use / denom, x_to_use)

    x, best_x, _ = optimizer(
        loss_function=loss_function,
        x=x_to_use,
        n_steps=steps,
        key=key,
        schedule=schedule,
        transforms=transforms,
        trajectory_fn=trajectory_fn,
        aux_context={"phase_name": name, "global_step": global_step},
    )

    # callbacks at end of phase
    for cb in callbacks or []:
        cb({"event": "end_phase", "phase": name, "trajectory": trajectory})

    return x, best_x, trajectory


def _decode_best_sequence(best_x) -> str:
    """Decode best_x (logits or probs) to an amino-acid string.

    Accepts numpy arrays or JAX arrays of shape [L, 20]; returns empty string otherwise.
    """
    arr = np.asarray(best_x)
    if getattr(arr, "ndim", 0) == 2 and arr.shape[1] == 20:
        vocab = "ARNDCQEGHILKMFPSTWYV"
        idx = np.argmax(arr, axis=-1)
        return "".join(vocab[int(i)] for i in idx)
    return ""


def run_workflow(workflow: dict) -> dict:
    phases = workflow["phases"]
    binder_len = int(workflow["binder_len"])
    seed = int(workflow.get("seed", 0))
    x0 = workflow.get("initial_x")
    callbacks = workflow.get("callbacks") or []
    trajectory_path = workflow.get("trajectory_path")
    # Configure env defaults to reduce profiler noise and enable latency hider
    os.environ.setdefault("JAX_ENABLE_PGLE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_latency_hiding_scheduler" not in xla_flags:
        xla_flags = (xla_flags + " --xla_gpu_enable_latency_hiding_scheduler=true").strip()
        os.environ["XLA_FLAGS"] = xla_flags

    # Configure loguru baseline
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    if x0 is None:
        x0 = np.random.randn(binder_len, 20).astype(np.float32) * 0.1

    key = jax.random.key(seed)
    x = x0
    best_x = x0
    global_step = 0
    all_traj = []

    for phase in phases:
        x, best_x, traj = _run_phase(
            phase=phase,
            x=x,
            key=jax.random.fold_in(key, global_step),
            global_step=global_step,
            callbacks=callbacks,
            trajectory_path=trajectory_path,
        )
        global_step += phase["steps"]
        all_traj.extend(traj if isinstance(traj, list) else [])

    best_sequence = _decode_best_sequence(best_x)
    return {
        "x": x,
        "best_x": best_x,
        "trajectory": all_traj,
        "best_sequence": best_sequence,
    }

