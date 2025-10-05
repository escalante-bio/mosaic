 
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, Dict, List
 


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

    # Build loss once per phase invocation without mutating the phase dict
    loss_built = build_loss() if callable(build_loss) else build_loss
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
        rec = {"step": len(trajectory)}
        # Surface metrics provided by optimizer
        aux_metrics = aux.get("metrics", {}) if isinstance(aux, dict) else {}
        if isinstance(aux_metrics, dict) and aux_metrics:
            rec["metrics"] = dict(aux_metrics)
        # Optionally append analyzer-derived metrics at configured cadence
        # Always run analyzers for side-effects (e.g., logging/streaming)
        aux_for_analyzers = dict(aux) if isinstance(aux, dict) else {"loss": aux}
        aux_for_analyzers["step"] = len(trajectory)
        aux_for_analyzers["phase"] = name  # Add phase name for logging
        # Provide streaming path and cadence so analyzers can handle JSONL writes
        aux_for_analyzers["trajectory_path"] = trajectory_path
        aux_for_analyzers["analyze_every"] = analyze_every
        analyzer_metrics_all = _apply_analyzers(analyzers, aux_for_analyzers)
        if analyze_every and (len(trajectory) % analyze_every == 0):
            if analyzer_metrics_all:
                rec.setdefault("metrics", {}).update(analyzer_metrics_all)
        trajectory.append(rec)
        return rec

    # Use x as-is; sanitization/renorm should be handled via explicit transforms
    x_to_use = x

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

    # Environment and logging configuration should live in entrypoints (e.g., modal_app)

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

