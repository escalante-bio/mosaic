from typing import Any, Dict
from loguru import logger


def _flatten(prefix: str, node: Any, out: Dict[str, Any]):
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _flatten(f"{prefix}.{i}" if prefix else str(i), v, out)
    else:
        out[prefix] = node


def flatten_aux(aux: dict) -> Dict[str, Any]:
    """Flatten nested aux/metrics into dotted-key dict for validators and logging."""
    flat: Dict[str, Any] = {}
    _flatten("", aux, flat)
    return flat


def colab_style_log_inline(aux: dict) -> Dict[str, Any]:
    """Analyzer: emit a single-line ColabDesign-style log of losses per step.

    Expects aux to include keys: 'loss' (float), 'losses' (dict of components), and optional 'step' (int).
    Returns an empty metrics dict.
    """
    step = int(aux.get("step", 0)) if isinstance(aux, dict) else 0
    parts: Dict[str, float] = {}
    if isinstance(aux, dict) and ("loss" in aux) and isinstance(aux["loss"], (int, float)):
        parts["loss"] = float(aux["loss"])  # total
    losses = aux.get("losses") if isinstance(aux, dict) else None
    if isinstance(losses, dict):
        flat = flatten_aux(losses)
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                parts[str(k)] = float(v)
    else:
        # Fallback: flatten nested 'metrics' or 'aux' to surface numeric fields
        cand = None
        if isinstance(aux, dict):
            if isinstance(aux.get("metrics"), dict):
                cand = aux.get("metrics")
            elif isinstance(aux.get("aux"), dict):
                cand = aux.get("aux")
        if isinstance(cand, dict):
            flat_any = flatten_aux(cand)
            for k, v in flat_any.items():
                if isinstance(v, (int, float)):
                    parts[str(k)] = float(v)
    ordered = ["loss"] + sorted([k for k in parts.keys() if k != "loss"])
    line = f"{step}"
    for k in ordered:
        if k in parts:
            line += f" {k} {parts[k]:.3f}"
    logger.info(line)
    return {}

