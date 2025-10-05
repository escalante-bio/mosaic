from typing import Any, Dict
from loguru import logger
import json
import os


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


def log_inline(aux: dict) -> Dict[str, Any]:
    """Analyzer: emit a single-line, host-only log per step.

    Reads only already-host floats from aux['loss'] and aux['metrics'] (no deep scans),
    and prints: "<step> loss <val> <k1> <v1> ...".
    """
    step = int(aux.get("step", 0)) if isinstance(aux, dict) else 0
    parts: Dict[str, float] = {}
    if isinstance(aux, dict):
        v = aux.get("loss")
        if isinstance(v, (int, float)):
            parts["loss"] = float(v)
        mets = aux.get("metrics")
        if isinstance(mets, dict):
            for k, m in mets.items():
                if isinstance(m, (int, float)):
                    parts[str(k)] = float(m)
    ordered = ["loss"] + sorted([k for k in parts.keys() if k != "loss"])
    line = f"{step}"
    for k in ordered:
        if k in parts:
            line += f" {k} {parts[k]:.3f}"
    if len(line) > 0:
        logger.info(line)
    return {}


def jsonl_stream(aux: dict) -> Dict[str, Any]:
    """Analyzer: append a compact JSONL record at configured cadence.

    Expects in aux: 'trajectory_path' (str), 'step' (int), 'phase' (str),
    optional 'analyze_every' (int). Writes only host floats from loss/metrics.
    """
    if not isinstance(aux, dict):
        return {}
    path = aux.get("trajectory_path")
    if not isinstance(path, str) or len(path) == 0:
        return {}
    step = int(aux.get("step", 0))
    ae = aux.get("analyze_every")
    cadence = int(ae) if isinstance(ae, int) else 1
    if cadence > 1 and (step % cadence) != 0:
        return {}
    rec: Dict[str, Any] = {
        "step": step,
        "phase": aux.get("phase"),
    }
    v = aux.get("loss")
    if isinstance(v, (int, float)):
        rec["loss"] = float(v)
    mets = aux.get("metrics")
    if isinstance(mets, dict):
        rec["metrics"] = {k: float(m) for k, m in mets.items() if isinstance(m, (int, float))}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
    return {}

