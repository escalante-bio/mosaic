import os
import json
from typing import Any, Dict, List, Tuple, Callable

from .design import run_workflow
from .analyzers import flatten_aux


def _state_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "state.json")


def _load_state(out_dir: str) -> Dict[str, Any]:
    with open(_state_path(out_dir), "r") as f:
        return json.load(f)


def _save_state(out_dir: str, st: Dict[str, Any]) -> None:
    with open(_state_path(out_dir), "w") as f:
        json.dump(st, f)


def _build_row(kind: str, spec: dict, res: dict) -> dict:
    trajectory = res.get("trajectory") or []
    last = trajectory[-1] if trajectory else {}
    aux_last = last.get("aux", {}) if isinstance(last, dict) else {}
    metrics = flatten_aux(aux_last) if isinstance(aux_last, dict) else {}
    row = {
        "kind": kind,
        "spec": spec,
        "best_sequence": res.get("best_sequence"),
        "metrics": metrics,
    }
    return row


def run_many(
    *,
    specs: List[dict],
    build: Callable[[dict], dict],
    spawn: Callable[[dict, dict, dict], List[Tuple[dict, Callable[[dict], dict]]]] | None = None,
    emit: Callable[[str, dict], None] | None = None,
    stop: Callable[[List[dict]], bool] | None = None,
    resume: bool = True,
    out_dir: str = ".",
) -> dict:
    state = {"index": 0}
    if resume and os.path.exists(_state_path(out_dir)):
        state = _load_state(out_dir)
    rows: List[dict] = []
    start_index = int(state.get("index", 0))

    for i in range(start_index, len(specs)):
        spec = specs[i]

        # Parent workflow
        wf_parent = build(spec)
        res_parent = run_workflow(wf_parent)
        row_parent = _build_row("parent", spec, res_parent)
        rows.append(row_parent)
        if emit:
            emit("parent", row_parent)

        if spawn is not None:
            for child_spec, child_build in (spawn(spec, res_parent, row_parent) or []):
                wf_child = child_build(child_spec)
                res_child = run_workflow(wf_child)
                row_child = _build_row("child", child_spec, res_child)
                row_child["parent_spec"] = spec
                rows.append(row_child)
                if emit:
                    emit("child", row_child)

        if stop and stop(rows):
            new_state = {"index": i + 1}
            if resume:
                _save_state(out_dir, new_state)
            return {"rows": rows, "state": new_state, "dir": out_dir}

        new_state = {"index": i + 1}
        if resume:
            _save_state(out_dir, new_state)

    return {"rows": rows, "state": {"index": len(specs)}, "dir": out_dir}


