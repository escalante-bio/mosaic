from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from ..outer import run_many


BuildParent = Callable[[dict], dict]
SpawnChildren = Callable[[dict, dict, dict], List[Tuple[dict, Callable[[dict], dict]]]]
EmitRow = Callable[[str, dict], None]


def run_pipeline(
    *,
    specs: List[dict],
    build_parent: BuildParent,
    spawn_children: SpawnChildren | None,
    emit_row: EmitRow | None,
    out_dir: str,
) -> dict:
    return run_many(
        specs=specs,
        build=build_parent,
        spawn=spawn_children,
        emit=emit_row,
        stop=None,
        resume=False,
        out_dir=out_dir,
    )


