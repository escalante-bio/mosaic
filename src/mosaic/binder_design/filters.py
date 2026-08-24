"""Filter evaluation for binder designs.

A faithful port of DdCraft's ``check_filters`` so existing filter JSON configs
keep working unchanged.  Behaviour worth remembering:

* A filter whose ``threshold`` is ``None`` is inactive.
* ``InterfaceAAs`` filters are nested one level deeper, keyed by amino acid.
* A missing value on a per-model metric (``1_``..``5_``) counts as a pass,
  because not every model is necessarily run; a missing value on an averaged
  metric counts as a failure -- and is logged loudly, since an uncomputed metric
  otherwise rejects every design in a way that looks like bad designs.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = ["INTERFACE_AA_LABELS", "check_filters", "unmet_filters"]

INTERFACE_AA_LABELS = frozenset(
    {"Average_InterfaceAAs", *(f"{i}_InterfaceAAs" for i in range(1, 6))}
)

_PER_MODEL_PREFIXES = tuple(f"{i}_" for i in range(1, 6))


def _fails(value: Any, threshold: Any, higher: bool, label: str) -> bool:
    try:
        return value < threshold if higher else value > threshold
    except (TypeError, ValueError) as error:
        logger.warning("Filter comparison error for %s: %s", label, error)
        return True


def unmet_filters(
    values: Mapping[str, Any], filters: Mapping[str, Any]
) -> list[str]:
    """Return the labels of every filter ``values`` fails."""
    unmet: list[str] = []

    for label, conditions in filters.items():
        if label in INTERFACE_AA_LABELS:
            counts = values.get(label)
            if counts is None:
                continue
            for aa, aa_conditions in conditions.items():
                threshold = aa_conditions["threshold"]
                value = counts.get(aa)
                if value is None or threshold is None:
                    continue
                if _fails(value, threshold, aa_conditions["higher"], f"{label}_{aa}"):
                    unmet.append(f"{label}_{aa}")
            continue

        threshold = conditions["threshold"]
        if threshold is None:
            continue

        value = values.get(label)
        if value is None:
            if label.startswith(_PER_MODEL_PREFIXES):
                continue
            logger.warning(
                "Filter '%s' has threshold %s but the value is None (metric not "
                "computed) - rejecting design. If this repeats for every design, "
                "the metric is missing rather than the designs being bad.",
                label,
                threshold,
            )
            unmet.append(label)
            continue

        if _fails(value, threshold, conditions["higher"], label):
            unmet.append(label)

    return unmet


def check_filters(
    design_values: Iterable[Any] | Mapping[str, Any],
    design_labels: Iterable[str] | None = None,
    filters: Mapping[str, Any] | None = None,
) -> bool | list[str]:
    """DdCraft-compatible entry point.

    Accepts either a mapping of label to value, or a parallel
    ``(values, labels)`` pair.  Returns ``True`` when every filter passes, or
    the list of unmet filter labels otherwise.
    """
    if filters is None:
        raise TypeError("filters is required")

    if isinstance(design_values, Mapping):
        values = dict(design_values)
    else:
        if design_labels is None:
            raise TypeError("design_labels is required when passing a value sequence")
        values = dict(zip(design_labels, design_values))

    unmet = unmet_filters(values, filters)
    return True if not unmet else unmet
