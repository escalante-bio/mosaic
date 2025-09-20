"""Regularisation helpers usable from RL optimisers.

Regularisers are registered under a string name and receive the current logits,
reference logits (if required), a weight, and a schedule dictionary. They return
``(penalty, grad, diagnostics)`` where ``grad`` matches the shape of the logits
and ``diagnostics`` is a flat dict merged into the trajectory aux metrics.
"""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp

RegulariserFn = Callable[[jnp.ndarray, jnp.ndarray | None, float, dict], tuple[jnp.ndarray, jnp.ndarray, dict]]

_REGISTRY: Dict[str, RegulariserFn] = {}


def register(name: str, fn: RegulariserFn) -> None:
    if name in _REGISTRY:
        raise ValueError(f"regulariser '{name}' already registered")
    _REGISTRY[name] = fn


def get(name: str) -> RegulariserFn:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown regulariser '{name}'") from exc


def names() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())


_EPS = 1e-8


def _reverse_kl(
    logits: jnp.ndarray,
    reference_logits: jnp.ndarray | None,
    weight: float,
    schedule: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    if reference_logits is None:
        raise ValueError("reverse KL regulariser requires reference_logits in aux_context")

    def penalty_fn(z: jnp.ndarray) -> jnp.ndarray:
        p = jax.nn.softmax(z, axis=-1)
        q = jax.nn.softmax(reference_logits, axis=-1)
        return weight * jnp.sum(p * (jnp.log(p + _EPS) - jnp.log(q + _EPS)))

    penalty, grad = jax.value_and_grad(penalty_fn)(logits)
    diagnostics = {"regularizer/reverse_kl": float(penalty)}
    return penalty, grad, diagnostics


def _forward_kl(
    logits: jnp.ndarray,
    reference_logits: jnp.ndarray | None,
    weight: float,
    schedule: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    if reference_logits is None:
        raise ValueError("forward KL regulariser requires reference_logits in aux_context")

    def penalty_fn(z: jnp.ndarray) -> jnp.ndarray:
        p = jax.nn.softmax(z, axis=-1)
        q = jax.nn.softmax(reference_logits, axis=-1)
        return weight * jnp.sum(q * (jnp.log(q + _EPS) - jnp.log(p + _EPS)))

    penalty, grad = jax.value_and_grad(penalty_fn)(logits)
    diagnostics = {"regularizer/forward_kl": float(penalty)}
    return penalty, grad, diagnostics


def _entropy_bonus(
    logits: jnp.ndarray,
    _: jnp.ndarray | None,
    weight: float,
    schedule: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    def penalty_fn(z: jnp.ndarray) -> jnp.ndarray:
        p = jax.nn.softmax(z, axis=-1)
        return -weight * jnp.sum(p * jnp.log(p + _EPS))

    penalty, grad = jax.value_and_grad(penalty_fn)(logits)
    diagnostics = {"regularizer/entropy": float(penalty)}
    return penalty, grad, diagnostics


# Register built-ins on import
register("reverse_kl", _reverse_kl)
register("forward_kl", _forward_kl)
register("entropy", _entropy_bonus)

