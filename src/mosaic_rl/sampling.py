"""Sampling utilities shared by RL optimisers."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .utils import AA_VOCAB


def sample_categorical_sequences(
    logits: jnp.ndarray,
    key: jax.random.PRNGKey,
    *,
    num_samples: int,
    temperature: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample sequences from independent categorical distributions.

    Args:
        logits: ``(L, V)`` array of policy logits.
        key: PRNG key used for sampling.
        num_samples: number of trajectories to draw.
        temperature: sampling temperature.

    Returns:
        indices: ``(num_samples, L)`` array of token indices.
        probs: ``(L, V)`` array of current softmax probabilities (shared across samples).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    temperature = max(float(temperature), 1e-6)
    logits = jnp.asarray(logits)
    logits_scaled = logits / temperature
    probs = jax.nn.softmax(logits_scaled, axis=-1)

    def sample_single(rng):
        draws = jax.random.categorical(rng, logits=logits_scaled, axis=-1)
        return draws.astype(jnp.int32)

    keys = jax.random.split(key, num_samples)
    indices = jax.vmap(sample_single)(keys)
    return indices, probs

