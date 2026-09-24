"""Dedicated ESMFold2 binder design with separately normalized logit gradients.

Implements the gradient update and temperature schedule of Algorithm 11 in
https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1. Model features and
the structural/PLL objectives are supplied by the caller; see
``examples/esmfold_minibinder.py`` for a complete setup.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from mosaic.optimizers import _print_iter, _ranking_leaf

if TYPE_CHECKING:
    from esmjfold2.esmc import ESMCForMaskedLM

    from mosaic.losses.esmc import ESMCPseudoPerplexity
    from mosaic.losses.esmfold2 import ESMFold2Loss


def _normalize_gradient(
    gradient: Float[Array, "N 20"], mask: Float[Array, "N 20"]
) -> Float[Array, "N 20"]:
    """Biohub's normalized_gradient_tensor, for one design in logit space."""
    gradient = gradient * mask
    # Match the reference implementation's effective mutable length, including
    # its treatment of positions with exactly zero gradient.
    effective_length = (jnp.square(gradient).sum(-1) > 0).sum()
    norm = jnp.linalg.norm(gradient)
    return gradient / (norm + 1e-7) * jnp.sqrt(effective_length)


class BiohubObjective(eqx.Module):
    """Store ESMFold2 replicas and PLL with one shared ESMC model.

    This is a design-specific evaluator, not a Mosaic ``LossTerm``. Pass raw
    ``ESMFold2Loss`` instances with ``esmc=None`` and a PLL term with ``esm=None``.
    Do not wrap the branches in ``NormedGradient``: normalization happens here,
    after differentiation through softmax, separately for each design/branch.

    The shared ESMC is inserted into the selected structure replica and PLL
    inside the compiled evaluation, so it appears only once in the input tree.
    """

    stacked: ESMFold2Loss
    static: ESMFold2Loss
    n: int = eqx.field(static=True)
    esmc: ESMCForMaskedLM
    pll: ESMCPseudoPerplexity
    pll_weight: Float[Array, ""]

    def __init__(
        self,
        structure_losses: Sequence[ESMFold2Loss],
        esmc: ESMCForMaskedLM,
        pll: ESMCPseudoPerplexity,
        pll_weight: float = 0.15,
    ) -> None:
        if not structure_losses:
            raise ValueError("BiohubObjective needs at least one structure loss")
        if any(loss.esmc is not None for loss in structure_losses):
            raise ValueError("Structure losses must have esmc=None to share ESMC")
        if pll.esm is not None:
            raise ValueError("PLL must have esm=None to share ESMC")
        if not np.isfinite(pll_weight) or pll_weight < 0:
            raise ValueError("pll_weight must be finite and nonnegative")

        dynamic, static = zip(
            *(eqx.partition(loss, eqx.is_array) for loss in structure_losses)
        )
        if any(not eqx.tree_equal(static[0], other) for other in static[1:]):
            raise ValueError("Structure replicas must have identical static structure")
        self.stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *dynamic)
        self.static = static[0]
        self.n = len(structure_losses)
        self.esmc = esmc
        self.pll = pll
        self.pll_weight = jnp.asarray(pll_weight, dtype=jnp.float32)

    @eqx.filter_jit
    def value_and_grad(
        self,
        logits: Float[Array, "B N 20"],
        *,
        temperature: Float[Array, ""],
        gradient_mask: Float[Array, "N 20"],
        keys: Array,
    ) -> tuple[Float[Array, " B"], PyTree[Array], Float[Array, "B N 20"]]:
        """Return batched values, aux, and ready-to-use logit gradients.

        ``temperature`` is a scalar array, keeping annealing dynamic under JIT.
        ``gradient_mask`` is binary and shared across designs. Model parameters
        are unmapped; random keys, activations, and normalization are per design.
        """

        pll = eqx.tree_at(
            lambda p: p.esm,
            self.pll,
            self.esmc,
            is_leaf=lambda x: x is None,
        )

        def single(z: Float[Array, "N 20"], key: Array):
            k_pick, k_struct, k_pll = jax.random.split(key, 3)
            index = jax.random.randint(k_pick, (), 0, self.n)
            model = eqx.combine(
                jax.tree.map(lambda x: x[index], self.stacked), self.static
            )
            model = eqx.tree_at(
                lambda m: m.esmc,
                model,
                self.esmc.esmc,
                is_leaf=lambda x: x is None,
            )

            def structure_loss(x: Float[Array, "N 20"]):
                return model(jax.nn.softmax(x / temperature, axis=-1), key=k_struct)

            def sequence_loss(x: Float[Array, "N 20"]):
                return pll(jax.nn.softmax(x / temperature, axis=-1), key=k_pll)

            (v_s, aux_s), g_s = jax.value_and_grad(structure_loss, has_aux=True)(z)
            (v_p, aux_p), g_p = jax.value_and_grad(sequence_loss, has_aux=True)(z)
            gradient = _normalize_gradient(g_s, gradient_mask) + (
                self.pll_weight * _normalize_gradient(g_p, gradient_mask)
            )
            return (
                v_s + self.pll_weight * v_p,
                {
                    "model_index": index,
                    "structure": aux_s,
                    "pll": aux_p,
                },
                gradient,
            )

        return jax.vmap(single)(logits, keys)


def biohub_design(
    *,
    objective: BiohubObjective,
    logits: Float[Array | np.ndarray, "B N 20"],
    key: Array,
    gradient_mask: Float[Array | np.ndarray, "N 20"] | None = None,
    n_steps: int = 150,
    learning_rate: float = 0.1,
    temperature_min: float = 0.01,
    confidence_temperature: float = 0.05,
    tail_objective: BiohubObjective | None = None,
    ranking_aux_name: str | None = "ranking_loss",
    verbose: bool = False,
) -> tuple[Float[np.ndarray, "B N 20"], Float[np.ndarray, " B"]]:
    """Design a batch with Biohub's separately normalized structural/PLL SGD.

    Initialize logits before calling (e.g. 0.01 * normal noise for mutable
    positions and -1e6 for forbidden cysteine entries). ``gradient_mask`` freezes
    logit entries at their supplied values; it does not mask softmax itself.
    To fix a residue, freeze its row and initialize all other entries to -1e6.
    Also restrict the PLL term's ``design_idx`` to the mutable positions.

    Below ``confidence_temperature``, use ``tail_objective`` if supplied and
    retain each design's best evaluated iterate. Its ``ranking_aux_name`` metric
    is minimized, so report negative ipTM as ``ranking_loss``. Set the name to
    None to rank by the weighted objective value instead. The tail objective
    should use the same structural/PLL losses plus a confidence monitor.

    Returns ``(best_probabilities, best_scores)`` sorted by ascending score.
    Decode sequences with ``best_probabilities.argmax(-1)``. This routine uses
    plain SGD, with no normalization of the combined gradient or momentum.
    """
    logits = jnp.asarray(logits, dtype=jnp.float32)
    if logits.ndim != 3 or logits.shape[-1] != 20 or min(logits.shape) == 0:
        raise ValueError("logits must have shape [B, N, 20] with B, N > 0")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not 0 < temperature_min < confidence_temperature <= 1:
        raise ValueError("Require 0 < temperature_min < confidence_temperature <= 1")
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and nonnegative")
    if gradient_mask is None:
        gradient_mask = np.ones(logits.shape[1:], dtype=np.float32)
    mask = np.asarray(gradient_mask)
    if mask.shape != logits.shape[1:] or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("gradient_mask must be binary with shape [N, 20]")
    gradient_mask = jnp.asarray(mask, dtype=logits.dtype)
    if tail_objective is None:
        tail_objective = objective

    batch_size = logits.shape[0]
    best_designs = np.zeros(logits.shape, dtype=np.float32)
    best_scores = np.full(batch_size, np.inf)

    for step in range(n_steps):
        t = (step + 1) / n_steps
        temperature = temperature_min + (1 - temperature_min) * (
            0.5 * (1 + np.cos(np.pi * t))
        )
        in_tail = temperature < confidence_temperature
        evaluator = tail_objective if in_tail else objective
        key, step_key = jax.random.split(key)
        values, aux, gradient = evaluator.value_and_grad(
            logits,
            temperature=jnp.asarray(temperature, dtype=logits.dtype),
            gradient_mask=gradient_mask,
            keys=jax.random.split(step_key, batch_size),
        )

        if in_tail:
            scores = (
                values
                if ranking_aux_name is None
                else _ranking_leaf(aux, ranking_aux_name)
            )
            if scores is None:
                raise ValueError(f"Tail objective did not report {ranking_aux_name!r}")
            scores = np.asarray(scores)
            if scores.shape != (batch_size,):
                raise ValueError("Tail ranking must report one scalar per design")
            improved = np.isfinite(scores) & (scores < best_scores)
            probabilities = np.asarray(jax.nn.softmax(logits / temperature, axis=-1))
            best_designs[improved] = probabilities[improved]
            best_scores[improved] = scores[improved]

        # Each branch is already normalized in logit space. Preserve its weight
        # and the resulting step magnitude by applying the sum directly.
        logits = logits - (learning_rate * temperature) * gradient
        if verbose:
            for i in range(batch_size):
                _print_iter(
                    f"{step}[{i}]",
                    {
                        "": jax.tree.map(lambda x, index=i: x[index], aux),
                        "temp": temperature,
                    },
                    values[i],
                )

    if not np.all(np.isfinite(best_scores)):
        raise ValueError("No finite tail ranking score for one or more designs")
    order = np.argsort(best_scores)
    return best_designs[order], best_scores[order]
