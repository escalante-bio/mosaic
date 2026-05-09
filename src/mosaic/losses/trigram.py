# N-gram natural-frequency losses against UniRef50 marginals.
#
# - `TrigramLL`     : expected log-likelihood of a soft sequence under a trigram model.
#                     Inspired by https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1
#                     and tends to suppress homopolymer stretches that ESM2 pseudo-likelihood
#                     can favor.
# - `UnigramExcess` : penalizes ONLY over-representation of single amino acids
#                     (relu(emp1 - nat1)^2). Sequence-level — composes with structure
#                     losses via LinearCombination.
# - `BigramExcess`  : same as Unigram for adjacent-pair frequencies.
#
# All three load from the same `mosaic/data/trigram_seg.pkl` UniRef50 trigram pickle.

from __future__ import annotations

import importlib.resources
import pickle
from pathlib import Path

import jax
import numpy as onp
from jax import numpy as jnp, vmap
from jaxtyping import Array, Float

from ..common import LossTerm, TOKENS


def _default_trigram_pkl_path() -> Path:
    return Path(str(importlib.resources.files("mosaic.data") / "trigram_seg.pkl"))


def load_trigram_frequencies(path: Path | str | None = None) -> onp.ndarray:
    """Load the raw 20×20×20 trigram count tensor from the bundled pickle.

    Default path is `mosaic/data/trigram_seg.pkl`. Counts are unnormalized —
    used by `TrigramLL.from_pkl` which clips then row-normalizes to get
    conditional probabilities.
    """
    if path is None:
        path = _default_trigram_pkl_path()
    with open(path, "rb") as f:
        ngram_dict = pickle.load(f)

    n = len(TOKENS)
    P = onp.zeros((n, n, n))
    for trimer, freq in ngram_dict.items():
        if all(c in TOKENS for c in trimer):
            i, j, k = (TOKENS.index(c) for c in trimer)
            P[i, j, k] = freq
    return P


def load_natural_marginals(
    path: Path | str | None = None,
) -> tuple[
    Float[Array, "20"],
    Float[Array, "20 20"],
    Float[Array, "20 20 20"],
]:
    """Load (P1, P2, P3) UniRef50 marginals from the bundled trigram pickle.

    Returns the joint trigram (normalized to sum to 1) plus its bigram and
    unigram marginals. Used by `UnigramExcess` / `BigramExcess`.
    """
    P = load_trigram_frequencies(path)
    P3 = P / P.sum()
    return (
        jnp.asarray(P3.sum(axis=(-1, -2)), dtype=jnp.float32),
        jnp.asarray(P3.sum(axis=-1), dtype=jnp.float32),
        jnp.asarray(P3, dtype=jnp.float32),
    )


class TrigramLL(LossTerm):
    log_probabilities: Float[Array, "20 20 20"]
    stop_grad: bool = False

    def __call__(self, soft_sequence: Float[Array, "N 20"], *, key):
        # Expected log likelihood of the soft sequence under the trigram model
        # if each position is independent: if s_i ~ Categorical(soft_sequence[i]),
        # this equals E_s[\sum_i log p(s_i | s_{i-1}, s_{i-2})].
        def eval_single_position(i: int):
            x_i = soft_sequence[i]
            x_j = soft_sequence[i + 1]
            x_k = soft_sequence[i + 2]
            if self.stop_grad:
                x_i = jax.lax.stop_gradient(x_i)
                x_j = jax.lax.stop_gradient(x_j)

            return jnp.einsum(
                "i,j,k,ijk->",
                x_i,
                x_j,
                x_k,
                self.log_probabilities,
            )

        ave_log_prob = vmap(eval_single_position)(
            jnp.arange(soft_sequence.shape[0] - 2)
        ).mean()

        return -ave_log_prob, {"trigram_ll": ave_log_prob}

    @staticmethod
    def from_pkl(path: Path | str | None = None, stop_grad: bool = False):
        frequencies = onp.clip(load_trigram_frequencies(path), 1e-5, 1.0)
        cond = frequencies / frequencies.sum(-1, keepdims=True)
        return TrigramLL(log_probabilities=onp.log(cond), stop_grad=stop_grad)


class UnigramExcess(LossTerm):
    """Σ relu(emp1[a] − nat1[a])² over the sequence's amino-acid frequencies.

    Penalizes ONLY over-representation. Use this to discourage homopolymer
    collapse without forcing the empirical distribution to match the natural
    marginal.
    """

    nat: Float[Array, "20"]
    name: str = "unigram_excess"

    def __call__(self, soft_sequence: Float[Array, "N 20"], *, key=None):
        emp = soft_sequence.mean(0)
        v = (jax.nn.relu(emp - self.nat) ** 2).sum()
        return v, {"unigram_excess": v}

    @staticmethod
    def from_pkl(path: Path | str | None = None) -> "UnigramExcess":
        nat1, _, _ = load_natural_marginals(path)
        return UnigramExcess(nat=nat1)


class BigramExcess(LossTerm):
    """Σ relu(emp2[a,b] − nat2[a,b])² over adjacent-pair frequencies."""

    nat: Float[Array, "20 20"]
    name: str = "bigram_excess"

    def __call__(self, soft_sequence: Float[Array, "N 20"], *, key=None):
        N = soft_sequence.shape[0]
        emp = jnp.einsum("ia,ib->ab", soft_sequence[:-1], soft_sequence[1:]) / (N - 1)
        v = (jax.nn.relu(emp - self.nat) ** 2).sum()
        return v, {"bigram_excess": v}

    @staticmethod
    def from_pkl(path: Path | str | None = None) -> "BigramExcess":
        _, nat2, _ = load_natural_marginals(path)
        return BigramExcess(nat=nat2)
