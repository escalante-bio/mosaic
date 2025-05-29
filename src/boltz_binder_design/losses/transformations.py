# Simple transformations of loss functions
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
import jax

from ..common import TOKENS, LinearCombination, LossTerm


class ClippedLoss(LossTerm):
    """
    Clips a loss function to a range [l, u].
    Useful for loss functions that might behave badly when over-optimized.
    For example, optimizing raw ESM2 psuedolikelihood often gives homopolymers.

    Properties:
    - loss: LossTerm
    - l: lower bound
    - u: upper bound
    """

    loss: LossTerm
    l: float
    u: float

    def __call__(self, *args, key, **kwargs):
        v, aux = self.loss(*args, key=key, **kwargs)
        return v.clip(self.l, self.u), aux | {f"clipped[{self.loss.__name__}]": v.clip(self.l, self.u)}


# Generic tools for fixing positions in a binder sequence
# Note: if you're finetuning an existing binder you might want to (additionally)
#  - If you're using Boltz: use a binder sequence (instead of all "X"'s) to generate features
#  - If using AF2: set the wildtype complex as the initial guess (maybe, this hasn't been tested)
#  - Add additional loss functions to constrain the design to be close to the wildtype (if you have a complex):
#    - ProteinMPNN inverse folding for the complex
#    - Some kind of distance metric on the predicted complex structure, e.g. DistogramCE
#
class SetPositions(LossTerm):
    """Precomposes loss functional with function that maps a soft sequence of ONLY VARIABLE positions to a full binder sequence to eliminate constraints/penalties.
    WARNING: Be sure to call `sequence` *after* optimization, e.g. `loss.sequence(jax.nn.softmax(logits))`."""

    wildtype: Int[Array, "N"]
    variable_positions: Int[Array, "M"]
    loss: LossTerm | LinearCombination

    def __call__(self, seq: Float[Array, "M 20"], *, key):
        assert seq.shape == (len(self.variable_positions), len(TOKENS))
        return self.loss(self.sequence(seq), key=key)

    def sequence(self, seq: Float[Array, "M 20"]):
        return (
            jax.nn.one_hot(self.wildtype, len(TOKENS))
            .at[self.variable_positions]
            .set(seq)
        )

    @staticmethod
    def from_sequence(wildtype: str, loss: LossTerm | LinearCombination):
        """Fix standard amino acids but allow variability at positions with 'X'"""
        wildtype_tokens = jnp.array([TOKENS.index(AA) for AA in wildtype])
        variable_positions = jnp.array(
            [i for i, AA in enumerate(wildtype) if AA == "X"]
        )
        return SetPositions(wildtype_tokens, variable_positions, loss)


class FixedPositionsPenalty(LossTerm):
    """Penalizes deviation from target at fixed positions using L2^2 loss. Might make optimization more difficult compared to `SetPositions` above, but is simpler"""

    position_mask: Bool[Array, "N"]
    target: Float[Array, "N 20"]

    def __call__(self, seq: Float[Array, "N 20"], *, key):
        r = (((seq - self.target) ** 2).sum(-1) * self.position_mask).sum()
        return r, {"fixed_position_penalty": r}

    @staticmethod
    def from_residues(sequence_length: int, positions_and_AAs: list[tuple[int, str]]):
        position_mask = np.zeros(sequence_length, dtype=bool)
        target = np.zeros((sequence_length, len(TOKENS)))
        for idx, AA in positions_and_AAs:
            position_mask[idx] = True
            target[idx, TOKENS.index(AA)] = 1.0

        return FixedPositionsPenalty(jnp.array(position_mask), jnp.array(target))


@jax.custom_vjp
def clip_gradient(threshold, x):
    return x


def clip_gradient_fwd(threshold, x):
    return x, (threshold,)


def clip_gradient_bwd(T, g):
    (threshold,) = T
    g = g - g.mean(axis=-1, keepdims=True)
    norm = jnp.sqrt((g**2).sum() + 1e-8)
    return (
        None,
        jax.lax.select(
            norm > threshold,
            g * (threshold / norm),
            g,
        ),
    )


clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)


class ClippedGradient(LossTerm):
    loss: LossTerm
    max_norm: float

    def __call__(self, seq, *, key):
        return self.loss(clip_gradient(self.max_norm, seq), key=key)


@jax.custom_vjp
def norm_gradient(x):
    return x


def norm_gradient_fwd(x):
    return x, None


def norm_gradient_bwd(_, g):
    g = g - g.mean(axis=-1, keepdims=True)
    norm = jnp.sqrt((g**2).sum() + 1e-8)
    return (
        g / norm,
    )


norm_gradient.defvjp(norm_gradient_fwd, norm_gradient_bwd)


class NormedGradient(LossTerm):
    loss: LossTerm

    def __call__(self, seq, *, key):
        return self.loss(norm_gradient(seq), key=key)
