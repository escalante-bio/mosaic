import equinox as eqx
import jax
import jax.numpy as jnp

TOKENS = "ARNDCQEGHILKMFPSTWYV"

class LossTerm(eqx.Module):
    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        raise NotImplementedError

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(losses=[self], weights=jnp.array([scalar]))

    def __add__(self, other):
        return 1.0 * self + 1.0 * other

    def __neg__(self):
        return (-1.0) * self


class LinearCombination(eqx.Module):
    """Weighted linear combination of loss terms."""

    # losses: list[tuple[float, any]]
    losses: list[LossTerm]
    weights: jax.Array

    def __call__(self, *args, key, **kwargs) -> tuple[float, dict]:
        r = 0.0
        aux = {}
        for w, loss in zip(self.weights, self.losses):
            v, a = loss(*args, key=key, **kwargs)
            key = jax.random.fold_in(key, 1)
            r += w * v
            aux.update({f"{k}": v for k, v in a.items()})
        return r, aux

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(
            losses=self.losses,
            weights=self.weights * scalar,
        )

    def __add__(self, other):
        if isinstance(other, LossTerm):
            other = 1.0 * other  # lift to LinearCombination

        if not isinstance(other, LinearCombination):
            return NotImplemented

        return LinearCombination(
            losses=self.losses + other.losses,
            weights=jnp.concatenate([self.weights, other.weights]),
        )

    def __neg__(self):
        return (-1.0) * self
