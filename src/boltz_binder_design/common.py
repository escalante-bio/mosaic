import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

TOKENS = "ARNDCQEGHILKMFPSTWYV"

class LossTerm(eqx.Module):
    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        raise NotImplementedError

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(l=[self], weights=jnp.array([scalar]))

    def __add__(self, other):
        return 1.0 * self + 1.0 * other

    def __neg__(self):
        return (-1.0) * self

    def __sub__(self, other):
        return self + (-1.0) * other

    
class LinearCombination(eqx.Module):
    """Weighted linear combination of loss terms."""

    # losses: list[tuple[float, any]]
    l: list[LossTerm]
    weights: jax.Array

    def __call__(self, *args, key, **kwargs) -> tuple[float, list]:
        r = 0.0
        aux_values = []
        for w, loss in zip(self.weights, self.l):
            v, a = loss(*args, key=key, **kwargs)
            key = jax.random.fold_in(key, 1)
            r += w * v
            aux_values.append(a)
        return r, aux_values

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(
            l=self.l,
            weights=self.weights * scalar,
        )

    def __add__(self, other):
        if isinstance(other, LossTerm):
            other = 1.0 * other  # lift to LinearCombination

        if not isinstance(other, LinearCombination):
            return NotImplemented

        return LinearCombination(
            l=self.l + other.l,
            weights=jnp.concatenate([self.weights, other.weights]),
        )
    
    def __sub__(self, other):
        return self + (-1.0) * other

    def __neg__(self):
        return (-1.0) * self


# This is high weirdness to support "stateful" losses (e.g. to interleave recycling and optimization steps). 
# The right way to do this is to plumb some output of a loss module as a new argument (e.g. a structure prediction loss that returns 
# the trunk state after recycling and takes in an initial trunk state in addition to sequence etc).
# In order to make this all work we need
#   1. a way for a module to indicate that it can be updated
#   2. to get the update from the output of the  _value_and_grad call,
#   3. and finally a way to link the update to the module and call an update method to update the full loss pytree


# The way this is done now is:
#   1. a module has a `state_index` property that is an instance of `StateIndex`
#   2. part of the aux pytree is a tuple of (StateIndex, value) where value is the argument to be passed to the `update_state` method
#   3. the `update_state` method is called for each module in the loss pytree that has a `state_index` property
#   This last step happens in the optimization loop after the value and gradient have been computed.

# As always we should use Patrick's elegant approach from equinox instead but I can't bring myself to pass additional arguments to all loss functions.
class StateIndex(eqx.Module):
    """ A marker that can be returned as part of the "aux" pytree to indicate a loss module is "stateful" and can be updated. 
    Such modules should have a matching `StateIndex` property named `state_index` and a method `update_state` to update the state.
    The aux pytree should contain a tuple of (StateIndex, value) where value is the argument to be passed to the `update_state` method.

    See `is_state_update` and `has_state_index` for how to use this.
    """
    id: int = eqx.field(default_factory=lambda: np.random.randint(0, 2**16 - 1, dtype=jnp.int32))

# This is run on aux output to find state updates
def is_state_update(n):
    return isinstance(n, tuple) and isinstance(n[0], StateIndex)

# This is run on modules to find stateful losses that can accept updates
def has_state_index(m):
    return isinstance(m, eqx.Module) and hasattr(m, "state_index") and isinstance(m.state_index, StateIndex)