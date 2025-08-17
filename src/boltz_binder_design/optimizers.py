import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable
from boltz_binder_design.common import is_state_update, has_state_index


def _print_iter(iter, aux, v):
    # first filter out anything that isn't a float or has number of dimensions > 0
    aux = eqx.filter(
        aux,
        lambda v: isinstance(v, float | str) or v.shape == (),
    )
    print(
        iter,
        f"loss: {v:0.2f}",
        " ".join(
            f"{jax.tree_util.keystr(k, simple=True, separator='.')}:{v: 0.2f}"
            for (k, v) in jax.tree_util.tree_leaves_with_path(aux)
            if hasattr(v, "item") or isinstance(v, float)
        ),
    )


# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
def _eval_loss_and_grad(loss_function, x, key):
    # standardize input to avoid recompilation
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _____eval_loss_and_grad(loss_function, x=x, key=key)
    return (v, aux), g - g.mean(axis=-1, keepdims=True)


# more underscores == more private
@eqx.filter_jit
def _____eval_loss_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


# this function is a mess
def update_states(aux, loss):
    state_index_to_update = dict(
        [
            (int(x[0].id), x[1])
            for x in jax.tree.leaves(aux, is_leaf=is_state_update)
            if is_state_update(x)
        ]
    )
    def get_modules_to_update(loss):
        return tuple([x for x in jax.tree.leaves(loss, is_leaf = has_state_index) if has_state_index(x)])

    def replace_fn(module):
        return module.update_state(state_index_to_update[int(module.state_index.id)])

    return eqx.tree_at(get_modules_to_update, loss, replace_fn=replace_fn)


def _proposal(sequence, g, temp):
    input = jax.nn.one_hot(sequence, 20)
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum() - g_i_x_i + g) / temp
    return jax.nn.softmax(logits), jax.nn.log_softmax(logits)


def gradient_MCMC(
    loss,
    sequence: Int[Array, "N"],
    temp=0.001,
    proposal_temp=0.01,
    max_path_length=2,
    steps=50,
    key: None = None,
    detailed_balance: bool = False,
):
    """
    Implements the gradient-assisted MCMC sampler from "Plug & Play Directed Evolution of Proteins with
    Gradient-based Discrete MCMC." Uses first-order taylor approximation of the loss to propose mutations.

        WARNING: Fixes random seed used for loss evaluation.

    Args:
    - loss: log-probability/function to minimize
    - sequence: initial sequence
    - proposal_temp: temperature of the proposal distribution
    - temp: temperature for the loss function
    - max_path_length: maximum number of mutations per step
    - steps: number of optimization steps
    - key: jax random key
    - detailed_balance: whether to maintain detailed balance

    """

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    key_model = key
    (v_0, aux_0), g_0 = _eval_loss_and_grad(
        loss, jax.nn.one_hot(sequence, 20), key=key_model
    )
    for iter in range(steps):
        ### generate a proposal

        _print_iter(iter, aux_0, v_0)
        proposal = sequence.copy()
        mutations = []
        log_q_forward = 0.0
        path_length = jax.random.randint(
            key=jax.random.key(np.random.randint(10000)),
            minval=1,
            maxval=max_path_length + 1,
            shape=(),
        )
        key = jax.random.fold_in(key, 0)
        for _ in range(path_length):
            p, log_p = _proposal(proposal, g_0, proposal_temp)
            mut_idx = jax.random.choice(
                key=key,
                a=len(np.ravel(p)),
                p=np.ravel(p),
                shape=(),
            )
            key = jax.random.fold_in(key, 0)
            position, AA = np.unravel_index(mut_idx, p.shape)
            log_q_forward += log_p[position, AA]
            mutations += [(position, AA)]
            proposal = proposal.at[position].set(AA)

        ### evaluate the proposal
        (v_1, aux_1), g_1 = _eval_loss_and_grad(
            loss, jax.nn.one_hot(proposal, 20), key=key_model
        )

        # next bit is to calculate the backward probability, which is only used
        # if detailed_balance is True
        prop_backward = proposal.copy()
        log_q_backward = 0.0
        for position, AA in reversed(mutations):
            p, log_p = _proposal(prop_backward, g_1, proposal_temp)
            log_q_backward += log_p[position, AA]
            prop_backward = prop_backward.at[position].set(AA)

        log_acceptance_probability = (v_0 - v_1) / temp + (
            (log_q_backward - log_q_forward) if detailed_balance else 0.0
        )

        log_acceptance_probability = min(0.0, log_acceptance_probability)

        print(
            f"iter: {iter}, accept {np.exp(log_acceptance_probability): 0.3f} {v_0: 0.3f} {v_1: 0.3f} {log_q_forward: 0.3f} {log_q_backward: 0.3f}"
        )
        print()
        if -jax.random.exponential(key=key) < log_acceptance_probability:
            sequence = proposal
            (v_0, aux_0), g_0 = (v_1, aux_1), g_1

        key = jax.random.fold_in(key, 0)

    return sequence


def projection_simplex(V, z=1):
    """
    From https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    """
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)



def simplex_APGM(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key=None,
    max_gradient_norm: float = 1.0,
    update_loss_state: bool = False,
    scale=1.0,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
    logspace: bool = False,
):
    """
    Accelerated projected gradient descent on the simplex.
    
    Args:
    - loss_function: function to minimize
    - x: initial sequence
    - n_steps: number of optimization steps
    - stepsize: step size for gradient descent
    - momentum: momentum factor
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - update_loss_state: whether to update the loss function state
    - scale: proximal scaling factor for L2 regularization (or entropic regularization if logspace=True), set to > 1.0 to encourage sparsity 
    - trajectory_fn: function to compute trajectory information, takes (aux, x) and returns any value.
    - logspace: whether to optimize in log space, which corresponds to a bregman proximal algorithm.

    returns:
    - x: final soft sequence after optimization
    - best_x: best soft sequence found during optimization
    - trajectory: list of trajectory information if `trajectory_fn` is provided, otherwise nothing.
    """

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    best_val = np.inf
    x = projection_simplex(x) if not logspace else x
    best_x = x

    x_prev = x

    trajectory = []

    for _iter in range(n_steps):
        v = jax.device_put(x + momentum * (x - x_prev))
        (value, aux), g = _eval_loss_and_grad(x=v if not logspace else jax.nn.softmax(v), loss_function=loss_function, key=key)
        if update_loss_state:
            loss_function = update_states(aux, loss_function)

        if trajectory_fn is not None:
            trajectory.append(trajectory_fn(aux, x))

        n = np.sqrt((g**2).sum())
        if n > max_gradient_norm:
            g = g * (max_gradient_norm / n)

        key = jax.random.fold_in(key, 0)
        
        if logspace:
            x_new = scale * (v - stepsize * g)
        else:
            x_new = projection_simplex(scale * (v - stepsize * g))

        x_prev = x
        x = x_new

        if value < best_val:
            best_val = value
            best_x = (
                x  # this isn't exactly right, because we evaluated loss at v, not x.
            )

        average_nnz = (x > 0.01).sum(-1).mean() if not logspace else (jax.nn.softmax(x) > 0.01).sum(-1).mean()
        _print_iter(
            _iter,
            eqx.filter(
                {"nnz": average_nnz, "aux": aux},
                lambda v: isinstance(v, float) or v.shape == (),
            ),
            value,
        )


    if logspace:
        x = jax.nn.softmax(x)
        best_x = jax.nn.softmax(best_x)

    if trajectory_fn is None:
        return x, best_x
    else:
        return x, best_x, trajectory

