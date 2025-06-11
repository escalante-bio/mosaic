import equinox as eqx
import jax
import numpy as np
import optax
from jaxtyping import Array, Float, Int


def _print_iter(iter, aux, v):
    print(
        iter,
        f"loss: {v:0.2f}",
        " ".join(
            f"{k}:{v: 0.2f}"
            for (k, v) in aux.items()
            if hasattr(v, "item") or isinstance(v, float)
        ),
    )


# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
@eqx.filter_jit
def _eval_loss_and_grad(loss_function, x, key):
    return eqx.filter_value_and_grad(loss_function, has_aux=True)(
        x,
        key=key,
    )


def _bregman_step_optax(*, optim, opt_state, x, loss_function, key):
    (v, aux), g = _eval_loss_and_grad(
        loss_function=loss_function, x=jax.nn.softmax(x), key=key
    )
    # remove per-residue mean
    g = g - g.mean(1, keepdims=True)
    updates, opt_state = optim.update(g, opt_state, x)
    x = optax.apply_updates(x, updates)

    x = jax.nn.log_softmax(x)  # do we need this?
    return x, v, aux, opt_state


def design_bregman_optax(
    *,
    loss_function,
    x,
    n_steps: int,
    optim=optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1e-1)),
):
    opt_state = optim.init(x)
    best = x
    best_v = np.inf
    for _iter in range(n_steps):
        x, v, aux, opt_state = _bregman_step_optax(
            x=x,
            loss_function=loss_function,
            key=jax.random.key(np.random.randint(0, 10000)),
            optim=optim,
            opt_state=opt_state,
        )

        entropy = -(jax.nn.log_softmax(x) * jax.nn.softmax(x)).sum(-1).mean()
        _print_iter(iter=_iter, aux={"entropy": entropy} | aux, v=v)
        if v < best_v:
            best = x
            best_v = v

    return x, best


# Manually implement backprop here to avoid recompilation when our optimizer changes
def _softmax_value_and_grad(optim, opt_state, params, t, loss_function, key):
    def colab_x(param):
        return t * jax.nn.softmax(param) + (1 - t) * param

    x, vjp = jax.vjp(colab_x, params)
    (v, aux), g = _eval_loss_and_grad(loss_function=loss_function, x=x, key=key)

    (g,) = vjp(g)
    updates, opt_state = optim.update(g, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, v, aux, opt_state


def design_softmax(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    optim=optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1e-1)),
    key=None,
):
    """
    ColabDesign-style optimization with logits + softmax.

    At iteration `I,` take one step w.r.t the gradient of

            loss_function(t * softmax(x) + (1 - t) * x)

    where t = I / (n_steps-1).

    Args:
    - loss_function: function to minimize
    - x: initial sequence
    - n_steps: number of optimization steps
    - optim: optax optimizer

    """

    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))
    opt_state = optim.init(x)

    for _iter in range(n_steps):
        t = _iter / (n_steps - 1)
        x, v, aux, opt_state = _softmax_value_and_grad(
            params=x,
            t=t,
            loss_function=loss_function,
            key=key,
            optim=optim,
            opt_state=opt_state,
        )
        key = jax.random.fold_in(key, 0)

        entropy = -(jax.nn.log_softmax(x) * jax.nn.softmax(x)).sum(-1).mean()
        _print_iter(_iter, {"entropy": entropy} | aux, v)

    return x


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
        key = jax.random.PRNGKey(np.random.randint(0, 10000))

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


def simplex_projected_gradient_descent(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    optim=None | optax.GradientTransformation,
    key=None,
):
    """
    Projected gradient descent on the simplex.

    Args:
    - loss_function: function to minimize
    - x: initial sequence
    - n_steps: number of optimization steps
    - optim: optax optimizer (or None for default)
    - key: jax random key

    """
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))

    if optim is None:
        binder_length = x.shape[0]
        optim = (
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.sgd(0.1 * np.sqrt(binder_length)),
            ),
        )

    opt_state = optim.init(x)

    best_val = np.inf
    best_x = x

    for _iter in range(n_steps):
        (v, aux), g = _eval_loss_and_grad(x=x, loss_function=loss_function, key=key)
        key = jax.random.fold_in(key, 0)

        updates, opt_state = optim.update(g, opt_state, x)
        x = optax.apply_updates(x, updates)
        x = projection_simplex(x)

        if v < best_val:
            best_val = v
            best_x = x

        _print_iter(_iter, aux, v)

    return x, best_x


def box_projected_gradient_descent(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    optim=None | optax.GradientTransformation,
    key=None,
):
    """
    Projected gradient descent on the box [0, 1]^N.

    """
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))
    if optim is None:
        binder_length = x.shape[0]
        optim = (
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.sgd(0.1 * np.sqrt(binder_length)),
            ),
        )

    opt_state = optim.init(x)

    best_val = np.inf
    best_x = x

    for _iter in range(n_steps):
        (v, aux), g = _eval_loss_and_grad(x=x, loss_function=loss_function, key=key)
        key = jax.random.fold_in(key, 0)

        updates, opt_state = optim.update(g, opt_state, x)
        x = optax.apply_updates(x, updates).clip(0, 1)

        if v < best_val:
            best_val = v
            best_x = x

        _print_iter(_iter, aux, v)

    return x, best_x



def simplex_APGM(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float,
    key=None,
    max_gradient_norm: float = 1.0,
):
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))

    best_val = np.inf
    x = projection_simplex(x)
    best_x = x

    x_prev = x

    for _iter in range(n_steps):
        v = jax.device_put(x + momentum * (x - x_prev))
        (value, aux), g = _eval_loss_and_grad(
            x=v, loss_function=loss_function, key=key
        )
        n = np.sqrt((g**2).sum())
        if n > max_gradient_norm:
            g = g * (max_gradient_norm / n)
        
        key = jax.random.fold_in(key, 0)

        x_new = projection_simplex(v- stepsize*g)
        x_prev = x
        x = x_new

        if value < best_val:
            best_val = value
            best_x = x # this isn't exactly right, because we evaluated loss at v, not x.


        _print_iter(_iter, aux, value)

    return x, best_x