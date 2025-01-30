from . import _eval_loss_and_grad, _print_iter
from jaxtyping import Float, Array, Int
import numpy as np
import jax, optax


# We manually implement backprop here to avoid recompilation when our optimizer changes
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

    At iteration I take one step w.r.t the gradient of

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
        _print_iter(_iter, aux, entropy, v)

    return x


def _proposal(sequence, g, temp):
    input = jax.nn.one_hot(sequence, 20)
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum() - g_i_x_i + g) / temp
    return jax.nn.softmax(logits), jax.nn.log_softmax(logits)


def gradient_MCMC(
    loss,
    sequence: Int[Array, "N"],
    proposal_temp=2.0,
    temp=1.0,
    max_path_length=2,
    steps=50,
    key: None = None,
):
    """
    Implements the gradient-assisted MCMC sampler from "Plug & Play Directed Evolution of Proteins with
    Gradient-based Discrete MCMC"

    Args:
    - loss: function to minimize
    - sequence: initial sequence
    - proposal_temp: temperature of the proposal distribution
    - temp: temperature for the loss function
    - max_path_length: maximum number of mutations per step
    - steps: number of optimization
    - key: jax random key

    """

    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))

    key_model = key
    for iter in range(steps):
        ### generate a proposal
        (v_0, aux_0), g_0 = _eval_loss_and_grad(
            loss, jax.nn.one_hot(sequence, 20), key=key_model
        )
        _print_iter(iter, aux_0, 0, v_0)
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
        prop_backward = proposal.copy()
        log_q_backward = 0.0
        for position, AA in reversed(mutations):
            p,log_p  = _proposal(prop_backward, g_1, proposal_temp)
            log_q_backward += log_p[position, AA]
            prop_backward = prop_backward.at[position].set(AA)

        

        acceptance_probability = min(
            1, np.exp((v_0 - v_1) / temp + log_q_backward - log_q_forward)
        )
        print(f"iter: {iter}, accept {acceptance_probability: 0.3f} {v_0: 0.3f} {v_1: 0.3f} {log_q_forward: 0.3f} {log_q_backward: 0.3f}")
        print()
        if jax.random.uniform(key=key) < acceptance_probability:
            sequence = proposal
        key = jax.random.fold_in(key, 0)

    return sequence
