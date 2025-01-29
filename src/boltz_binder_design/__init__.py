### Binder design using hallucination (following ColabFold)
import equinox as eqx
import jax
import numpy as np
import optax


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
        _print_iter(_iter, aux, entropy, v)
        if v < best_v:
            best = x
            best_v = v

    return x, best


def _print_iter(iter, aux, entropy, v):
    print(
        iter,
        f"loss: {v:0.2f} [entropy: {entropy:0.2f}]",
        " ".join(f"{k}:{v: 0.2f}" for (k, v) in aux.items() if hasattr(v, "item")),
    )
