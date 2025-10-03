# pyright: reportMissingTypeStubs=false
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Any as _Any
optax: _Any = __import__("optax")

from mosaic.optimizers import simplex_APGM as mosaic_simplex_APGM
from mosaic.optimizers import gradient_MCMC as mosaic_gradient_MCMC
from mosaic.optimizers import update_states


def _eval_loss_and_grad(loss_function, x, key):
    # Keep data on device; avoid host copies. Ensure dtype without leaving JAX.
    x = jnp.asarray(x, dtype=jnp.float32)
    (v, aux), g = _jit_value_and_grad(loss_function, x=x, key=key)
    return (v, aux), g - g.mean(axis=-1, keepdims=True)


@eqx.filter_jit
def _jit_value_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


def _apply_transforms(kind: str, transforms: dict | None, arr, ctx):
    if not transforms:
        return arr
    fns = transforms.get(kind) or []
    for fn in fns:
        arr = fn(arr, ctx)
    return arr

def simplex_APGM_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, logspace: bool = False, update_loss_state: bool = False, **kwargs):
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    stepsize = float(sched0.get("stepsize", 0.1))
    scale = float(sched0.get("scale", 1.0))

    # pre-transform logits to probs if user asks
    logits = _apply_transforms("pre_logits", transforms, x, {"schedule": sched0})
    probs = jax.nn.softmax(logits)
    probs = _apply_transforms("pre_probs", transforms, probs, {"schedule": sched0})

    def traj(aux, x_soft):
        if trajectory_fn is None:
            return None
        return trajectory_fn(aux, x_soft)

    x_soft, best_x_soft, tr = mosaic_simplex_APGM(
        loss_function=loss_function,
        x=probs if not logspace else logits,
        n_steps=n_steps,
        stepsize=stepsize,
        key=key,
        scale=scale,
        trajectory_fn=traj,
        logspace=logspace,
        update_loss_state=update_loss_state,
    )

    # map back to logits
    if logspace:
        x_logits = x_soft
        best_logits = best_x_soft
    else:
        x_logits = jnp.log(jnp.clip(x_soft, 1e-9, 1.0))
        best_logits = jnp.log(jnp.clip(best_x_soft, 1e-9, 1.0))

    x_logits = _apply_transforms("post_logits", transforms, x_logits, {"schedule": sched0})
    best_logits = _apply_transforms("post_logits", transforms, best_logits, {"schedule": sched0})
    return x_logits, best_logits, tr


def gradient_MCMC_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    temp = float(sched0.get("temperature", 0.001))
    proposal_temp = float(sched0.get("proposal_temp", 0.01))

    logits = _apply_transforms("pre_logits", transforms, x, {"schedule": sched0})
    probs = jax.nn.softmax(logits)
    probs = _apply_transforms("pre_probs", transforms, probs, {"schedule": sched0})
    seq = jnp.argmax(probs, axis=-1).astype(jnp.int32)

    seq = mosaic_gradient_MCMC(
        loss=loss_function,
        sequence=jnp.asarray(seq),
        temp=temp,
        proposal_temp=proposal_temp,
        steps=n_steps,
        key=key,
    )
    x_logits = jax.nn.one_hot(seq, 20)
    x_logits = _apply_transforms("post_logits", transforms, x_logits, {"schedule": sched0})
    return x_logits, x_logits, None


def rao_gumbel_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    """Rao-Blackwellized straight-through Gumbel estimator (hard forward, conditional Gumbel surrogate).

    Forward uses a hard one-hot sample D. Gradients flow through a surrogate
    that averages K conditional Gumbel-Softmax relaxations softmax((logits + G|D)/T).
    Conditional noise is stop_gradient'ed to avoid backprop through G|D.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    best_val = np.inf
    best_x = x

    # Prebuild per-task compiled value_and_grad functions to avoid per-step recompiles
    from mosaic.common import LinearCombination
    task_specs = []
    if isinstance(loss_function, LinearCombination):
        for w, l in zip(loss_function.weights, loss_function.l):
            task_specs.append((float(w), l))
    else:
        task_specs.append((1.0, loss_function))

    compiled_fns = []
    for (w, loss_term) in task_specs:
        def make_task_fn(term, weight):
            def loss_i(p, *, key):
                v, aux = term(p, key=key)
                v = jnp.asarray(v) * float(weight)
                return v, aux
            return loss_i
        compiled_fns.append(eqx.filter_value_and_grad(make_task_fn(loss_term, w), has_aux=True))
    logits = x
    K = int(kwargs.get("num_samples", 4))

    def _conditional_gumbel_noise(rng_key, logits, D, k):
        # E ~ Exp(1) shape [k, L, A]
        E = jax.random.exponential(rng_key, shape=(k,) + logits.shape)
        # Ei: [k, L, 1]
        Ei = jnp.sum(D[None, ...] * E, axis=-1, keepdims=True)
        Z = jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
        # adjusted logits s.t. argmax(logits + noise) = D
        adjusted = (D[None, ...] * (-jnp.log(Ei + 1e-12) + jnp.log(Z + 1e-12)) +
                    (1.0 - D[None, ...]) * -jnp.log(E / (jnp.exp(logits)[None, ...] + 1e-12) + Ei / (Z + 1e-12)))
        # conditional noise G|D = adjusted - logits
        cond = adjusted - logits[None, ...]
        return jax.lax.stop_gradient(cond)

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)
        temp = float(sched.get("temperature", 1.0))

        # Split keys for sampling
        key, k_cat, k_g = jax.random.split(key, 3)

        # Hard sample D (one-hot)
        idx = jax.random.categorical(k_cat, logits=logits, axis=-1)
        D = jax.nn.one_hot(idx, logits.shape[-1])

        # Conditional Gumbel surrogates
        cond = _conditional_gumbel_noise(k_g, logits, D, K)
        adjusted = logits[None, ...] + cond
        surrogate = jax.nn.softmax(adjusted / max(1e-6, temp), axis=-1).mean(axis=0)

        # Replace-gradient: forward hard D, gradient from surrogate
        probs_input = D + (surrogate - jax.lax.stop_gradient(surrogate))
        probs_input = _apply_transforms("pre_probs", transforms, probs_input, ctx)

        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs_input, key=key)
        if update_loss_state:
            loss_function = update_states(aux, loss_function)
        # key already advanced via split above

        g = _apply_transforms("grad", transforms, g, ctx)
        logits = logits - float(sched.get("lr", 0.1)) * g
        logits = _apply_transforms("post_logits", transforms, logits, ctx)

        if float(value) < best_val:
            best_val = float(value)
            best_x = logits

        if trajectory_fn is not None:
            aux = {"loss": float(value), "aux": aux}
            trajectory_fn(aux, probs_input)

    return logits, best_x, None


def st_gumbel_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    best_val = np.inf
    best_x = x
    logits = x
    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)
        temp = float(sched.get("temperature", 1.0))
        key, k_u = jax.random.split(key)
        gumbel = -jnp.log(-jnp.log(jax.random.uniform(k_u, logits.shape) + 1e-8) + 1e-8)
        y = (logits + gumbel) / max(1e-6, temp)
        probs_relaxed = jax.nn.softmax(y, axis=-1)
        probs_relaxed = _apply_transforms("pre_probs", transforms, probs_relaxed, ctx)
        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs_relaxed, key=key)
        if update_loss_state:
            loss_function = update_states(aux, loss_function)
        # key already advanced via split above
        g = _apply_transforms("grad", transforms, g, ctx)
        logits = logits - float(sched.get("lr", 0.1)) * g
        logits = _apply_transforms("post_logits", transforms, logits, ctx)
        if value < best_val:
            best_val = float(value)
            best_x = logits
        if trajectory_fn is not None:
            aux = {"loss": float(value), "aux": aux}
            trajectory_fn(aux, probs_relaxed)
    return logits, best_x, None


def zgr_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    best_val = np.inf
    best_x = x
    logits = x
    clip = float(kwargs.get("gradient_clip", 10.0))
    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)

        # Discrete sample for forward
        key, k_cat = jax.random.split(key)
        idx = jax.random.categorical(k_cat, logits=logits, axis=-1)
        x_onehot = jax.nn.one_hot(idx, logits.shape[-1])

        # ZGR surrogate: 0.5*(ST + DARN(phi_bar))
        p = jax.nn.softmax(logits, axis=-1)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        log_px = jnp.take_along_axis(log_p, idx[..., None], axis=-1)[..., 0]
        log_px = jnp.clip(log_px, a_min=-clip, a_max=clip)

        dx_st = p
        dx_darn = (x_onehot - jax.lax.stop_gradient(p)) * log_px[..., None]
        dx = 0.5 * (dx_st + dx_darn)

        # Straight-through: forward x_onehot; backward through dx
        probs_input = x_onehot + (dx - jax.lax.stop_gradient(dx))
        probs_input = _apply_transforms("pre_probs", transforms, probs_input, ctx)
        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs_input, key=key)
        if update_loss_state:
            loss_function = update_states(aux, loss_function)
        # key already advanced via split above
        g = _apply_transforms("grad", transforms, g, ctx)
        logits = logits - float(sched.get("lr", 0.1)) * g
        logits = _apply_transforms("post_logits", transforms, logits, ctx)
        if value < best_val:
            best_val = float(value)
            best_x = logits
        if trajectory_fn is not None:
            aux = {"loss": float(value), "aux": aux}
            trajectory_fn(aux, probs_input)
    return logits, best_x, None


def semi_greedy_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, proposals_per_step: int = 10, position_weighting: str = "1-plddt", **kwargs):
    """Discrete mutation search guided by model confidence.

    Expects that calling `loss_function(probs, key)` returns (value, aux) where aux may
    contain "plddt_per_residue" (e.g., when combined with a reporter loss term).
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    logits = x
    best_x = x
    best_val = np.inf

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = _apply_transforms("pre_probs", transforms, probs, ctx)

        # Germinal-style convergence stopping: honor ctx["stop_metric"] if set by transforms
        sm = ctx.get("stop_metric") if isinstance(ctx, dict) else None
        stop_metric = dict(sm) if isinstance(sm, dict) and bool(sm.get("met")) else None

        (value, aux), _ = _eval_loss_and_grad(loss_function, x=probs, key=key)
        if update_loss_state:
            loss_function = update_states(aux, loss_function)
        key = jax.random.fold_in(key, 0)

        # derive per-position weights
        binder_len = probs.shape[0]
        if position_weighting == "1-plddt" and isinstance(aux, dict):
            # Try to extract pLDDT vector from aux in a Mosaic-compatible way
            plddt = None
            # Direct
            plddt = aux.get("plddt_per_residue") if plddt is None else plddt
            # Under Boltz1 Loss wrapper (may be a list of aux dicts)
            inner = aux.get("boltz1")
            if plddt is None and isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict) and "plddt_per_residue" in item:
                        plddt = item["plddt_per_residue"]
                        break
            if plddt is None and isinstance(inner, dict):
                plddt = inner.get("plddt_per_residue")

            if plddt is not None:
                w = np.array(jnp.asarray(1.0 - plddt))
            else:
                w = np.ones((binder_len,), dtype=np.float32)
        else:
            w = np.ones((binder_len,), dtype=np.float32)
        w = w / (w.sum() + 1e-8)

        # generate proposals
        rng = np.random.default_rng(int(jnp.abs(jnp.sum(jnp.asarray(probs)*1e6))) % (2**32-1))
        candidates = []
        scores = []
        lam = float(sched.get("iglm_scale", 0.0))
        iglm_model = None
        if lam > 0.0:
            import torch  # type: ignore
            from iglm import IgLM  # type: ignore
            class _IgLMWrap(torch.nn.Module, IgLM):
                def __init__(self):
                    torch.nn.Module.__init__(self)
                    IgLM.__init__(self, model_name="IgLM")
                    self.model.to(self.device)
                    for p in self.model.parameters():
                        p.requires_grad = False
                    self.aa = list("ARNDCQEGHILKMFPSTWYV")
                    self.aa_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(a) for a in self.aa], device=self.device)
                    self.chain_id = self.tokenizer.convert_tokens_to_ids("[HEAVY]")
                    self.species_id = self.tokenizer.convert_tokens_to_ids("[HUMAN]")
                    self.sep = self.tokenizer.sep_token_id
                def ll_from_onehot(self, onehot_np, temp: float = 0.6):
                    import torch.nn.functional as F
                    onehot = torch.tensor(onehot_np, device=self.device, dtype=torch.float32)
                    embed = self.model.get_input_embeddings()(self.aa_ids)
                    var = onehot @ embed
                    prefix = self.model.get_input_embeddings()(torch.tensor([self.chain_id, self.species_id], device=self.device))
                    suffix = self.model.get_input_embeddings()(torch.tensor([self.sep], device=self.device))
                    full = torch.cat([prefix, var, suffix], dim=0).unsqueeze(0)
                    out = self.model(inputs_embeds=full)
                    logits_full = out.logits
                    var_token_ids = self.aa_ids[onehot.argmax(dim=-1)]
                    tgt = torch.cat([torch.tensor([self.chain_id, self.species_id], device=self.device), var_token_ids, torch.tensor([self.sep], device=self.device)], dim=0).unsqueeze(0)
                    loss = F.cross_entropy(logits_full[:, :-1, :].reshape(-1, logits_full.size(-1)), tgt[:, 1:].reshape(-1), reduction='mean')
                    return float(-loss.detach().cpu().item())
            iglm_model = _IgLMWrap()

        for t in range(int(sched.get("proposals_per_step", proposals_per_step))):
            i = rng.choice(np.arange(binder_len), p=w)
            p_i = np.array(probs[i])
            p_i = p_i / (p_i.sum() + 1e-8)
            aa = rng.choice(np.arange(p_i.shape[-1]), p=p_i)
            # discrete one-hot sequence from probs, mutate position i
            seq = np.eye(probs.shape[-1], dtype=np.float32)[np.argmax(np.array(probs), axis=-1)]
            seq[i] = 0.0
            seq[i, aa] = 1.0
            seq = jnp.asarray(seq)
            v, _ = loss_function(seq, key=key)
            score = float(v)
            if lam > 0.0:
                assert iglm_model is not None
                ll = iglm_model.ll_from_onehot(np.array(seq))  # type: ignore[attr-defined]
                score = float(v) - lam * float(ll)
            candidates.append(seq)
            scores.append(score)

        # pick best (lowest loss)
        best_idx = int(np.argmin(scores)) if scores else -1
        if best_idx >= 0 and scores[best_idx] < float(value):
            chosen = candidates[best_idx]
            logits = jnp.where(chosen > 0.5, 10.0, -10.0)
            value = scores[best_idx]

        logits = _apply_transforms("post_logits", transforms, logits, ctx)

        if float(value) < best_val:
            best_val = float(value)
            best_x = logits

        if trajectory_fn is not None:
            trajectory_fn({"loss": float(value), "aux": aux}, jax.nn.softmax(logits, axis=-1))

    return logits, best_x, None


def rso_box(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, optim=None, update_loss_state: bool = False, **kwargs):
    """Box-constrained optax optimizer over probabilities.

    This optimizer treats `x` as probabilities in [0,1] (not necessarily simplex) and updates them
    directly using an Optax optimizer. It supports the `pre_probs` and `grad` transform chains.

    Notes:
    - If you need simplex constraints, use `simplex_APGM_adapter` instead.
    - Schedules can still be used; we scale updates by the schedule LR each step.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    # default optimizer: clip then SGD with unit LR; per-step LR comes from schedule
    if optim is None:
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=1.0))
    opt_state = optim.init(x)

    best_val = np.inf
    best_x = x

    # Build task specs once and prepare compiled value_and_grad functions per task
    from mosaic.common import LinearCombination
    task_specs = []
    if isinstance(loss_function, LinearCombination):
        for w, l in zip(loss_function.weights, loss_function.l):
            task_specs.append((float(w), l))
    else:
        task_specs.append((1.0, loss_function))

    compiled_fns = []
    for (w, loss_term) in task_specs:
        def make_task_fn(term, weight):
            def loss_i(p, *, key):
                v, aux = term(p, key=key)
                v = jnp.asarray(v) * float(weight)
                return v, aux
            return loss_i
        compiled_fns.append(eqx.filter_value_and_grad(make_task_fn(loss_term, w), has_aux=True))

    # Fast path: fully JIT-compile the inner loop with lax.scan when transforms/state/trajectory are disabled
    if (transforms is None) and (trajectory_fn is None) and (not bool(update_loss_state)):
        # Precompute per-step LR array on host
        lrs = []
        for step in range(n_steps):
            sched = schedule(step, step) if callable(schedule) else (schedule or {})
            lrs.append(float(sched.get("learning_rate", sched.get("lr", 0.1))))
        lr_arr = jnp.asarray(lrs, dtype=jnp.float32)

        def _step(carry, inputs):
            xx, os_, sk, lr = carry
            (value, _aux), g = _eval_loss_and_grad(loss_function, x=xx, key=sk)
            updates, os_next = optim.update(g, os_, xx)
            # scale updates by lr without lambda to appease static type checker
            def _scale(u):
                return u * lr
            updates = jax.tree.map(_scale, updates)
            xx_next = optax.apply_updates(xx, updates)
            sk_next = jax.random.fold_in(sk, 0)
            return (xx_next, os_next, sk_next, lr), value

        init = (x, opt_state, key, lr_arr[0])
        (x, opt_state, key, _), _ = jax.lax.scan(_step, init, lr_arr)
        return jnp.clip(x, 0.0, 1.0), jnp.clip(x, 0.0, 1.0), None

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}

        # Treat x as probs; allow transforms on pre_probs
        probs = _apply_transforms("pre_probs", transforms, x, ctx)

        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs, key=key)
        if update_loss_state:
            loss_function = update_states(aux, loss_function)
        key = jax.random.fold_in(key, 0)

        # apply grad transforms
        g = _apply_transforms("grad", transforms, g, ctx)

        updates, opt_state = optim.update(g, opt_state, x)

        # scale updates by schedule LR
        lr = float(sched.get("learning_rate", sched.get("lr", 0.1)))
        def _scale(u):
            return u * lr
        updates = jax.tree.map(_scale, updates)

        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, 0.0, 1.0)

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        if trajectory_fn is not None:
            aux = {"loss": float(value), "aux": aux}
            trajectory_fn(aux, x)

    return x, best_x, None


def optax_logits(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, optim=None, update_loss_state: bool = False, **kwargs):
    """Optax-based optimizer on logits (with softmax forward), Mosaic-style.

    - Applies pre_logits → softmax → pre_probs → loss
    - Gradients are taken w.r.t. probs and used to update the transformed logits (same approximation as sgd_logits)
    - Applies post_logits after each step
    - Learning rate comes from schedule; the optax chain should use lr=1.0 internally
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    if optim is None:
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=1.0))
    opt_state = optim.init(x)

    best_val = np.inf
    best_x = x

    # Build per-task compiled value_and_grad functions once for JD
    from mosaic.common import LinearCombination as _LC
    _jd_task_specs = []
    if isinstance(loss_function, _LC):
        for _w, _l in zip(loss_function.weights, loss_function.l):
            _jd_task_specs.append((float(_w), _l))
    else:
        _jd_task_specs.append((1.0, loss_function))

    _jd_compiled = []
    for (_w, _term) in _jd_task_specs:
        def _mk(term, weight):
            def _li(p, *, key):
                v, aux = term(p, key=key)
                v = jnp.asarray(v) * float(weight)
                return v, aux
            return _li
        _jd_compiled.append(eqx.filter_value_and_grad(_mk(_term, _w), has_aux=True))

    # Fast path: fully JIT-compile inner loop with lax.scan when transforms/state/trajectory are disabled
    if (transforms is None) and (trajectory_fn is None) and (not bool(update_loss_state)):
        # Precompute LR array
        lrs = []
        for step in range(n_steps):
            sched = schedule(step, step) if callable(schedule) else (schedule or {})
            lrs.append(float(sched.get("learning_rate", sched.get("lr", 0.1))))
        lr_arr = jnp.asarray(lrs, dtype=jnp.float32)

        def _step(carry, lr):
            logits, os_, sk = carry
            probs = jax.nn.softmax(logits, axis=-1)
            (value, _aux), g = _eval_loss_and_grad(loss_function, x=probs, key=sk)
            updates, os_next = optim.update(g, os_, logits)
            def _scale(u):
                return u * lr
            updates = jax.tree.map(_scale, updates)
            logits_next = optax.apply_updates(logits, updates)
            sk_next = jax.random.fold_in(sk, 0)
            return (logits_next, os_next, sk_next), value

        init = (x, opt_state, key)
        (x, opt_state, key), _ = jax.lax.scan(_step, init, lr_arr)
        return x, x, None

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}

        logits = _apply_transforms("pre_logits", transforms, x, ctx)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = _apply_transforms("pre_probs", transforms, probs, ctx)

        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs, key=key)
        if update_loss_state:
            loss_function = update_states(aux, loss_function)
        key = jax.random.fold_in(key, 0)

        g = _apply_transforms("grad", transforms, g, ctx)

        updates, opt_state = optim.update(g, opt_state, logits)
        lr = float(sched.get("learning_rate", sched.get("lr", 0.1)))
        def _scale(u):
            return u * lr
        updates = jax.tree.map(_scale, updates)
        logits = optax.apply_updates(logits, updates)
        x = _apply_transforms("post_logits", transforms, logits, ctx)

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        if trajectory_fn is not None:
            aux = {"loss": float(value), "aux": aux}
            trajectory_fn(aux, jax.nn.softmax(x, axis=-1))

    return x, best_x, None


def sgd_logits_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, clip: float = 1.0, momentum: float = 0.0, update_loss_state: bool = False, **kwargs):
    optim = optax.chain(optax.clip_by_global_norm(clip), optax.sgd(learning_rate=1.0, momentum=momentum))
    return optax_logits(
        loss_function=loss_function,
        x=x,
        n_steps=n_steps,
        key=key,
        schedule=schedule,
        transforms=transforms,
        trajectory_fn=trajectory_fn,
        aux_context=aux_context,
        optim=optim,
        update_loss_state=update_loss_state,
        **kwargs,
    )


def adamw_logits_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, clip: float = 1.0, weight_decay: float = 0.01, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, update_loss_state: bool = False, **kwargs):
    optim = optax.chain(optax.clip_by_global_norm(clip), optax.adamw(learning_rate=1.0, weight_decay=weight_decay, b1=b1, b2=b2, eps=eps))
    return optax_logits(
        loss_function=loss_function,
        x=x,
        n_steps=n_steps,
        key=key,
        schedule=schedule,
        transforms=transforms,
        trajectory_fn=trajectory_fn,
        aux_context=aux_context,
        optim=optim,
        update_loss_state=update_loss_state,
        **kwargs,
    )


@jax.jit
def jd_mean_aggregator(J):
    """Aggregate per-task gradients by simple mean across tasks.

    Args:
        J: jax.Array of shape [num_tasks, num_params]
    Returns:
        jax.Array of shape [num_params]
    """
    return jnp.mean(J, axis=0)


@jax.jit
def jd_pcgrad_aggregator(J, *, eps: float = 1e-12):
    """Project conflicting gradients (PCGrad), then average.

    Deterministic pairwise projection without randomization.

    Args:
        J: jax.Array of shape [num_tasks, num_params]
        eps: small constant to avoid division by zero
    Returns:
        jax.Array of shape [num_params]
    """
    m = J.shape[0]

    def proj_once(G_in):
        def body_i(i, G_acc):
            gi = G_acc[i]
            def body_j(j, gi_cur):
                gj = G_acc[j]
                dot = jnp.vdot(gi_cur, gj).real
                norm2 = jnp.vdot(gj, gj).real
                cond = dot < 0.0
                gi_new = gi_cur - jnp.where(cond, dot / (norm2 + eps), 0.0) * gj
                return gi_new
            gi_out = jax.lax.fori_loop(0, m, body_j, gi)
            G_next = G_acc.at[i].set(gi_out)
            return G_next
        return jax.lax.fori_loop(0, m, body_i, G_in)

    G_proj = proj_once(J)
    return jnp.mean(G_proj, axis=0)


def jd_upgrad_aggregator(J, *, pref_vector=None, norm_eps: float = 1e-4, reg_eps: float = 1e-4, solver: str = "quadprog"):
    """UPGrad (exact parity): solves the QP in weight space per TorchJD.

    minimize v^T G v subject to v >= u, with G = regularize(normalize(J J^T), reg_eps),
    u = diag(weights) row for each i (weights default to mean if pref_vector is None).
    Returns g = (sum_i w_i) @ J.
    """
    import numpy as _np
    from qpsolvers import solve_qp as _solve_qp  # type: ignore

    J_np = _np.asarray(J, dtype=_np.float64)
    m, p = J_np.shape
    G = J_np @ J_np.T
    tr = _np.trace(G)
    if tr < float(norm_eps):
        Gn = _np.zeros_like(G)
    else:
        Gn = G / tr
    Gn = Gn + float(reg_eps) * _np.eye(m, dtype=Gn.dtype)

    if pref_vector is None:
        wvec = _np.ones((m,), dtype=_np.float64) / float(m)
    else:
        wvec = _np.asarray(pref_vector, dtype=_np.float64)
        s = float(_np.sum(wvec))
        wvec = wvec / (s + 1e-12)

    U = _np.diag(wvec)  # [m,m]; each row i is u_i e_i
    W = _np.zeros_like(U)
    Ineg = -_np.eye(m, dtype=Gn.dtype)
    zeros = _np.zeros((m,), dtype=Gn.dtype)

    for i in range(m):
        u = U[i]
        w = _solve_qp(Gn, zeros, Ineg, -u, solver=solver)
        if w is None:
            raise ValueError("Failed to solve UPGrad QP (weights projection)")
        W[i] = w

    w_sum = _np.sum(W, axis=0)
    g = w_sum @ J_np
    return jnp.asarray(g, dtype=J.dtype)


def jacobian_descent_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, aggregator=None, update_loss_state: bool = False, **kwargs):
    """Jacobian Descent on logits with softmax forward, Mosaic-style.

    Computes per-task gradients for a LinearCombination loss (or a single LossTerm),
    aggregates them with a user-supplied aggregator over the Jacobian rows, and
    takes a step on logits. Mirrors the behavior of TorchJD-style aggregators but
    implemented in JAX over Mosaic primitives.

    Inputs/Outputs follow other adapters: pre_logits → softmax → pre_probs → loss,
    grad transforms applied on the aggregated gradient, then post_logits.

    Args:
        loss_function: Mosaic loss; LinearCombination or single LossTerm
        x: initial logits [L,20]
        n_steps: number of optimization steps
        schedule: callable (global_step, phase_step) -> dict with "lr" at least
        transforms: dict with optional chains: pre_logits, pre_probs, grad, post_logits
        aggregator: callable J -> g, default: mean across tasks
        update_loss_state: whether to call update_states on aux when available
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    if aggregator is None:
        aggregator = jd_mean_aggregator

    best_val = np.inf
    best_x = x

    # JD: prebuild per-task compiled value_and_grad functions once
    from mosaic.common import LinearCombination as _LC_JD
    _jd_task_specs = []
    if isinstance(loss_function, _LC_JD):
        for _w, _l in zip(loss_function.weights, loss_function.l):
            _jd_task_specs.append((float(_w), _l))
    else:
        _jd_task_specs.append((1.0, loss_function))

    _jd_compiled = []
    for (_w, _term) in _jd_task_specs:
        def _mk_jd(term, weight):
            def _li(p, *, key):
                v, aux = term(p, key=key)
                v = jnp.asarray(v) * float(weight)
                return v, aux
            return _li
        _jd_compiled.append(eqx.filter_value_and_grad(_mk_jd(_term, _w), has_aux=True))

    # Build per-task switch branches and a compiled value_and_grad once per call (reuse across steps)
    def _mk_branch(term, weight):
        def _f(args):
            p, k = args
            v, aux = term(p, key=k)
            return jnp.asarray(v) * float(weight), aux
        return _f
    _branches = [_mk_branch(_term, _w) for (_w, _term) in _jd_task_specs]
    def _loss_select(p, k, idx):
        return jax.lax.switch(idx, _branches, (p, k))
    def _select_loss_wrapped(p, k, i):
        return _loss_select(p, k, i)
    _loss_select_vg = eqx.filter_value_and_grad(_select_loss_wrapped, has_aux=True)
    m = len(_jd_task_specs)
    _idxs = jnp.arange(m, dtype=jnp.int32)

    # Fast path: compile inner loop with lax.scan when transforms/state/trajectory are disabled and aggregator is mean
    _can_fastpath = (transforms is None) and (trajectory_fn is None) and (not bool(update_loss_state)) and ((aggregator is None) or (aggregator is jd_mean_aggregator))
    if _can_fastpath:
        # Precompute per-step LR
        lrs = []
        for step in range(n_steps):
            sched = schedule(step, step) if callable(schedule) else (schedule or {})
            lrs.append(float(sched.get("learning_rate", sched.get("lr", 0.1))))
        lr_arr = jnp.asarray(lrs, dtype=jnp.float32)

        def _step(carry, lr):
            logits, sk = carry
            probs = jax.nn.softmax(logits, axis=-1)
            subkeys = jax.random.split(jax.random.fold_in(sk, 0), m)
            def _vmap_body(i, s):
                return _loss_select_vg(probs, s, i)
            (vals, _aux_list), grads = jax.vmap(_vmap_body)(_idxs, subkeys)
            J = jnp.reshape(grads, (m, -1))
            g_flat = jd_mean_aggregator(J)
            g = jnp.reshape(g_flat, probs.shape)
            logits_next = logits - lr * g
            sk_next = jax.random.fold_in(sk, 0)
            value = jnp.sum(vals)
            return (logits_next, sk_next), value

        init = (x, key)
        (x, key), _ = jax.lax.scan(_step, init, lr_arr)
        return x, x, None

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}

        logits = _apply_transforms("pre_logits", transforms, x, ctx)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = _apply_transforms("pre_probs", transforms, probs, ctx)

        # Capture convergence signal from pre_probs transforms, if provided
        stop_metric = ctx.get("stop_metric") if isinstance(ctx, dict) else None
        if not (isinstance(stop_metric, dict) and bool(stop_metric.get("met", False))):
            stop_metric = None

        # Evaluate per-task grads with a single vmapped compiled function to avoid Python overhead
        m = len(_jd_task_specs)
        subkeys = jax.random.split(jax.random.fold_in(key, 0), m)
        def _vmap_body2(i, sk):
            return _loss_select_vg(probs, sk, i)
        (vals, aux_list), grads = jax.vmap(_vmap_body2)(_idxs, subkeys)
        if hasattr(vals, "shape"):
            _vals_flat = jnp.ravel(vals)
            per_vals = [v for v in _vals_flat]
        else:
            per_vals = [vals]
        # aux_list is already a pytree list per task
        per_aux = list(aux_list)
        J_rows = [g.reshape(-1) for g in grads]
        key = jax.random.fold_in(key, 0)

        # Combine values for tracking and best_x (before applying update)
        value = jnp.sum(jnp.stack([jnp.asarray(v) for v in per_vals]))
        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        # If convergence threshold met and after minimum steps, stop early
        min_stop = int((sched or {}).get("min_stop_step", 5))
        if (stop_metric is not None) and (int(step) >= min_stop):
            if trajectory_fn is not None:
                aux = {"loss": float(value), "tasks": [float(jnp.asarray(v)) for v in per_vals], "aux": per_aux}
                raw_metrics = (ctx.get("metrics") or {}) if isinstance(ctx, dict) else {}
                metrics_dict = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
                metrics_dict = {**metrics_dict, "converged": True, "stop_metric": stop_metric}
                aux = {**aux, "metrics": metrics_dict}
                trajectory_fn(aux, jax.nn.softmax(x, axis=-1))
            break

        # Aggregate Jacobian rows to a single gradient
        J = jnp.stack(J_rows, axis=0)  # [M, P]
        g_flat = aggregator(J)
        g = g_flat.reshape(probs.shape)

        # Optional gradient transforms (e.g., sequence-norm, masking)
        g = _apply_transforms("grad", transforms, g, ctx)

        # Update logits in-place via LR
        lr = float(sched.get("learning_rate", sched.get("lr", 0.1)))
        logits = logits - lr * g
        x = _apply_transforms("post_logits", transforms, logits, ctx)

        # Combine values for tracking and best_x (post-update trajectory record uses new x)

        # Optionally update loss states using concatenated aux
        if update_loss_state:
            # Pack aux as a list to preserve per-task structure
            loss_function = update_states(per_aux, loss_function)

        if trajectory_fn is not None:
            aux = {"loss": float(value), "tasks": [float(jnp.asarray(v)) for v in per_vals], "aux": per_aux}
            metrics = (ctx.get("metrics") or {}) if isinstance(ctx, dict) else {}
            if metrics:
                aux["metrics"] = dict(metrics)
            trajectory_fn(aux, jax.nn.softmax(x, axis=-1))

    return x, best_x, None

