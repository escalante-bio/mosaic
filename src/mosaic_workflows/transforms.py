import numpy as np
import jax
import jax.numpy as jnp


def _apply_chain(fns, arr, ctx):
    if not fns:
        return arr
    for fn in fns:
        arr = fn(arr, ctx)
    return arr


def softmax_temperature_on_logits(temperature: float, on_logits: bool = True):
    def _fn(x, ctx):
        if on_logits:
            return jax.nn.log_softmax(x / max(1e-6, temperature))
        else:
            return jax.nn.softmax(x / max(1e-6, temperature))
    return _fn


def scale_logits(x, alpha: float):
    return x * alpha


def temperature_on_logits():
    """Return a pre_logits transform that divides logits by the current schedule temperature.

    Expects ctx["schedule"]["temperature"]. If missing, uses 1.0 (no-op).
    """
    def _fn(x, ctx):
        t = float((ctx.get("schedule") or {}).get("temperature", 1.0))
        return x / max(1e-6, t)
    return _fn


def e_soft_on_logits():
    """Flatten/sharpen logits by schedule['e_soft'] (e<1 flattens, e>1 sharpens)."""
    def _fn(x, ctx):
        e = float((ctx.get("schedule") or {}).get("e_soft", 1.0))
        return x * e
    return _fn


def token_restrict(allowed_tokens: list[int] | None = None, avoid_residues: list[str] | None = None):
    vocab = "ARNDCQEGHILKMFPSTWYV"
    avoid_idx = set(vocab.index(r) for r in (avoid_residues or []) if r in vocab)
    allowed_idx = set(allowed_tokens) if allowed_tokens is not None else set(range(20))
    masked = np.array([1.0 if (i in allowed_idx and i not in avoid_idx) else 0.0 for i in range(20)], dtype=np.float32)

    def _pre_probs(p, ctx):
        return p * masked

    return _pre_probs  # use in pre_probs typically


def token_restrict_post_logits(allowed_tokens: list[int] | None = None, avoid_residues: list[str] | None = None):
    vocab = "ARNDCQEGHILKMFPSTWYV"
    avoid_idx = set(vocab.index(r) for r in (avoid_residues or []) if r in vocab)
    allowed_idx = set(allowed_tokens) if allowed_tokens is not None else set(range(20))
    masked = np.array([1.0 if (i in allowed_idx and i not in avoid_idx) else 0.0 for i in range(20)], dtype=np.float32)

    def _post_logits(x, ctx):
        neg_inf = -1e9
        m = jnp.array(masked)
        # add -inf to disallowed positions
        return x + (1.0 - m) * neg_inf
    return _post_logits


def zero_disallowed(restrict_to_canon: bool = True, avoid_residues: list[str] | None = None):
    vocab = "ARNDCQEGHILKMFPSTWYV"
    avoid_idx = set(vocab.index(r) for r in (avoid_residues or []) if r in vocab)
    allowed_idx = set(range(20)) if restrict_to_canon else set(range(33))
    mask = np.array([1.0 if (i in allowed_idx and i not in avoid_idx) else 0.0 for i in range(20)], dtype=np.float32)

    def _grad(g, ctx):
        return g * mask

    return _grad


# Already available as a loss-level transform (we do not need a step level transform) in the mosaic library/losses/transforms.py
def gradient_normalizer(mode: str = "l2", eps: float = 1e-6, log_norm: bool = False):
    def _fn(g, ctx):
        if mode == "per_chain":
            # assumes ctx provides chain_indices for positions; fallback to global norm
            n = jnp.sqrt((g**2).sum())
            if log_norm:
                pass  # no-op; logging handled by caller's trajectory_fn
            return jnp.where(n > eps, g / (n + eps), g)
        if mode == "clip":
            n = jnp.sqrt((g**2).sum())
            return jnp.where(n > eps, g * (eps / (n + 1e-9)), g)
        if mode == "l2_effL":
            # Global L2 normalization scaled by sqrt(effective_length)
            # effective_length = number of positions with non-zero gradient vector
            per_pos_norm2 = jnp.sum(g * g, axis=-1, keepdims=True)
            eff_L = jnp.sum(per_pos_norm2 > 0.0, axis=-2, keepdims=True).astype(jnp.float32)
            n = jnp.sqrt((g**2).sum())
            scale = jnp.sqrt(jnp.maximum(eff_L, 1.0))
            return jnp.where(n > eps, g * scale / (n + eps), g)
        # l2 normalize by default
        n = jnp.sqrt((g**2).sum())
        return jnp.where(n > eps, g / (n + eps), g)
    return _fn


def hard_one_hot(st: bool = False):
    def _fn(x, ctx):
        idx = jnp.argmax(x, axis=-1)
        oh = jax.nn.one_hot(idx, 20)
        # map to crisp logits
        return oh * 10.0 + (1.0 - oh) * -10.0
    return _fn


def position_mask(mask: np.ndarray):
    """Mask positions (L,) with 1=free, 0=fixed; zero gradients at fixed, keep logits unchanged.

    Use with grad chain; for logits, pair with fixed_positions_logits below.
    """
    m = jnp.array(mask).astype(jnp.float32)
    def _grad(g, ctx):
        return g * m[:, None]
    return _grad


def fixed_positions_logits(mask: np.ndarray, fixed_logits: np.ndarray | None = None):
    """Override logits at fixed positions; if fixed_logits is None, keep original logits and only -inf disallowed tokens.
    """
    m = jnp.array(mask).astype(jnp.float32)  # 1 free, 0 fixed
    fixed = None if fixed_logits is None else jnp.array(fixed_logits)
    neg_inf = -1e9
    def _post(x, ctx):
        if fixed is None:
            return x * m[:, None] + (1.0 - m)[:, None] * neg_inf
        return x * m[:, None] + (1.0 - m)[:, None] * fixed
    return _post


def per_position_allowed_tokens(allowed: np.ndarray):
    """allowed: (L, 20) 1/0 mask. Applies -inf to disallowed tokens per position.
    """
    m = jnp.array(allowed).astype(jnp.float32)
    neg_inf = -1e9
    def _post(x, ctx):
        return x + (1.0 - m) * neg_inf
    return _post


def per_position_allowed_probs(allowed: np.ndarray):
    """allowed: (L, 20) 1/0 mask. Zeros disallowed tokens in probabilities and renormalizes.

    Use in pre_probs so probability-based optimizers (simplex APGM) respect hard identity clamps.
    """
    m = jnp.array(allowed).astype(jnp.float32)
    def _pre_probs(p, ctx):
        q = p * m
        q = q / (q.sum(-1, keepdims=True) + 1e-8)
        return q
    return _pre_probs



def record_probs_max_mean(mask: np.ndarray | None = None, key: str = "probs_max_mean"):
    """Pre-probs transform that records convergence metric avg(max(prob_i)) into ctx["metrics"].

    Args:
        mask: Optional (L,) boolean/float mask; 1 includes position, 0 excludes.
        key:  Metric name under ctx["metrics"].
    """
    m = None if mask is None else jnp.array(mask).astype(jnp.float32)
    def _pre_probs(p, ctx):
        # p: [L, 20]
        pm = p if m is None else p * m[:, None]
        denom = 1.0 if m is None else (jnp.sum(m) + 1e-8)
        val = jnp.sum(jnp.max(pm, axis=-1)) / denom
        metrics = ctx.setdefault("metrics", {})
        metrics[key] = float(val)
        return p
    return _pre_probs


def freeze_grad_on_metric(threshold: float, key: str = "probs_max_mean"):
    """Grad transform that zeros updates once a ctx metric reaches a threshold.

    Typical usage with record_probs_max_mean to stop updates when logits/probs converge.
    """
    thr = float(threshold)
    def _grad(g, ctx):
        val = float((ctx.get("metrics") or {}).get(key, 0.0))
        done = val >= thr
        if done:
            metrics = ctx.setdefault("metrics", {})
            metrics["converged"] = True
            return jnp.zeros_like(g)
        return g
    return _grad


def germinal_softmax_convergence(mask: np.ndarray | None, *, threshold: float, key: str = "probs_max_mean"):
    """Pre-probs transform that computes Germinal-style convergence: avg max(prob) over masked positions.

    Stores into ctx["metrics"][key] and ctx["stop_metric"] = {"key", "threshold", "value", "met"}.
    If mask is None, uses all positions.
    """
    m = None if mask is None else jnp.array(mask).astype(jnp.float32)

    def _pre_probs(p, ctx):
        pm = p if m is None else p * m[:, None]
        if m is None:
            denom = p.shape[0]
        else:
            denom = jnp.sum(m) + 1e-8
        val = jnp.sum(jnp.max(pm, axis=-1)) / denom
        metrics = ctx.setdefault("metrics", {})
        metrics[key] = float(val)
        ctx["stop_metric"] = {"key": key, "threshold": float(threshold), "value": float(val), "met": bool(val >= float(threshold))}
        return p

    return _pre_probs


