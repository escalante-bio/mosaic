import numpy as np
from typing import Any as _Any
import jax
import jax.numpy as jnp
import jax.dlpack as jdlpack


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
    masked_np = np.array([1.0 if (i in allowed_idx and i not in avoid_idx) else 0.0 for i in range(20)], dtype=np.float32)
    masked = jnp.array(masked_np)

    def _pre_probs(p, ctx):
        return p * masked

    return _pre_probs  # use in pre_probs typically


def token_restrict_post_logits(allowed_tokens: list[int] | None = None, avoid_residues: list[str] | None = None):
    vocab = "ARNDCQEGHILKMFPSTWYV"
    avoid_idx = set(vocab.index(r) for r in (avoid_residues or []) if r in vocab)
    allowed_idx = set(allowed_tokens) if allowed_tokens is not None else set(range(20))
    masked_np = np.array([1.0 if (i in allowed_idx and i not in avoid_idx) else 0.0 for i in range(20)], dtype=np.float32)
    m = jnp.array(masked_np)

    def _post_logits(x, ctx):
        neg_inf = -1e9
        # add -inf to disallowed positions
        return x + (1.0 - m) * neg_inf
    return _post_logits


def zero_disallowed(restrict_to_canon: bool = True, avoid_residues: list[str] | None = None):
    vocab = "ARNDCQEGHILKMFPSTWYV"
    avoid_idx = set(vocab.index(r) for r in (avoid_residues or []) if r in vocab)
    allowed_idx = set(range(20)) if restrict_to_canon else set(range(33))
    mask_np = np.array([1.0 if (i in allowed_idx and i not in avoid_idx) else 0.0 for i in range(20)], dtype=np.float32)
    mask = jnp.array(mask_np)

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


def record_logits_in_ctx():
    """Pre-logits transform that stores current logits into ctx for later grad transforms."""
    def _fn(x, ctx):
        if isinstance(ctx, dict):
            ctx.setdefault("tensors", {})["logits"] = x
        return x
    return _fn


def record_probs_in_ctx():
    """Pre-probs transform that stores current probabilities into ctx for later grad transforms."""
    def _fn(p, ctx):
        if isinstance(ctx, dict):
            ctx.setdefault("tensors", {})["probs"] = p
        return p
    return _fn


def zymctrl_pcgrad_merge(ec_label: str, temp: float = 1.0):
    """Grad transform: merge AF-M gradient with ZymCTRL prior (PCGrad/MGDA/scale).

    - Reads schedule['zymctrl_scale'] to weight ZymCTRL relative to AF-M.
    - Supports schedule['grad_merge_method'] in {"pcgrad","mgda","scale","none"}.
    - Mirrors IgLM merge semantics and post-merge normalization + sqrt(effective_length) scaling.
    """
    _state = {"model": None, "tokenizer": None}

    def _load():
        if _state["model"] is not None:
            return _state
        import torch  # type: ignore
        from transformers import GPT2LMHeadModel, AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL")
        mdl = GPT2LMHeadModel.from_pretrained("AI4PD/ZymCTRL")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(device)
        mdl.eval()
        for p in mdl.parameters():
            p.requires_grad = False
        # Build AA vocab ids
        aa = list("ARNDCQEGHILKMFPSTWYV")
        aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in aa], device=device)
        # Encode prefix and suffix tokens
        prefix_ids = tok.encode(ec_label + "<sep>", add_special_tokens=False)
        suffix_ids = tok.encode("<|endoftext|>", add_special_tokens=False)
        _state["model"] = mdl
        _state["tokenizer"] = tok
        _state["aa_ids"] = aa_ids
        _state["prefix_ids"] = torch.tensor(prefix_ids, device=device)
        _state["suffix_ids"] = torch.tensor(suffix_ids, device=device)
        _state["device"] = device
        return _state

    def _pc(a_in: jnp.ndarray, b_in: jnp.ndarray, lam: float) -> jnp.ndarray:
        a = jnp.reshape(a_in, (-1,))
        b = jnp.reshape(b_in, (-1,))
        a_n = jnp.linalg.norm(a) + 1e-12
        b_n = jnp.linalg.norm(b) + 1e-12
        def when_zero():
            return jnp.reshape(b_in * lam, a_in.shape)
        def when_ok():
            a_hat = a / a_n
            b_hat = b / b_n
            dot = jnp.vdot(a_hat, b_hat).real
            b_proj = jnp.where(dot < 0.0, b_hat - dot * a_hat, b_hat)
            merged = a_hat + float(lam) * b_proj
            return jnp.reshape(merged, a_in.shape)
        return jax.lax.cond(a_n <= 1e-11, when_zero, when_ok)

    def _grad(g, ctx):
        sched = (ctx or {}).get("schedule", {})
        # Optional frequency gating (default every step)
        k = int(sched.get("guidance_every_k", 1) or 1)
        step_idx = int((ctx or {}).get("phase_step", 0) or 0)
        if int(k) > 1 and (int(step_idx) % int(k) != 0):
            return jnp.asarray(g, dtype=jnp.float32)
        lam = float(sched.get("zymctrl_scale", 0.0))
        method = str(sched.get("grad_merge_method", "pcgrad")).lower()
        # Explicit disabled path
        if lam <= 0.0 or method == "none":
            return jnp.asarray(g, dtype=jnp.float32)
        tensors = (ctx or {}).get("tensors", {})
        logits = tensors.get("logits")
        if logits is None:
            probs = tensors.get("probs")
            if probs is not None:
                logits = jnp.log(jnp.clip(probs, 1e-9, 1.0))
        if logits is None:
            raise RuntimeError("ZymCTRL gradient merge requires logits/probs in ctx; missing tensors.")
        st = _load()
        import torch  # type: ignore
        import torch.utils.dlpack as tdl  # type: ignore
        mdl = st["model"]
        tok = st["tokenizer"]
        device = st["device"]
        aa_ids = st["aa_ids"]
        prefix_ids = st["prefix_ids"]
        suffix_ids = st["suffix_ids"]

        t_logits = tdl.from_dlpack(jdlpack.to_dlpack(logits))
        if not t_logits.requires_grad:
            t_logits = t_logits.clone().detach().requires_grad_(True)
        import torch.nn.functional as F  # type: ignore
        soft = F.softmax(t_logits / max(1e-6, float(temp)), dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=soft.size(-1)).float()
        ste = hard + (soft - soft.detach())
        emb = mdl.get_input_embeddings()(aa_ids)
        var = ste @ emb
        prefix = mdl.get_input_embeddings()(prefix_ids)
        suffix = mdl.get_input_embeddings()(suffix_ids)
        full = torch.cat([prefix, var, suffix], dim=0).unsqueeze(0)
        out = mdl(inputs_embeds=full)
        logits_full = out.logits
        var_token_ids = aa_ids[hard.argmax(dim=-1)]
        tgt = torch.cat([prefix_ids, var_token_ids, suffix_ids[:1]], dim=0).unsqueeze(0)
        loss = F.cross_entropy(logits_full[:, :-1, :].reshape(-1, logits_full.size(-1)), tgt[:, 1:].reshape(-1), reduction='mean')
        g2_t = torch.autograd.grad(loss, t_logits)[0].detach()
        target_dev = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]
        # Avoid JAX DLPack CPU backend path (not available in some GPU-only builds): use host -> device_put
        g2 = jax.device_put(jnp.asarray(g2_t.detach().to("cpu").numpy(), dtype=jnp.float32), device=target_dev)

        g1 = jnp.asarray(g, dtype=jnp.float32)
        # Record LM prior metric (log-likelihood) into ctx for trajectory logging
        try:
            metrics = (ctx or {}).setdefault("metrics", {})
            metrics["zymctrl_pll"] = float((-loss).detach().cpu().item())
        except Exception:
            pass
        def _unit(x):
            n = jnp.linalg.norm(jnp.reshape(x, (-1,))) + 1e-12
            return jnp.where(n > 0.0, x / n, x)
        g1_u = _unit(g1)
        g2_u = _unit(g2)
        if method == "pcgrad":
            merged = _pc(g1_u, g2_u, lam)
        elif method == "mgda":
            a = jnp.vdot(g1_u.reshape(-1), g1_u.reshape(-1)).real
            b = jnp.vdot(g1_u.reshape(-1), g2_u.reshape(-1)).real
            c = jnp.vdot(g2_u.reshape(-1), g2_u.reshape(-1)).real
            denom = a + c - 2.0 * b + 1e-12
            alpha = jnp.clip((c - b) / denom, 0.0, 1.0)
            merged = alpha * g1_u + (1.0 - alpha) * g2_u
        elif method == "scale":
            merged = g1_u + float(lam) * g2_u
        else:
            raise ValueError(f"Unknown grad_merge_method: {method}")
        flat = jnp.reshape(merged, (-1,))
        n = jnp.linalg.norm(flat) + 1e-12
        merged = jnp.where(n > 0.0, merged / n, merged)
        per_pos_norm = jnp.sqrt(jnp.sum(jnp.asarray(g, dtype=jnp.float32) ** 2, axis=-1))
        effL = jnp.sum(per_pos_norm > 0.0)
        scale = jnp.sqrt(jnp.maximum(effL, 1.0))
        merged = merged * scale
        return merged.astype(jnp.float32)

    return _grad

# Germinal IgLM gradient mixing; hacky atm, need to either reimplement IgLM in JAX and use the pcgrad optimizer in optimizers.py. Also need to implement UPCGrad
def iglm_pcgrad_merge(chain_token: str = "[HEAVY]", species: str = "[HUMAN]", temp: float = 0.6, vh_len: int | None = None, vl_len: int | None = None):
    """Grad transform: merge AF-M gradient with IgLM gradient using weighted PCGrad.

    - Reads schedule['iglm_scale'] to weight IgLM relative to AF-M (0.0-1.0 typical).
    - Requires ctx['tensors']['logits'] or ctx['tensors']['probs'] to compute IgLM grad.
    - Strict: raises if IgLM is unavailable or inputs are misconfigured.
    """
    # Use JAX-native IgLM converted via torch2jax
    from .iglm_jax import JAXIgLMGuidance
    _loaded = {"iglm": None, "jit": None}
    _cache: dict[str, _Any] = {"step": -1, "g2": None}

    def _ensure_iglm():
        if _loaded["iglm"] is None:
            ig = JAXIgLMGuidance(chain_token=chain_token, species=species)
            _loaded["iglm"] = ig
            _loaded["jit"] = ig.jit_grad()
        return _loaded["iglm"], _loaded["jit"]

    @jax.jit
    def _pcgrad_merge_jax(a_in: jnp.ndarray, b_in: jnp.ndarray, lam: float) -> jnp.ndarray:
        """PCGrad merge with projection when gradients conflict."""
        a = jnp.reshape(a_in, (-1,))
        b = jnp.reshape(b_in, (-1,))
        a_n = jnp.linalg.norm(a) + 1e-12
        b_n = jnp.linalg.norm(b) + 1e-12
        lam_f = jnp.asarray(lam, dtype=a.dtype)
        def when_zero():
            return jnp.reshape(b_in * lam_f, a_in.shape)
        def when_nonzero():
            a_hat = a / a_n
            b_hat = b / b_n
            dot = jnp.vdot(a_hat, b_hat).real
            b_proj = jnp.where(dot < 0.0, b_hat - dot * a_hat, b_hat)
            merged = a_hat + lam_f * b_proj
            return jnp.reshape(merged, a_in.shape)
        return jax.lax.cond(a_n <= 1e-11, when_zero, when_nonzero)

    def _full_merge_jax_impl(g1: jnp.ndarray, g2: jnp.ndarray, lam: float, method: str) -> jnp.ndarray:
        """Fully JIT'd gradient merge including normalization and post-processing.

        Args:
            g1: First gradient (AF gradient)
            g2: Second gradient (IgLM gradient)
            lam: Weight for second gradient
            method: Merge method ("pcgrad", "mgda", or "scale") - static argument
        """
        # Unit normalization
        g1_flat = jnp.reshape(g1, (-1,))
        g2_flat = jnp.reshape(g2, (-1,))
        n1 = jnp.linalg.norm(g1_flat) + 1e-12
        n2 = jnp.linalg.norm(g2_flat) + 1e-12
        g1_u = g1 / n1
        g2_u = g2 / n2

        # Merge based on method (use Python conditionals since method is static)
        if method == "pcgrad":
            merged = _pcgrad_merge_jax(g1_u, g2_u, lam)
        elif method == "mgda":
            g1_u_flat = jnp.reshape(g1_u, (-1,))
            g2_u_flat = jnp.reshape(g2_u, (-1,))
            a = jnp.vdot(g1_u_flat, g1_u_flat).real
            b = jnp.vdot(g1_u_flat, g2_u_flat).real
            c = jnp.vdot(g2_u_flat, g2_u_flat).real
            denom = a + c - 2.0 * b + 1e-12
            alpha = jnp.clip((c - b) / denom, 0.0, 1.0)
            merged = alpha * g1_u + (1.0 - alpha) * g2_u
        else:  # "scale"
            merged = g1_u + lam * g2_u

        # Post-merge normalization
        merged_flat = jnp.reshape(merged, (-1,))
        n_merged = jnp.linalg.norm(merged_flat) + 1e-12
        merged_norm = jnp.where(n_merged > 0.0, merged / n_merged, merged)

        # Effective length scaling
        per_pos_norm = jnp.sqrt(jnp.sum(g1 ** 2, axis=-1))
        effL = jnp.sum(per_pos_norm > 0.0)
        scale = jnp.sqrt(jnp.maximum(effL, 1.0))

        return (merged_norm * scale).astype(g1.dtype)

    # Apply JIT with static method argument
    _full_merge_jax = jax.jit(_full_merge_jax_impl, static_argnames=("method",))

    def _grad(g, ctx):
        sched = (ctx or {}).get("schedule", {})
        lam = float(sched.get("iglm_scale", 0.0))
        # If disabled or scheduled off this step, pass gradients through
        every = int(sched.get("iglm_every", 1) or 1)
        step = int(sched.get("phase_step", 0) or 0)
        if lam <= 0.0:
            return g
        method = str(sched.get("grad_merge_method", "pcgrad")).lower()
        tensors = (ctx or {}).get("tensors", {})
        logits = tensors.get("logits")
        if logits is None:
            probs = tensors.get("probs")
            if probs is not None:
                logits = jnp.log(jnp.clip(probs, 1e-9, 1.0))
        if logits is None:
            raise RuntimeError("IgLM gradient merge requires logits/probs in ctx; missing tensors.")
        # Reuse cached IgLM gradient when allowed by cadence
        use_cached = (every > 1) and (_cache.get("g2") is not None) and ((step % every) != 0)
        if use_cached:
            g2 = _cache["g2"]
        else:
            iglm, jitted = _ensure_iglm()
            val, g2 = jitted(jnp.asarray(logits, dtype=jnp.float32), jnp.asarray(temp, dtype=jnp.float32))
            g2 = jnp.asarray(g2, dtype=jnp.float32)
            _cache["g2"] = g2
            _cache["step"] = step

        g1 = jnp.asarray(g, dtype=jnp.float32)
        g2 = jnp.asarray(g2, dtype=jnp.float32)

        # Use fully JIT'd merge (includes normalization and post-processing)
        merged = _full_merge_jax(g1, g2, float(lam), method)

        return merged.astype(jnp.float32)

    return _grad


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
        # Keep as JAX scalar; analyzers can convert to Python for logging
        metrics[key] = val
        return p
    return _pre_probs


def freeze_grad_on_metric(threshold: float, key: str = "probs_max_mean"):
    """Grad transform that zeros updates once a ctx metric reaches a threshold.

    Typical usage with record_probs_max_mean to stop updates when logits/probs converge.
    """
    thr = float(threshold)
    def _grad(g, ctx):
        val_any = (ctx.get("metrics") or {}).get(key, 0.0)
        val_j = jnp.asarray(val_any)
        done = jnp.asarray(val_j >= thr)
        if done:
            metrics = ctx.setdefault("metrics", {})
            metrics["converged"] = True
            return jnp.zeros_like(g)
        return g
    return _grad

# Germinal convergence criterion
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
        # Keep JAX scalar in metrics; analyzers can convert when logging
        metrics[key] = val
        # Store raw values; consumer can convert when needed
        ctx["stop_metric"] = {"key": key, "threshold": threshold, "value": val, "met": val >= float(threshold)}
        return p

    return _pre_probs

# For Germinal
def framework_sequence_bias_on_logits(fr_positions: list[int], framework_sequence: str, bias: float = 10.0):
    """Post-logits transform: add a constant bias to logits at framework positions for the framework identities.

    Args:
        fr_positions: Binder indices (0-based within binder) that are framework (non-CDR) positions.
        framework_sequence: Full binder framework sequence (length >= max(fr_positions)+1). Only letters at fr_positions are used.
        bias: Additive bias value to add to the logit of the framework identity at each framework position.
    """
    vocab = "ARNDCQEGHILKMFPSTWYV"
    fr_positions_arr = jnp.array(fr_positions, dtype=jnp.int32)
    # Map framework identities at positions to token indices
    token_idx = jnp.array([vocab.index(framework_sequence[int(i)]) for i in fr_positions], dtype=jnp.int32)

    def _post_logits(x, ctx):
        # x: [L, 20]; add bias at specified positions to the token matching framework identity
        L = x.shape[0]
        updates = jnp.zeros_like(x)
        updates = updates.at[fr_positions_arr, token_idx].add(float(bias))
        return x + updates

    return _post_logits
