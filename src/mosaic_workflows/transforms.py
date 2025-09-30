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


# Germinal IgLM gradient mixing; hacky atm, need to either reimplement IgLM in JAX and use the pcgrad optimizer in optimizers.py. Also need to implement UPCGrad
def iglm_pcgrad_merge(chain_token: str = "[HEAVY]", species: str = "[HUMAN]", temp: float = 0.6, vh_len: int | None = None, vl_len: int | None = None):
    """Grad transform: merge AF-M gradient with IgLM gradient using weighted PCGrad.

    - Reads schedule['iglm_scale'] to weight IgLM relative to AF-M (0.0-1.0 typical).
    - Requires ctx['tensors']['logits'] or ctx['tensors']['probs'] to compute IgLM grad.
    - Strict: raises if IgLM is unavailable or inputs are misconfigured.
    """
    _loaded = {"model": None}
    _cache: dict[str, _Any] = {"step": -1, "g2": None}

    def _load_model():
        if _loaded["model"] is not None:
            return _loaded["model"]
        import torch  # type: ignore
        from iglm import IgLM  # type: ignore
        class _Wrapper(torch.nn.Module, IgLM):  # type: ignore
            def __init__(self):
                torch.nn.Module.__init__(self)
                IgLM.__init__(self, model_name="IgLM")
                self.model.to(self.device)
                for p in self.model.parameters():
                    p.requires_grad = False
                self.chain_id = self.tokenizer.convert_tokens_to_ids(chain_token)
                self.species_id = self.tokenizer.convert_tokens_to_ids(species)
                self.suffix_id = self.tokenizer.sep_token_id
                self.amino_acids = list("ARNDCQEGHILKMFPSTWYV")
                self.aa_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(a) for a in self.amino_acids], device=self.device)

            def grad_from_torch_logits(self, logits):
                import torch.nn.functional as F
                # logits is a torch.Tensor on the correct device
                if not logits.requires_grad:
                    logits = logits.clone().detach().requires_grad_(True)
                soft = F.softmax(logits / max(1e-6, float(temp)), dim=-1)
                hard = F.one_hot(soft.argmax(dim=-1), num_classes=soft.size(-1)).float()
                ste = hard + (soft - soft.detach())
                emb = self.model.get_input_embeddings()(self.aa_ids)
                var = ste @ emb
                prefix_ids = torch.tensor([self.chain_id, self.species_id], device=self.device)
                prefix = self.model.get_input_embeddings()(prefix_ids)
                suffix = self.model.get_input_embeddings()(torch.tensor([self.suffix_id], device=self.device))
                full = torch.cat([prefix, var, suffix], dim=0).unsqueeze(0)
                out = self.model(inputs_embeds=full)
                logits_full = out.logits
                var_token_ids = self.aa_ids[hard.argmax(dim=-1)]
                tgt = torch.cat([prefix_ids, var_token_ids, torch.tensor([self.suffix_id], device=self.device)], dim=0).unsqueeze(0)
                loss = F.cross_entropy(logits_full[:, :-1, :].reshape(-1, logits_full.size(-1)), tgt[:, 1:].reshape(-1), reduction='mean')
                g = torch.autograd.grad(loss, logits)[0]
                return g.detach(), float(-loss.detach().cpu().item())

        _loaded["model"] = _Wrapper()
        return _loaded["model"]

    def _pcgrad_merge_jax(a_in: jnp.ndarray, b_in: jnp.ndarray, lam: float) -> jnp.ndarray:
        a = jnp.reshape(a_in, (-1,))
        b = jnp.reshape(b_in, (-1,))
        a_n = jnp.linalg.norm(a) + 1e-12
        b_n = jnp.linalg.norm(b) + 1e-12
        def when_zero():
            return jnp.reshape(b_in * lam, a_in.shape)
        def when_nonzero():
            a_hat = a / a_n
            b_hat = b / b_n
            dot = jnp.vdot(a_hat, b_hat).real
            b_proj = jnp.where(dot < 0.0, b_hat - dot * a_hat, b_hat)
            merged = a_hat + float(lam) * b_proj
            return jnp.reshape(merged, a_in.shape)
        return jax.lax.cond(a_n <= 1e-11, when_zero, when_nonzero)

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
            model = _load_model()
            if model is None:
                raise ImportError("IgLM model failed to load; ensure iglm is installed and available.")
            # JAX -> Torch via DLPack (device-zero-copy)
            import torch  # type: ignore
            import torch.utils.dlpack as tdl  # type: ignore
            t_logits = tdl.from_dlpack(jdlpack.to_dlpack(logits))
            g2_t, _ = model.grad_from_torch_logits(t_logits)
            # Prefer Torch->JAX zero-copy via DLPack; fall back to host copy if unsupported
            try:
                from torch.utils import dlpack as _tdl  # type: ignore
                g2 = jdlpack.from_dlpack(_tdl.to_dlpack(g2_t))
                g2 = jnp.asarray(g2, dtype=jnp.float32)
            except Exception:
                target_device = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]
                g2_np = g2_t.detach().to("cpu").numpy()
                g2 = jax.device_put(jnp.asarray(g2_np, dtype=jnp.float32), device=target_device)
            # cache for reuse until next recompute
            _cache["g2"] = g2
            _cache["step"] = step
   
        g1 = jnp.asarray(g, dtype=jnp.float32)
        g2 = jnp.asarray(g2, dtype=jnp.float32)

        # normalize both first
        def _unit(x):
            n = jnp.linalg.norm(jnp.reshape(x, (-1,))) + 1e-12
            return jnp.where(n > 0.0, x / n, x)
        g1_u = _unit(g1)
        g2_u = _unit(g2)

        if method == "pcgrad":
            merged = _pcgrad_merge_jax(g1_u, g2_u, lam)
        elif method == "mgda":
            # Solve for alpha in [0,1] minimizing || alpha g1 + (1-alpha) g2 ||^2
            a = jnp.vdot(g1_u.reshape(-1), g1_u.reshape(-1)).real
            b = jnp.vdot(g1_u.reshape(-1), g2_u.reshape(-1)).real
            c = jnp.vdot(g2_u.reshape(-1), g2_u.reshape(-1)).real
            denom = a + c - 2.0 * b + 1e-12
            alpha = jnp.clip((c - b) / denom, 0.0, 1.0)
            merged = alpha * g1_u + (1.0 - alpha) * g2_u
        elif method == "scale":
            # Weighted sum after unit norm
            merged = g1_u + float(lam) * g2_u
        else:
            raise ValueError(f"Unknown grad_merge_method: {method}")

        # Post-merge normalization and effective-length scaling (parity with Germinal) in JAX
        flat = jnp.reshape(merged, (-1,))
        n = jnp.linalg.norm(flat) + 1e-12
        merged = jnp.where(n > 0.0, merged / n, merged)
        per_pos_norm = jnp.sqrt(jnp.sum(jnp.asarray(g, dtype=jnp.float32) ** 2, axis=-1))
        effL = jnp.sum(per_pos_norm > 0.0)
        scale = jnp.sqrt(jnp.maximum(effL, 1.0))
        merged = merged * scale

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
        metrics[key] = float(val)
        ctx["stop_metric"] = {"key": key, "threshold": float(threshold), "value": float(val), "met": bool(val >= float(threshold))}
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
