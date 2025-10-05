import jax
import jax.numpy as jnp
from typing import Any, Dict, Callable, Tuple

# Minimal LossTerm import to avoid heavy dependencies at import time
try:
    from mosaic.common import LossTerm  # type: ignore
except Exception:  # pragma: no cover
    LossTerm = object  # type: ignore


def _ensure_torch2jax_imported():
    from torch2jax import t2j, Torchish, auto_implements, implements  # type: ignore
    import torch  # type: ignore

    def _size(self, dim=None):  # type: ignore[no-redef]
        if dim is None:
            return self.shape
        return self.shape[dim]
    setattr(Torchish, "size", _size)

    import numpy as _np
    @implements(torch.split, Torchishify_output=False, Torchish_member=True)  # type: ignore
    def _split(input, split_size_or_sections, dim=0):
        x = input.value if isinstance(input, Torchish) else input
        axis = int(dim)
        if isinstance(split_size_or_sections, int):
            size = int(split_size_or_sections)
            n = x.shape[axis]
            sections = [size] * (n // size) + ([n % size] if (n % size) else [])
            if len(sections) == 0:
                return tuple()
            if len(sections) == 1:
                return (Torchish(x),)
            idx = _np.cumsum(_np.asarray(sections[:-1])).tolist()
            parts = jnp.split(x, idx, axis=axis)
        else:
            sizes = [int(s) for s in split_size_or_sections]
            if len(sizes) == 0:
                return tuple()
            if len(sizes) == 1:
                return (Torchish(x),)
            idx = _np.cumsum(_np.asarray(sizes[:-1])).tolist()
            parts = jnp.split(x, idx, axis=axis)
        return tuple(Torchish(p) for p in parts)
    return t2j


class JAXZymCTRL:
    """JAX wrapper for ZymCTRL GPT2 LM via torch2jax.

    Builds a JAX-callable forward() that accepts inputs_embeds and returns logits.
    """

    def __init__(self, ec_label: str):
        self.ec_label = str(ec_label)
        self.jax_model: Callable[..., Any] | None = None
        self.state_dict: Dict[str, jax.Array] = {}
        self.embed_key: str | None = None
        self.aa_ids: jax.Array = jnp.zeros((0,), dtype=jnp.int32)
        self.prefix_ids: jax.Array = jnp.zeros((0,), dtype=jnp.int32)
        self.suffix_ids: jax.Array = jnp.zeros((0,), dtype=jnp.int32)
        self._build()

    def _build(self):
        import torch  # type: ignore
        from transformers import GPT2LMHeadModel, AutoTokenizer  # type: ignore
        t2j = _ensure_torch2jax_imported()

        tok = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL")
        mdl = GPT2LMHeadModel.from_pretrained("AI4PD/ZymCTRL")
        mdl.eval()
        for p in mdl.parameters():
            p.requires_grad = False

        class _Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, inputs_embeds):  # type: ignore[override]
                return self.inner(inputs_embeds=inputs_embeds)

        wrap = _Wrap(mdl)
        self.jax_model = t2j(wrap)
        # Remap state dict keys: mirror wrap.state_dict structure
        sd_torch = mdl.state_dict()
        wrap_keys = list(wrap.state_dict().keys())
        remapped: Dict[str, Any] = {}
        for k in wrap_keys:
            if k.startswith("inner."):
                base = k[len("inner."):]
                remapped[k] = sd_torch[base]
            else:
                remapped[k] = sd_torch[k]
        self.state_dict = {k: t2j(v) for k, v in remapped.items()}  # type: ignore

        # Precompute token ids
        aa = list("ARNDCQEGHILKMFPSTWYV")
        self.aa_ids = jnp.asarray([tok.convert_tokens_to_ids(a) for a in aa], dtype=jnp.int32)
        self.prefix_ids = jnp.asarray(tok.encode(self.ec_label + "<sep>", add_special_tokens=False), dtype=jnp.int32)
        suf = tok.encode("<|endoftext|>", add_special_tokens=False)
        self.suffix_ids = jnp.asarray(suf[:1] if len(suf) > 0 else suf, dtype=jnp.int32)

        # Find input embedding weight key heuristically
        cand = [(k, v.shape) for (k, v) in self.state_dict.items() if hasattr(v, "shape") and len(v.shape) == 2]
        chosen: str | None = None
        for (k, shape) in cand:
            if k.endswith(".wte.weight") or "embeddings.word_embeddings.weight" in k:
                chosen = k
                break
        if chosen is None and len(cand) > 0:
            chosen = cand[0][0]
        if chosen is None:
            raise RuntimeError("Could not locate input embedding weight in ZymCTRL state_dict")
        self.embed_key = chosen

    def _embedding_weight(self) -> jax.Array:
        assert self.embed_key is not None
        return self.state_dict[self.embed_key]

    def _build_inputs_embeds(self, logits: jax.Array, temp: float) -> jax.Array:
        W = self._embedding_weight()  # [V, D]
        aa_emb = W[self.aa_ids]      # [20, D]
        t = jnp.asarray(temp, dtype=logits.dtype)
        soft = jax.nn.softmax(logits / jnp.maximum(jnp.asarray(1e-6, dtype=logits.dtype), t), axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(soft, axis=-1), logits.shape[-1]).astype(soft.dtype)
        ste = hard + (soft - jax.lax.stop_gradient(soft))
        var = ste @ aa_emb  # [L, D]
        prefix = W[self.prefix_ids]  # [P, D]
        suffix = W[self.suffix_ids]  # [S, D]
        full = jnp.concatenate([prefix, var, suffix], axis=0)  # [L+P+S, D]
        return full[None, ...]

    def _build_targets(self, logits: jax.Array) -> jax.Array:
        idx = jnp.argmax(logits, axis=-1)
        var_token_ids = self.aa_ids[idx]
        tgt = jnp.concatenate([self.prefix_ids, var_token_ids.astype(jnp.int32), self.suffix_ids], axis=0)
        return tgt[None, ...]

    def ce_loss_from_logits(self, logits: jax.Array, *, temp: float) -> jax.Array:
        inputs_embeds = self._build_inputs_embeds(logits, temp)
        jax_model = self.jax_model  # type: ignore
        out = jax_model(inputs_embeds, state_dict=self.state_dict)
        if isinstance(out, dict) and ("logits" in out):
            logits_full = out["logits"]
        elif hasattr(out, "logits"):
            logits_full = getattr(out, "logits")
        else:
            leaves, _ = jax.tree_util.tree_flatten(out)
            cand = [x for x in leaves if hasattr(x, "shape") and len(x.shape) == 3]
            if not cand:
                raise RuntimeError("ZymCTRL forward did not return a logits tensor")
            logits_full = cand[0]
        tgt = self._build_targets(logits)
        pred = logits_full[:, :-1, :]
        y = tgt[:, 1:]
        logp = jax.nn.log_softmax(pred, axis=-1)
        nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
        return jnp.mean(nll)


class ZymCTRLLoss(LossTerm):
    def __init__(self, ec_label: str, temp: float = 1.0):
        self._zym = JAXZymCTRL(ec_label=ec_label)
        self._temp = float(temp)

    def __call__(self, p, *, key): 
        logits = jnp.log(jnp.clip(p, 1e-9, 1.0))
        val = self._zym.ce_loss_from_logits(logits, temp=self._temp)
        return val, {"zymctrl_nll": val}


