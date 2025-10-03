import os
import sys
from typing import Any, Dict, Tuple, Callable, cast

import jax
import jax.numpy as jnp


def _ensure_torch2jax_imported():
    from torch2jax import t2j, Torchish, auto_implements, implements  # type: ignore
    import torch  # type: ignore
    # Monkey-patch Torchish.size to accept optional dim like torch.Tensor.size
    def _size(self, dim=None):  # type: ignore[no-redef]
        if dim is None:
            return self.shape
        return self.shape[dim]
    setattr(Torchish, "size", _size)
    # Implement torch.addmm via JAX
    def _addmm(input, mat1, mat2, beta=1, alpha=1):
        y = mat1 @ mat2
        y = y * jnp.asarray(alpha, dtype=y.dtype)
        b = input * jnp.asarray(beta, dtype=y.dtype)
        # Broadcast bias across leading dimension if needed
        return y + b
    auto_implements(torch.addmm, _addmm)
    # Implement torch.split and Tensor.split
    import numpy as _np
    @implements(torch.split, Torchishify_output=False, Torchish_member=True)
    def _split(input, split_size_or_sections, dim=0):  # type: ignore[no-redef]
        x = input.value if isinstance(input, Torchish) else input
        axis = int(dim)
        if isinstance(split_size_or_sections, int):
            size = int(split_size_or_sections)
            n = x.shape[axis]
            # Build indices for jnp.split
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


class JAXIgLMGuidance:
    """JAX-native IgLM guidance via torch2jax.

    - Loads the Torch IgLM once, converts it to a JAX function, and captures its state dict as JAX arrays.
    - Exposes a pure-JAX loss and gradient wrt variable logits using inputs_embeds straight-through relaxation.
    """

    def __init__(self, chain_token: str = "[HEAVY]", species: str = "[HUMAN]"):
        self._ready = False
        self.chain_token = str(chain_token)
        self.species = str(species)
        self.jax_model: Callable[..., Any] | None = None
        self.state_dict: Dict[str, jax.Array] = {}
        self.embed_key: str | None = None
        self.aa_ids: jax.Array = jnp.zeros((0,), dtype=jnp.int32)
        self.chain_id: int = 0
        self.species_id: int = 0
        self.sep_id: int = 0
        self._build()

    def _build(self):
        # Import here to avoid hard dependency during module import
        import torch  # type: ignore
        from iglm import IgLM  # type: ignore

        t2j = _ensure_torch2jax_imported()

        # Initialize Torch IgLM (eval, frozen)
        class _Wrap(torch.nn.Module, IgLM):
            def __init__(self):
                torch.nn.Module.__init__(self)
                IgLM.__init__(self, model_name="IgLM")
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False

        mod = _Wrap()

        # Wrap to accept inputs_embeds positionally (torch2jax t2j_module passes args, not kwargs)
        class _InputsEmbedsWrapper(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module):
                super().__init__()
                self.model = inner
            def forward(self, inputs_embeds):  # type: ignore[override]
                # Call the underlying HF model stored on IgLM as `.model`
                return self.model.model(inputs_embeds=inputs_embeds)

        wrap = _InputsEmbedsWrapper(mod)

        # Convert to JAX callable and capture state_dict as JAX arrays
        self.jax_model = t2j(wrap)
        # Build wrapper-aligned state_dict (prefix underlying keys with 'model.')
        sd_torch = mod.state_dict()
        wrap_keys = list(wrap.state_dict().keys())
        remapped: Dict[str, Any] = {}
        for k in wrap_keys:
            if k.startswith("model."):
                base = k[len("model."):]
                if base not in sd_torch:
                    raise KeyError(f"Missing base key in IgLM state_dict: {base}")
                remapped[k] = sd_torch[base]
            else:
                if k not in sd_torch:
                    raise KeyError(f"Missing key in IgLM state_dict: {k}")
                remapped[k] = sd_torch[k]
        self.state_dict = {k: t2j(v) for k, v in remapped.items()}  # type: ignore

        # Tokenizer-derived ids
        tok = mod.tokenizer
        self.aa_ids = jnp.asarray([tok.convert_tokens_to_ids(a) for a in list("ARNDCQEGHILKMFPSTWYV")], dtype=jnp.int32)
        self.chain_id = int(tok.convert_tokens_to_ids(self.chain_token))
        self.species_id = int(tok.convert_tokens_to_ids(self.species))
        self.sep_id = int(tok.sep_token_id)

        # Heuristically find input embedding weight key from state_dict
        # Prefer common HF names; fall back to first 2D weight with vocab dimension matching tokenizer length
        vocab_sizes = set()
        if hasattr(tok, "vocab_size"):
            vocab_sizes.add(int(getattr(tok, "vocab_size")))
        # Many tokenizers implement __len__
        if hasattr(tok, "__len__"):
            vocab_sizes.add(int(len(tok)))

        candidate_keys: list[Tuple[str, Tuple[int, ...]]] = []
        for k, v in self.state_dict.items():
            if hasattr(v, "shape") and len(v.shape) == 2:
                candidate_keys.append((k, tuple(v.shape)))

        # Preferred keys
        preferred = [
            ".*.embeddings.word_embeddings.weight",
            ".*wte.weight",
            ".*get_input_embeddings.weight",
        ]

        def _matches(pref: str, key: str) -> bool:
            # crude glob-like match for a few common patterns
            parts = pref.split(".*")
            pos = 0
            for part in parts:
                if not part:
                    continue
                i = key.find(part, pos)
                if i < 0:
                    return False
                pos = i + len(part)
            return True

        chosen: str | None = None
        for pat in preferred:
            for (k, shape) in candidate_keys:
                if _matches(pat, k):
                    if (len(shape) == 2) and ((not vocab_sizes) or (shape[0] in vocab_sizes)):
                        chosen = k
                        break
            if chosen is not None:
                break
        if chosen is None:
            # Fallback: first 2D param with vocab-like first dim
            for (k, shape) in candidate_keys:
                if (len(shape) == 2) and ((not vocab_sizes) or (shape[0] in vocab_sizes)):
                    chosen = k
                    break
        if chosen is None:
            raise RuntimeError("Could not locate input embedding weight in IgLM state_dict")

        self.embed_key = chosen
        self._ready = True

    def _embedding_weight(self) -> jax.Array:
        assert self.embed_key is not None
        return self.state_dict[self.embed_key]

    def _build_inputs_embeds(self, logits: jax.Array, temp: jax.Array | float) -> jax.Array:
        """Construct inputs_embeds [1, L+3, D] from relaxed logits and cached embeddings.

        Sequence: [chain_token, species_token] + variable_region + [sep_token]
        """
        W = self._embedding_weight()  # [V, D]
        # aa embedding matrix: select 20 amino acids
        aa_emb = W[self.aa_ids]  # [20, D]
        # Straight-through relaxed embedding for variable region
        t = jnp.asarray(temp, dtype=logits.dtype)
        soft = jax.nn.softmax(logits / jnp.maximum(jnp.asarray(1e-6, dtype=logits.dtype), t), axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(soft, axis=-1), logits.shape[-1]).astype(soft.dtype)
        ste = hard + (soft - jax.lax.stop_gradient(soft))
        var = ste @ aa_emb  # [L, D]
        # Prefix/suffix embeddings
        prefix_ids = jnp.asarray([self.chain_id, self.species_id], dtype=jnp.int32)
        prefix = W[prefix_ids]  # [2, D]
        suffix = W[jnp.asarray([self.sep_id], dtype=jnp.int32)]  # [1, D]
        full = jnp.concatenate([prefix, var, suffix], axis=0)  # [L+3, D]
        return full[None, ...]  # [1, L+3, D]

    def _build_targets(self, logits: jax.Array) -> jax.Array:
        # Convert relaxed hard assignment to token ids for_targets
        idx = jnp.argmax(logits, axis=-1)  # [L]
        aa_ids = cast(jax.Array, self.aa_ids)
        var_token_ids = aa_ids[idx]
        tgt = jnp.concatenate(
            [
                jnp.asarray([self.chain_id, self.species_id], dtype=jnp.int32),
                var_token_ids.astype(jnp.int32),
                jnp.asarray([self.sep_id], dtype=jnp.int32),
            ],
            axis=0,
        )  # [L+3]
        return tgt[None, ...]  # [1, L+3]

    def _iglm_ce_loss(self, logits: jax.Array, *, temp: jax.Array | float) -> jax.Array:
        # Build inputs_embeds and targets, call JAX IgLM, compute CE over next-token
        inputs_embeds = self._build_inputs_embeds(logits, temp)
        jax_model = cast(Callable[..., Any], self.jax_model)
        # torch2jax t2j_module expects positional args for module inputs; pass inputs_embeds positionally
        out = jax_model(inputs_embeds, state_dict=self.state_dict)
        # Expect HF-style output with "logits" key; otherwise try attribute or positional
        if isinstance(out, dict) and ("logits" in out):
            logits_full = out["logits"]
        elif hasattr(out, "logits"):
            logits_full = getattr(out, "logits")
        else:
            leaves, _ = jax.tree_util.tree_flatten(out)
            cand = [x for x in leaves if hasattr(x, "shape") and len(x.shape) == 3]
            if not cand:
                raise RuntimeError("IgLM forward did not return a logits tensor")
            logits_full = cand[0]

        # Shifted CE: predict next token
        tgt = self._build_targets(logits)  # [1, L+3]
        pred = logits_full[:, :-1, :]  # [1, L+2, V]
        y = tgt[:, 1:]  # [1, L+2]
        logp = jax.nn.log_softmax(pred, axis=-1)
        nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
        return jnp.mean(nll)

    def grad_from_logits(self, logits: jax.Array, *, temp: float) -> Tuple[jax.Array, float]:
        """Return (grad, neg_log_likelihood) fully in JAX."""
        loss_fn = lambda zz: self._iglm_ce_loss(zz, temp=temp)
        val, g = jax.value_and_grad(loss_fn)(logits)
        return g, float(val)

    def jit_grad(self):
        def _f(zz, t):
            tt = jnp.asarray(t, dtype=zz.dtype)
            return jax.value_and_grad(lambda z: self._iglm_ce_loss(z, temp=tt))(zz)
        return jax.jit(_f)


