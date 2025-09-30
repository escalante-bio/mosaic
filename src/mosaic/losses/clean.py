from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LossTerm, TOKENS


class CleanHead(eqx.Module):
    hidden_dim: int
    out_dim: int
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(self, hidden_dim: int = 512, out_dim: int = 128, key=None):
        if key is None:
            key = jax.random.key(0)
        k1, k2, k3 = jax.random.split(key, 3)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.fc1 = eqx.nn.Linear(1280, hidden_dim, key=k1)
        self.ln1 = eqx.nn.LayerNorm(shape=hidden_dim)
        self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.ln2 = eqx.nn.LayerNorm(shape=hidden_dim)
        self.fc3 = eqx.nn.Linear(hidden_dim, out_dim, key=k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.fc1(x)
        h = self.ln1(h)
        h = jax.nn.relu(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = jax.nn.relu(h)
        z = self.fc3(h)
        return z


def _torch_tensor_to_jnp(t) -> jax.Array:
    """Utility to convert individual torch tensors to ``float32`` JAX arrays."""

    try:  # Lazy import avoids hard dependency for users that already have JAX weights
        import torch
    except ImportError as exc:  # pragma: no cover - defensive, eager import done in caller
        raise ImportError("torch is required to convert CLEAN weights") from exc

    if not isinstance(t, torch.Tensor):  # pragma: no cover - developer misuse guard
        raise TypeError(f"Expected torch.Tensor, got {type(t)!r}")
    return jnp.asarray(t.detach().cpu().numpy(), dtype=jnp.float32)


def load_clean_head_from_torch(
    weights_path: str | Path,
    *,
    hidden_dim: int = 512,
    out_dim: int = 128,
    key=None,
) -> CleanHead:
    """Instantiate :class:`CleanHead` from a PyTorch LayerNormNet checkpoint.

    The reference implementation that ships with CLEAN stores weights using the
    architecture defined in ``CLEAN/app/src/CLEAN/model.py``. This helper mirrors
    that layout and transfers the parameters into the equivalent Equinox module so
    downstream JAX code can reproduce the original behaviour exactly.

    Parameters
    ----------
    weights_path
        Path to the ``.pth`` checkpoint produced by the PyTorch code.
    hidden_dim, out_dim
        Dimensions used when the original model was trained (CLEAN defaults are
        512 and 128 respectively). If custom values were used, pass them here so
        the Equinox module matches.
    key
        Optional PRNG key to seed the initial Equinox module. The parameters are
        immediately overwritten by the checkpoint, so the key only controls the
        initial allocation.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - import tested through usage
        raise ImportError(
            "torch is required to load CLEAN weights. Install it or provide a JAX-native checkpoint."
        ) from exc

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"CLEAN weight file not found: {path}")

    state = torch.load(path, map_location="cpu")
    expected: Iterable[str] = (
        "fc1.weight",
        "fc1.bias",
        "ln1.weight",
        "ln1.bias",
        "fc2.weight",
        "fc2.bias",
        "ln2.weight",
        "ln2.bias",
        "fc3.weight",
        "fc3.bias",
    )
    missing = [k for k in expected if k not in state]
    if missing:  # pragma: no cover - defensive guard for unexpected checkpoints
        raise KeyError(f"CLEAN checkpoint missing parameters: {missing}")

    head = CleanHead(hidden_dim=hidden_dim, out_dim=out_dim, key=key)

    # Linear layers
    head = eqx.tree_at(lambda h: h.fc1.weight, head, _torch_tensor_to_jnp(state["fc1.weight"]))
    head = eqx.tree_at(lambda h: h.fc1.bias, head, _torch_tensor_to_jnp(state["fc1.bias"]))
    head = eqx.tree_at(lambda h: h.fc2.weight, head, _torch_tensor_to_jnp(state["fc2.weight"]))
    head = eqx.tree_at(lambda h: h.fc2.bias, head, _torch_tensor_to_jnp(state["fc2.bias"]))
    head = eqx.tree_at(lambda h: h.fc3.weight, head, _torch_tensor_to_jnp(state["fc3.weight"]))
    head = eqx.tree_at(lambda h: h.fc3.bias, head, _torch_tensor_to_jnp(state["fc3.bias"]))

    # LayerNorm parameters to keep parity with the PyTorch model
    head = eqx.tree_at(lambda h: h.ln1.weight, head, _torch_tensor_to_jnp(state["ln1.weight"]))
    head = eqx.tree_at(lambda h: h.ln1.bias, head, _torch_tensor_to_jnp(state["ln1.bias"]))
    head = eqx.tree_at(lambda h: h.ln2.weight, head, _torch_tensor_to_jnp(state["ln2.weight"]))
    head = eqx.tree_at(lambda h: h.ln2.bias, head, _torch_tensor_to_jnp(state["ln2.bias"]))

    return head


class CleanCosineSimilarityLoss(LossTerm):
    """Cosine-similarity loss against a CLEAN target embedding.

    The loss mirrors the RL-style reward used in the reference CLEAN project:
    logits are converted to a discrete sequence via ``argmax`` and scored against
    a fixed target embedding. This makes the objective non-differentiable with
    respect to the logits, so it should only be used inside policy-gradient or
    evolutionary loops.

    To use the loss inside gradient-based optimizers, pass ``differentiable=True``
    together with ``soft_embed_fn``—a callable that maps the probability tensor to
    a differentiable 1280-d CLEAN embedding. In that mode, the discrete
    ``embed_fn`` is still used for logging but gradients flow exclusively through
    the soft embedding pathway.
    """

    clean_head: CleanHead
    target_embedding: jax.Array
    embed_fn: Callable[[str], np.ndarray] = eqx.static_field()
    soft_embed_fn: Callable[[jax.Array], jax.Array] | None = eqx.static_field()
    name: str = eqx.static_field()
    differentiable: bool = eqx.static_field()
    vocab: str = eqx.static_field()

    def __init__(
        self,
        *,
        clean_head: CleanHead,
        target_embedding: np.ndarray,
        embed_fn: Callable[[str], np.ndarray],
        name: str = "clean",
        differentiable: bool = False,
        vocab: str = TOKENS,
        soft_embed_fn: Callable[[jax.Array], jax.Array] | None = None,
    ):
        self.clean_head = clean_head
        self.target_embedding = jnp.asarray(target_embedding, dtype=jnp.float32)
        self.embed_fn = embed_fn
        self.name = name
        self.differentiable = bool(differentiable)
        self.vocab = vocab
        self.soft_embed_fn = soft_embed_fn

    def __call__(self, probs: jax.Array, *, key) -> tuple[float, dict]:
        probs = jnp.asarray(probs, dtype=jnp.float32)
        seq: str | None = None
        if self.differentiable:
            if self.soft_embed_fn is None:
                raise RuntimeError("differentiable=True requires soft_embed_fn")
            x = self.soft_embed_fn(probs)
        else:
            probs_np = np.asarray(probs)
            idx_np = probs_np.argmax(axis=-1)
            seq = "".join([self.vocab[int(i)] for i in idx_np])
            emb1280 = self.embed_fn(seq)
            x = jnp.asarray(emb1280, dtype=jnp.float32)
        z = self.clean_head(x)
        tgt = self.target_embedding
        cos = jnp.vdot(z, tgt) / (jnp.linalg.norm(z) * jnp.linalg.norm(tgt) + 1e-8)
        value = -cos  # maximize similarity -> minimize negative
        cos_for_aux: float | jax.Array
        if self.differentiable:
            cos_for_aux = cos
        else:
            cos_for_aux = float(cos)

        aux = {self.name: {"cosine": cos_for_aux, "sequence": seq}}

        if not self.differentiable:
            return float(value), aux

        return value, aux
