import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable

from mosaic.common import LossTerm


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


class CleanCosineSimilarityLoss(LossTerm):
    clean_head: CleanHead
    target_embedding: jax.Array
    embed_fn: Callable[[str], np.ndarray] = eqx.static_field()
    name: str = eqx.static_field()

    def __init__(self, *, clean_head: CleanHead, target_embedding: np.ndarray, embed_fn: Callable[[str], np.ndarray], name: str = "clean"):
        self.clean_head = clean_head
        self.target_embedding = jnp.asarray(target_embedding, dtype=jnp.float32)
        self.embed_fn = embed_fn
        self.name = name

    def __call__(self, probs: jax.Array, *, key) -> tuple[float, dict]:
        # Convert probs to sequence via argmax (non-differentiable; intended for RL usage)
        vocab = "ARNDCQEGHILKMFPSTWYV"
        idx = jnp.argmax(probs, axis=-1)
        seq = "".join([vocab[int(i)] for i in np.asarray(idx)])
        emb1280 = self.embed_fn(seq)
        x = jnp.asarray(emb1280, dtype=jnp.float32)
        z = self.clean_head(x)
        tgt = self.target_embedding
        # cosine similarity
        cos = jnp.vdot(z, tgt) / (jnp.linalg.norm(z) * jnp.linalg.norm(tgt) + 1e-8)
        value = -cos  # maximize similarity -> minimize negative
        aux = {self.name: {"cosine": float(cos), "sequence": seq}}
        return float(value), aux


