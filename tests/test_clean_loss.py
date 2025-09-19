import numpy as np

import jax
import jax.numpy as jnp

from mosaic.losses.clean import CleanHead, CleanCosineSimilarityLoss
from mosaic.common import TOKENS


def _dummy_embedder():
    rng = np.random.default_rng(0)
    table = {aa: rng.standard_normal(1280).astype(np.float32) for aa in TOKENS}

    def embed(seq: str) -> np.ndarray:
        vecs = np.stack([table[aa] for aa in seq], axis=0)
        return vecs.mean(axis=0)

    stacked = np.stack([table[aa] for aa in TOKENS], axis=0)
    return embed, stacked


def test_clean_loss_returns_gradients_in_differentiable_mode():
    embed_fn, stacked = _dummy_embedder()
    head = CleanHead(key=jax.random.key(0))
    target = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    table = jnp.asarray(stacked)

    def soft_embed(probs: jnp.ndarray) -> jnp.ndarray:
        per_pos = probs @ table  # (L, 1280)
        return per_pos.mean(axis=0)

    loss = CleanCosineSimilarityLoss(
        clean_head=head,
        target_embedding=target,
        embed_fn=embed_fn,
        differentiable=True,
        soft_embed_fn=soft_embed,
    )

    vocab = len(TOKENS)
    logits = jax.random.normal(jax.random.key(1), shape=(8, vocab))
    probs = jax.nn.softmax(logits, axis=-1)

    value, aux = loss(probs, key=jax.random.key(2))
    assert jnp.isscalar(value), "differentiable path should return JAX scalar"
    assert loss.name in aux

    grad = jax.grad(lambda p: loss(p, key=jax.random.key(2))[0])(probs)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.linalg.norm(grad) > 0.0


def test_clean_loss_rl_mode_keeps_float_output():
    embed_fn, _ = _dummy_embedder()
    head = CleanHead(key=jax.random.key(3))
    target = np.zeros(128, dtype=np.float32)
    loss = CleanCosineSimilarityLoss(
        clean_head=head,
        target_embedding=target,
        embed_fn=embed_fn,
        differentiable=False,
    )

    probs = jnp.full((4, len(TOKENS)), 1.0 / len(TOKENS), dtype=jnp.float32)
    value, aux = loss(probs, key=jax.random.key(4))
    assert isinstance(value, float)
    assert loss.name in aux
