from jax import vmap, numpy as jnp
from jaxtyping import Array, Bool, Float
import equinox as eqx

from ..util import fold_in


def rotary_embedding(
    x: Float[Array, "len heads qk_dim"], base_exponent: int = 10000
) -> Float[Array, "len heads qk_dim"]:
    dim = x.shape[-1]
    assert dim % 2 == 0
    # Compute the per-dimension frequencies
    exponents = jnp.arange(0, dim, 2, dtype=x.dtype)
    inv_freq = 1.0 / (base_exponent ** (exponents / dim))

    # Compute the per element phase (to pass into sin and cos)
    t = jnp.arange(x.shape[0], dtype=x.dtype)
    phase = jnp.einsum("i,j->ij", t, inv_freq)
    phase = jnp.tile(phase, reps=(1, 2))[:, None, :]

    x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)

    return x


def rotate_half(x: Float[Array, "len heads dim"]) -> Float[Array, "len heads dim"]:
    "Obtain the rotated counterpart of each feature"
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


class RoPEMultiHeadDotProductAttention(eqx.Module):
    num_heads: int
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    o: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        qkv_features: int,
        out_features: int,
        q_dim: int,
        kv_dim: int,
        *,
        key
    ):
        assert (
            qkv_features % num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        # head_dim = qkv_features // num_heads
        self.num_heads = num_heads
        self.q = eqx.nn.Linear(q_dim, qkv_features, key=fold_in(key, "q"))
        self.k = eqx.nn.Linear(kv_dim, qkv_features, key=fold_in(key, "k"))
        self.v = eqx.nn.Linear(kv_dim, qkv_features, key=fold_in(key, "v"))
        self.o = eqx.nn.Linear(qkv_features, out_features, key=fold_in(key, "o"))

    def __call__(
        self,
        inputs: Float[Array, "len dim"],
        mask: Bool[Array, "len len"] | None = None,
    ) -> Float[Array, "len outdim"]:
        return self.cross_attention(inputs, inputs, mask)

    def cross_attention(
        self,
        inputs_q: Float[Array, "q_len q_dim"],
        inputs_kv: Float[Array, "kv_len kv_dim"],
        mask: Bool[Array, "q_len kv_len"] | None = None,
    ) -> Float[Array, "q_len outdim"]:

        # Pt. 1: Compute the query, key, and value vectors
        q = vmap(self.q)(inputs_q).reshape(inputs_q.shape[0], self.num_heads, -1)
        k = vmap(self.k)(inputs_kv).reshape(inputs_kv.shape[0], self.num_heads, -1)
        v = vmap(self.v)(inputs_kv).reshape(inputs_kv.shape[0], self.num_heads, -1)
        # dims: [length, n_heads, n_features_per_head]

        # Pt. 2: Apply the rotary embedding to query and key.
        q, k = rotary_embedding(q), rotary_embedding(k)

        # Pt. 3: Compute the attention weights.
        # TODO: Might be a mismatch between eqx and flax here!
        attn_weights = vmap(
            lambda q, k: eqx.nn._attention.dot_product_attention_weights(
                q, k, mask=mask
            ),
            in_axes=(1, 1),
        )(q, k)
        x = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v)
        # # Pt. 4: Apply output map to last two dims of x.
        return vmap(lambda a: self.o(a.reshape(-1)))(x), attn_weights
