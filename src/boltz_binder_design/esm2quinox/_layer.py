import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray


class TransformerLayer(eqx.Module):
    embed_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    attn: eqx.nn.MultiheadAttention
    rotary_embed: eqx.nn.RotaryPositionalEmbedding
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(
        self, embed_size: int, hidden_size: int, num_heads: int, key: PRNGKeyArray
    ):
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        key1, key2, key3 = jr.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads,
            self.embed_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=key1,
        )
        self.rotary_embed = eqx.nn.RotaryPositionalEmbedding(embed_size // num_heads)
        self.layer_norm1 = eqx.nn.LayerNorm(self.embed_size)
        self.layer_norm2 = eqx.nn.LayerNorm(self.embed_size)
        self.linear1 = eqx.nn.Linear(self.embed_size, self.hidden_size, key=key2)
        self.linear2 = eqx.nn.Linear(self.hidden_size, self.embed_size, key=key3)

    def __call__(
        self,
        x: Float[Array, "length {self.embed_size}"],
        is_pad: Bool[Array, " length"],
    ) -> Float[Array, "length {self.embed_size}"]:
        def process_heads(
            query_heads: Float[Array, "seq_length num_heads qk_size"],
            key_heads: Float[Array, "seq_length num_heads qk_size"],
            value_heads: Float[Array, "seq_length num_heads vo_size"],
        ) -> tuple[
            Float[Array, "seq_length num_heads qk_size"],
            Float[Array, "seq_length num_heads qk_size"],
            Float[Array, "seq_length num_heads vo_size"],
        ]:
            query_heads = jax.vmap(self.rotary_embed, in_axes=1, out_axes=1)(
                query_heads
            )
            key_heads = jax.vmap(self.rotary_embed, in_axes=1, out_axes=1)(key_heads)

            return query_heads, key_heads, value_heads

        length, _ = x.shape
        [length2] = is_pad.shape
        assert length == length2
        mask = jnp.broadcast_to(jnp.logical_not(is_pad), (length, length))
        y = jax.vmap(self.layer_norm1)(x)
        y = self.attn(y, y, y, process_heads=process_heads, mask=mask)
        x = x + y
        y = jax.vmap(self.layer_norm2)(x)
        y = jax.vmap(self.linear1)(y)
        y = jax.nn.gelu(y, approximate=False)
        y = jax.vmap(self.linear2)(y)
        x = x + y
        return x
