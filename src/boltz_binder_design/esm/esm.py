# Adapted from https://github.com/irhum/esmjax
import jax
from jax.lax import scan
from jax import vmap, numpy as jnp
from jaxtyping import Array, Bool, Float, Integer
import equinox as eqx

from ..util import fold_in
from .mha import RoPEMultiHeadDotProductAttention
from .tokenizer import PAD_IDX, MASK_IDX, TOKENIZER


class EncoderLayer(eqx.Module):
    self_attn_layer_norm: eqx.nn.LayerNorm
    self_attn: RoPEMultiHeadDotProductAttention
    final_layer_norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        q_dim: int,
        kv_dim: int,
        *,
        key,
    ):
        self.self_attn_layer_norm = eqx.nn.LayerNorm(shape=embed_dim, eps=1e-5)
        self.self_attn = RoPEMultiHeadDotProductAttention(
            num_heads,
            embed_dim,
            embed_dim,
            q_dim,
            kv_dim,
            key=fold_in(key, "self_attn"),
        )
        self.final_layer_norm = eqx.nn.LayerNorm(shape=embed_dim, eps=1e-5)
        self.fc1 = eqx.nn.Linear(embed_dim, ffn_embed_dim, key=fold_in(key, "fc1"))
        self.fc2 = eqx.nn.Linear(ffn_embed_dim, embed_dim, key=fold_in(key, "fc2"))

    def __call__(
        self,
        x: Float[Array, "len embed"],
        mask: Bool[Array, "len len"] | None = None,
    ) -> Float[Array, "len embed"]:
        return self._mha_weights_and_output(x, mask)[0]

    def _mha_weights_and_output(
        self,
        x: Float[Array, "len embed"],
        mask: Bool[Array, "len len"] | None = None,
    ) -> tuple[Float[Array, "len embed"], Float[Array, "heads len len"]]:
        # ln + mha
        mha_output, mha_weights = self.self_attn(
            vmap(self.self_attn_layer_norm)(x), mask
        )
        x = x + mha_output

        # ln + mlp
        def mlp_block(v):
            v = self.final_layer_norm(v)
            return self.fc2(jax.nn.gelu(self.fc1(v), approximate=False))

        x = x + vmap(mlp_block)(x)

        return x, mha_weights


class LMHead(eqx.Module):
    bias: Float[Array, "vocab"]
    dense: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    weight: Float[Array, "vocab embed"]

    def __init__(self, embed_dim: int, vocab_dim: int, key: jax.random.PRNGKey):
        self.dense = eqx.nn.Linear(embed_dim, embed_dim, key=fold_in(key, "dense"))
        self.bias = jnp.zeros(vocab_dim)
        self.layer_norm = eqx.nn.LayerNorm(shape=embed_dim, eps=1e-5)
        self.weight = jnp.zeros((vocab_dim, embed_dim))

    def __call__(self, x: Float[Array, "len embed"]) -> Float[Array, "len vocab"]:
        def _per_token(x):
            return (
                self.weight
                @ self.layer_norm(jax.nn.gelu(self.dense(x), approximate=False))
                + self.bias
            )

        return vmap(_per_token)(x)


class ESM2(eqx.Module):
    embed_tokens: eqx.nn.Embedding
    layers: EncoderLayer
    num_layers: int
    emb_layer_norm_after: eqx.nn.LayerNorm
    lm_head: LMHead

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        *,
        key,
    ):
        self.embed_tokens = eqx.nn.Embedding(
            len(TOKENIZER), embed_dim, key=fold_in(key, "embedding")
        )
        self.layers = eqx.filter_vmap(
            lambda k: EncoderLayer(
                num_heads, embed_dim, embed_dim * 4, embed_dim, embed_dim, key=k
            )
        )(jax.random.split(key, num_layers))
        self.num_layers = num_layers
        self.emb_layer_norm_after = eqx.nn.LayerNorm(shape=embed_dim, eps=1e-5)
        self.lm_head = LMHead(embed_dim, len(TOKENIZER), key=key)

    def __call__(self, tokens: Integer[Array, "len"]) -> Float[Array, "len embed"]:
        # compute padding masks
        pad_embed_mask, pad_att_mask = self.pad_masks(tokens)

        # compute initial embeddings and scale
        embeds = vmap(self.embed_tokens)(tokens)
        embeds = self.rescale_masked_tokens(tokens, embeds)

        ### scan the trunk
        embeds = self._apply_trunk(embeds, pad_att_mask)
        return embeds * pad_embed_mask
    
    def _apply_trunk(self,embeds, pad_att_mask):
        ### scan the trunk
        stacked_params, not_params = eqx.partition(self.layers, eqx.is_array)

        def _body(embeddings, layer_params):
            layer = eqx.combine(layer_params, not_params)
            return layer(embeddings, mask=pad_att_mask), None

        embeds, _ = scan(_body, embeds, stacked_params)

        return vmap(self.emb_layer_norm_after)(embeds)


    def attention_weights(
        self, tokens: Integer[Array, "len"]
    ) -> Float[Array, "depth heads len len"]:
        # compute padding masks
        _, pad_att_mask = self.pad_masks(tokens)

        # compute initial embeddings and scale
        embeds = vmap(self.embed_tokens)(tokens)
        embeds = self.rescale_masked_tokens(tokens, embeds)

        # DO THIS FIRST, in constructor. (Duh)
        stacked_params, not_params = eqx.partition(self.layers, eqx.is_array)

        def _body(embeddings, layer_params):
            layer = eqx.combine(layer_params, not_params)
            return layer._mha_weights_and_output(embeds, mask=pad_att_mask)

        _, stacked_atts = scan(_body, embeds, stacked_params)

        return stacked_atts

    def pad_masks(self, tokens):
        pad_embed_mask = tokens != PAD_IDX
        pad_embed_mask = pad_embed_mask[:, None]
        pad_att_mask = jnp.einsum("xh,yh->xy", pad_embed_mask, pad_embed_mask)
        return pad_embed_mask, pad_att_mask

    def rescale_masked_tokens(self, tokens, embeds):
        # zero masked embeddings
        embeds = embeds * (tokens != MASK_IDX)[:, None]

        mask_ratio_train = 0.15 * 0.8
        src_length = (tokens != PAD_IDX).sum() + 1e-8
        # Get the mask token to sequence length ratio
        mask_ratio_observed = (tokens == MASK_IDX).sum() / src_length
        # Rescale by the ratio between theoretical and observed.
        embeds = embeds * (1 - mask_ratio_train) / (1 - mask_ratio_observed)

        return embeds
