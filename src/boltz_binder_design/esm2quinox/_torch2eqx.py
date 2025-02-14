import sys
import types
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from ._esm2 import ESM2
from ._layer import TransformerLayer


if TYPE_CHECKING:
    import esm
    import torch


def _check_esm(layer_name: str) -> types.ModuleType:
    try:
        esm = sys.modules["esm"]
    except KeyError as e:
        raise RuntimeError(
            "The PyTorch layer passed should be an instance of a "
            f"`{layer_name}`. However, ESM does not appear to be "
            "imported."
        ) from e
    if esm.__version__ not in ("2.0.0", "2.0.1"):
        raise RuntimeError("`torch2eqx` only suports ESM versions 2.0.0 and 2.0.1.")

    return esm


def _assert_no_structs_leaf(xi):
    assert not isinstance(xi, jax.ShapeDtypeStruct)


def _assert_no_structs(x: PyTree):
    jtu.tree_map(_assert_no_structs_leaf, x)


def torch2eqx_tensor(x):
    return jnp.asarray(np.asarray(x.detach().cpu()))


def torch2eqx_linear(torch_linear: "torch.nn.Linear") -> eqx.nn.Linear:  # pyright: ignore
    def make():
        return eqx.nn.Linear(
            torch_linear.in_features, torch_linear.out_features, key=jr.key(0)
        )

    eqx_linear = jax.eval_shape(make)
    assert eqx_linear.weight.shape == torch_linear.weight.shape
    assert eqx_linear.bias.shape == torch_linear.bias.shape
    eqx_linear = eqx.tree_at(
        lambda m: (m.weight, m.bias),
        eqx_linear,
        (torch2eqx_tensor(torch_linear.weight), torch2eqx_tensor(torch_linear.bias)),
    )
    _assert_no_structs(eqx_linear)
    return eqx_linear


def torch2eqx_layer_norm(torch_layer_norm: "torch.nn.LayerNorm") -> eqx.nn.LayerNorm:  # pyright: ignore
    def make():
        return eqx.nn.LayerNorm(
            torch_layer_norm.normalized_shape,
            eps=torch_layer_norm.eps,
            use_weight=torch_layer_norm.elementwise_affine,
            use_bias=torch_layer_norm.elementwise_affine,
        )

    eqx_layer_norm = jax.eval_shape(make)
    assert eqx_layer_norm.weight.shape == torch_layer_norm.weight.shape
    assert eqx_layer_norm.bias.shape == torch_layer_norm.bias.shape
    eqx_layer_norm = eqx.tree_at(
        lambda m: (m.weight, m.bias),
        eqx_layer_norm,
        (
            torch2eqx_tensor(torch_layer_norm.weight),
            torch2eqx_tensor(torch_layer_norm.bias),
        ),
    )
    _assert_no_structs(eqx_layer_norm)
    return eqx_layer_norm


def torch2eqx_transformer_layer(
    torch_layer: "esm.modules.TransformerLayer",  # pyright: ignore
) -> TransformerLayer:
    esm = _check_esm("esm.modules.TransformerLayer")

    if not isinstance(torch_layer, esm.modules.TransformerLayer):
        raise ValueError(
            "The PyTorch layer passed should be an instance of a "
            "`esm.modules.TransformerLayer`."
        )
    if torch_layer.self_attn.bias_k is not None:
        raise ValueError("Only `add_bias_kv=False` is supported.")
    if not isinstance(torch_layer.final_layer_norm, esm.modules.ESM1bLayerNorm):
        raise ValueError("Only `use_esm1b_layer_norm=True` is supported.")
    if not torch_layer.use_rotary_embeddings:
        raise ValueError("Only `use_rotary_embeddings=True` is supported.")

    def make():
        return TransformerLayer(
            embed_size=torch_layer.embed_dim,
            hidden_size=torch_layer.ffn_embed_dim,
            num_heads=torch_layer.attention_heads,
            key=jr.key(0),
        )

    eqx_layer = jax.eval_shape(make)
    query_proj = torch2eqx_linear(torch_layer.self_attn.q_proj)
    key_proj = torch2eqx_linear(torch_layer.self_attn.k_proj)
    value_proj = torch2eqx_linear(torch_layer.self_attn.v_proj)
    output_proj = torch2eqx_linear(torch_layer.self_attn.out_proj)
    layer_norm1 = torch2eqx_layer_norm(torch_layer.self_attn_layer_norm)
    layer_norm2 = torch2eqx_layer_norm(torch_layer.final_layer_norm)
    linear1 = torch2eqx_linear(torch_layer.fc1)
    linear2 = torch2eqx_linear(torch_layer.fc2)
    eqx_layer = eqx.tree_at(
        lambda m: (
            m.attn.query_proj,
            m.attn.key_proj,
            m.attn.value_proj,
            m.attn.output_proj,
            m.attn.dropout,
            m.layer_norm1,
            m.layer_norm2,
            m.linear1,
            m.linear2,
        ),
        eqx_layer,
        (
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            eqx.nn.Dropout(p=0.0),
            layer_norm1,
            layer_norm2,
            linear1,
            linear2,
        ),
    )
    _assert_no_structs(eqx_layer)
    return eqx_layer


def _stack(*x):
    is_arrays = [eqx.is_array(xi) for xi in x]
    if all(is_arrays):
        return jnp.stack(x)
    elif not any(is_arrays):
        values = set(x)
        if len(values) == 1:
            return values.pop()
        else:
            assert False
    else:
        assert False


def torch2eqx_esm2(torch_layer: "esm.ESM2") -> ESM2:  # pyright: ignore
    esm = _check_esm("esm.ESM2")

    if not isinstance(torch_layer, esm.ESM2):
        raise ValueError(
            "The PyTorch layer passed should be an instance of a `esm.ESM2`."
        )

    def make():
        return ESM2(
            num_layers=torch_layer.num_layers,
            embed_size=torch_layer.embed_dim,
            num_heads=torch_layer.attention_heads,
            token_dropout=torch_layer.token_dropout,
            key=jr.key(0),
        )

    eqx_layer = jax.eval_shape(make)
    layers = []
    for torch_transformer_layer in torch_layer.layers:
        layers.append(torch2eqx_transformer_layer(torch_transformer_layer))
    layers = jtu.tree_map(_stack, *layers)
    layer_norm1 = torch2eqx_layer_norm(torch_layer.emb_layer_norm_after)
    layer_norm2 = torch2eqx_layer_norm(torch_layer.lm_head.layer_norm)
    linear1 = torch2eqx_linear(torch_layer.lm_head.dense)
    # linear2:
    weight = torch2eqx_tensor(torch_layer.lm_head.weight)
    bias = torch2eqx_tensor(torch_layer.lm_head.bias)
    assert eqx_layer.logit_head.linear2.weight.shape == weight.shape
    assert eqx_layer.logit_head.linear2.bias.shape == bias.shape
    eqx_layer = eqx.tree_at(
        lambda m: (
            m.layers,
            m.layer_norm,
            m.logit_head.layer_norm,
            m.logit_head.linear1,
            m.logit_head.linear2.weight,
            m.logit_head.linear2.bias,
        ),
        eqx_layer,
        (layers, layer_norm1, layer_norm2, linear1, weight, bias),
    )
    _assert_no_structs(eqx_layer)
    return eqx_layer


def from_torch(torch_esm2: "esm.ESM2") -> ESM2:  # pyright: ignore
    """Loads an ESM2 model from the equivalent PyTorch model provided by the original
    `esm` library.

    **Arguments:**

    - `torch_esm`: an `esm.ESM2` PyTorch module.

    **Returns:**

    An `esm2equinox.ESM2` model.

    !!! Example

        ```python
        import esm  # pip install fair-esm==2.0.0
        import esm2quinox

        torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
        model = esm2quinox.from_torch(torch_model)
        ```
    """
    return torch2eqx_esm2(torch_esm2)
