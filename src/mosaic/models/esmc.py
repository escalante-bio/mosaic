"""ESMC precision conversion without changing the upstream JAX model globally."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import equinox as eqx
import esmjfold2
import jax
import jax.numpy as jnp
from esmjfold2.esmc import (
    ESMCForMaskedLM,
    LayerNormLinear,
    LayerNormMLP,
    TransformerStack,
)
from esmjfold2.primitives import LayerNorm, Linear
from jaxtyping import Array

if TYPE_CHECKING:
    from esm.models.esmc import EsmcForMaskedLM


def _layer_norm(
    x: Array, weight: Array | None, bias: Array | None, eps: float
) -> Array:
    x = x.astype(jnp.float32)
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    x = centered * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        x = x * weight.astype(jnp.float32)
    if bias is not None:
        x = x + bias.astype(jnp.float32)
    return x


class _Float32LayerNorm(LayerNorm):
    def __call__(self, x: Array) -> Array:
        return _layer_norm(x, self.weight, self.bias, self.eps).astype(x.dtype)


class _Float32LayerNormLinear(LayerNormLinear):
    def __call__(self, x: Array) -> Array:
        x = _layer_norm(
            x, self.layer_norm_weight, self.layer_norm_bias, self.eps
        ).astype(self.weight.dtype)
        return jnp.einsum(
            "...i,oi->...o", x, self.weight, preferred_element_type=jnp.float32
        )


class _Bfloat16Linear(Linear):
    def __call__(self, x: Array) -> Array:
        output = jnp.einsum(
            "...i,oi->...o",
            x.astype(self.weight.dtype),
            self.weight,
            preferred_element_type=jnp.float32,
        )
        if self.bias is not None:
            output = output + self.bias.astype(jnp.float32)
        return output


class _Float32LayerNormMLP(LayerNormMLP):
    def __call__(self, x: Array) -> Array:
        x = _layer_norm(
            x, self.layer_norm_weight, self.layer_norm_bias, self.eps
        ).astype(self.fc1_weight.dtype)
        x = jnp.einsum(
            "...i,oi->...o", x, self.fc1_weight, preferred_element_type=jnp.float32
        )
        gate, value = jnp.split(x, 2, axis=-1)
        x = jax.nn.silu(gate) * value
        return jnp.einsum(
            "...i,oi->...o",
            x.astype(self.fc2_weight.dtype),
            self.fc2_weight,
            preferred_element_type=jnp.float32,
        )


class _Float32ResidualStack(TransformerStack):
    def __call__(
        self,
        x: Array,
        sequence_id: Array | None = None,
        *,
        collect_hidden_states: bool = True,
    ) -> tuple[Array, Array | None]:
        output, hidden = super().__call__(
            x.astype(jnp.float32),
            sequence_id,
            collect_hidden_states=collect_hidden_states,
        )
        return output, (None if hidden is None else hidden.astype(jnp.bfloat16))


def esmc_bfloat16[Model: eqx.Module](model: Model) -> Model:
    """Use bf16 projection operands with float32 outputs and attention.

    Works on either an ESMC backbone or a masked-LM model, including the
    partitioned transformer blocks used by ``lax.scan``. Other model instances
    and the upstream module classes are unchanged.
    """
    replacements = {
        LayerNorm: _Float32LayerNorm,
        Linear: _Bfloat16Linear,
        LayerNormLinear: _Float32LayerNormLinear,
        LayerNormMLP: _Float32LayerNormMLP,
        TransformerStack: _Float32ResidualStack,
    }

    def convert(module):
        cls = replacements.get(type(module))
        if cls is None:
            return (
                module.astype(jnp.bfloat16) if eqx.is_inexact_array(module) else module
            )
        values = {}
        for field in fields(module):
            value = getattr(module, field.name)
            if isinstance(module, LayerNorm) or field.name in (
                "layer_norm_weight",
                "layer_norm_bias",
            ):
                values[field.name] = (
                    value.astype(jnp.float32) if eqx.is_inexact_array(value) else value
                )
            else:
                values[field.name] = jax.tree.map(
                    convert, value, is_leaf=lambda x: type(x) in replacements
                )
        return cls(**values)

    return jax.tree.map(convert, model, is_leaf=lambda x: type(x) in replacements)


def esmc_from_torch(
    model: EsmcForMaskedLM, *, bfloat16: bool = False
) -> ESMCForMaskedLM:
    """Convert on CPU before bf16 transfer, avoiding an fp32 GPU model copy."""
    if not bfloat16:
        return esmjfold2.from_torch(model)
    destination = jnp.empty(()).device
    with jax.default_device(jax.devices("cpu")[0]):
        converted = esmc_bfloat16(esmjfold2.from_torch(model))
    return jax.tree.map(
        lambda x: jax.device_put(x, destination) if eqx.is_array(x) else x, converted
    )
