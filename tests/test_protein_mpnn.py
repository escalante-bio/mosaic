from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.losses.protein_mpnn import (
    _autoregressive_inverse_fold,
    _per_chain_residue_idx,
    boltz_to_mpnn_matrix,
    inverse_fold,
)
from mosaic.proteinmpnn.mpnn import ProteinMPNN


def _problem(target=(3, 7)):
    hidden_dim = 2

    def encode(*, X, **kwargs):
        length = X.shape[0]
        neighbors = jnp.broadcast_to(jnp.arange(length), (length, length))[None]
        return (
            jnp.zeros((1, length, hidden_dim)),
            jnp.zeros((1, length, length, hidden_dim)),
            neighbors,
        )

    def decoder_layer(h_V, h_E, mask_V=None):
        sequence = h_E[..., hidden_dim : 2 * hidden_dim]
        neighbors = h_E[..., 2 * hidden_dim :]
        messages = (sequence + 0.5 * neighbors).sum(-2)
        return (h_V + messages) * mask_V[..., None]

    mpnn = SimpleNamespace(
        W_s=SimpleNamespace(weight=jnp.arange(42).reshape(21, hidden_dim) / 20),
        W_out=lambda x: x @ jnp.sin(jnp.arange(42).reshape(21, hidden_dim)).T,
        decoder_layers=(decoder_layer,),
        encode=encode,
    )
    mpnn.decode = lambda **kwargs: ProteinMPNN.decode(mpnn, **kwargs)
    output = SimpleNamespace(
        backbone_coordinates=jnp.zeros((6, 4, 3)),
        full_sequence=jnp.vstack(
            (jnp.zeros((4, 20)), jax.nn.one_hot(jnp.array(target), 20))
        ),
        asym_id=jnp.array([0, 0, 0, 0, 1, 1]),
        residue_idx=jnp.array([0, 1, 2, 3, 0, 1]),
    )
    return mpnn, output


def _full_decoder_sample(mpnn, binder_length, output, temp, key, bias):
    total_length = output.full_sequence.shape[0]
    mask = jnp.ones(total_length, dtype=jnp.int32)
    encode_key, order_key, sample_key = jax.random.split(key, 3)
    h_V, h_E, E_idx = mpnn.encode(
        X=output.backbone_coordinates,
        mask=mask,
        residue_idx=_per_chain_residue_idx(output.asym_id, output.residue_idx),
        chain_encoding_all=output.asym_id,
        key=encode_key,
    )
    decoding_order = jax.random.uniform(order_key, (total_length,))
    decoding_order = decoding_order.at[:binder_length].add(2.0)
    binder_order = np.asarray(jnp.argsort(decoding_order[:binder_length]))
    gumbel = jax.random.gumbel(sample_key, (binder_length, 20))
    token_matrix = jnp.asarray(boltz_to_mpnn_matrix())
    sequence = output.full_sequence

    for position in binder_order:
        logits = (
            mpnn.decode(
                S=sequence @ token_matrix,
                h_V=h_V,
                h_E=h_E,
                E_idx=E_idx,
                mask=mask,
                decoding_order=decoding_order,
            )[0, position]
            @ token_matrix.T
        )
        residue = (logits + bias[position] + temp * gumbel[position]).argmax()
        sequence = sequence.at[position].set(jax.nn.one_hot(residue, 20))

    return sequence[:binder_length].argmax(-1)


@pytest.mark.parametrize(
    ("temperature", "seed", "target", "expected"),
    [
        (0.0, 11, (3, 7), [12, 12, 12, 12]),
        (0.2, 12, (3, 7), [5, 12, 12, 5]),
        (0.8, 13, (2, 9), [12, 18, 5, 12]),
    ],
)
def test_autoregressive_sampler_matches_full_decoder(
    temperature, seed, target, expected
) -> None:
    mpnn, output = _problem(target)
    key = jax.random.key(seed)
    bias = jnp.broadcast_to(jnp.linspace(-0.2, 0.2, 20), (4, 20))

    reference = _full_decoder_sample(mpnn, 4, output, temperature, key, bias)
    np.testing.assert_array_equal(reference, expected)
    actual = _autoregressive_inverse_fold(
        mpnn, 4, output, temperature, key, bias
    )

    np.testing.assert_array_equal(actual, reference)


def test_autoregressive_sampler_matches_full_decoder_under_jit() -> None:
    mpnn, output = _problem()
    keys = jax.random.split(jax.random.key(7), 3)
    bias = jnp.broadcast_to(jnp.linspace(-0.2, 0.2, 20), (4, 20))

    expected = np.stack(
        [_full_decoder_sample(mpnn, 4, output, 0.0, key, bias) for key in keys]
    )
    np.testing.assert_array_equal(expected, np.full((3, 4), 12))

    sample = eqx.filter_jit(
        jax.vmap(
            lambda key: inverse_fold(
                mpnn, 4, output, jnp.array(0.0), key, bias=bias
            )
        )
    )
    actual = sample(keys)

    np.testing.assert_array_equal(actual, expected)
