"""`DiffusionStructurePredictionModel`'s shared logic on a tiny fake model."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mosaic.structure_prediction as sp
from mosaic.common import LossTerm
from mosaic.sampling import Sampler, schedules

T, C, B = 6, 4, 3  # tokens, pair channels, bins


class FakeFeatures(eqx.Module):
    sequence: jax.Array  # [T, 20]


class FakeInputs(eqx.Module):
    pair_init: jax.Array  # [T, T, C]


def _target(denoise_target, x, sigma):
    del x, sigma
    return denoise_target


class FakeModel(sp.DiffusionStructurePredictionModel[FakeFeatures, FakeInputs]):
    w_in: jax.Array
    w_rec: jax.Array
    has_confidence: bool = eqx.field(static=True, default=True)
    calls: dict = eqx.field(static=True, default_factory=dict)
    design_recycling_steps: int = eqx.field(static=True, default=3)
    inference_recycling_steps: int = eqx.field(static=True, default=3)
    backprop_all_passes: bool = eqx.field(static=True, default=False)

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def upstream_sampler(self, num_steps=None, *, schedule=None):
        sigmas = (schedule or schedules.Karras())(np.linspace(0.0, 1.0, num_steps or 3))
        return Sampler(sigmas=schedules.append_zero(sigmas))

    def target_only_features(self, chains):
        raise NotImplementedError

    def binder_features(self, binder_length, chains):
        raise NotImplementedError

    def to_structure(self, features, writer, output, sequence):
        raise NotImplementedError

    def set_sequence(self, features, sequence, *, key):
        del key
        return FakeFeatures(features.sequence.at[: sequence.shape[0]].set(sequence))

    def embed(self, features, *, key):
        del key
        self._count("embed")
        h = features.sequence @ self.w_in  # [T, C]
        return FakeInputs(pair_init=h[:, None] + h[None, :])

    def init_state(self, features, inputs, *, key):
        del features
        return sp.TrunkState(pair=0.1 * jax.random.normal(key, inputs.pair_init.shape))

    def step(self, features, inputs, state, *, key):
        del features
        noise = 0.01 * jax.random.normal(key, state.pair.shape)
        return sp.TrunkState(pair=jnp.tanh(inputs.pair_init + state.pair @ self.w_rec) + noise)

    def structure_denoiser(self, trunk):
        pair = trunk.state.pair
        target = jnp.stack([pair[..., 0].sum(-1), pair[..., 1].sum(-1), jnp.arange(T)], -1)
        return eqx.Partial(_target, target), jnp.ones(T)

    def distogram(self, trunk):
        return sp.Distogram(logits=trunk.state.pair[..., :B], bins=jnp.arange(B, dtype=jnp.float32))

    def confidence(self, trunk, coords):
        if not self.has_confidence:
            return sp.Confidence.unavailable(T)
        pair = trunk.state.pair
        return sp.Confidence(
            plddt=jax.nn.sigmoid(jnp.diagonal(pair[..., 0]) + coords[:, 0]),
            pae_logits=pair[..., :B],
            pae_bins=jnp.arange(B, dtype=jnp.float32),
        )

    def tokens(self, features):
        return sp.TokenInfo(
            sequence=features.sequence,
            asym_id=jnp.zeros(T, jnp.int32),
            residue_index=jnp.arange(T),
            mask=jnp.ones(T, bool),
        )

    def atom37(self, features, coords):
        del features
        atom37 = jnp.zeros((T, 37, 3)).at[:, 1].set(coords)  # CA only
        return sp.Atom37(coords=atom37, mask=jnp.zeros((T, 37)).at[:, 1].set(1.0))


def _model(**kwargs):
    k1, k2 = jax.random.split(jax.random.key(0))
    return FakeModel(
        w_in=jax.random.normal(k1, (20, C)), w_rec=0.5 * jax.random.normal(k2, (C, C)), **kwargs
    )


def _features():
    return FakeFeatures(jax.nn.one_hot(jnp.arange(T) % 20, 20))


SEQUENCE = jax.nn.softmax(jax.random.normal(jax.random.key(1), (2, 20)))


@pytest.mark.parametrize("recycling_steps", [1, 4])
def test_steps_compose_to_run_trunk(recycling_steps):
    model, key = _model(), jax.random.key(2)
    features, inputs = model.prepare(_features(), SEQUENCE, key=key)

    state = model.run_trunk(features, inputs, recycling_steps=recycling_steps, key=key).state

    k_init, k_passes = jax.random.split(key)
    manual = model.init_state(features, inputs, key=k_init)
    for k in jax.random.split(k_passes, recycling_steps):
        manual = model.step(features, inputs, manual, key=k)

    np.testing.assert_allclose(manual.pair, state.pair, atol=1e-6)


def test_backprop_modes_share_the_forward_and_differ_in_gradient():
    key = jax.random.key(3)

    def trunk(sequence, all_passes):
        model = dataclasses.replace(_model(), backprop_all_passes=all_passes)
        features, inputs = model.prepare(_features(), sequence, key=key)
        trunk = model.run_trunk(features, inputs, recycling_steps=4, key=key)
        return trunk.state.pair.sum()

    v_all, g_all = jax.value_and_grad(trunk)(SEQUENCE, True)
    v_last, g_last = jax.value_and_grad(trunk)(SEQUENCE, False)

    np.testing.assert_allclose(v_all, v_last, rtol=1e-6)
    assert not np.allclose(g_all, g_last)


class _SampleLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        del sequence
        value = output.plddt.mean() + jax.random.uniform(key)
        return value, {"value": value}


@pytest.mark.parametrize("num_samples", [1, 3])
def test_multisample_loss_runs_trunk_once_and_reduces(num_samples):
    model = _model()
    loss = model.build_loss(loss=_SampleLoss(), features=_features(), num_samples=num_samples)

    value, aux = eqx.filter_jit(loss)(SEQUENCE, key=jax.random.key(4))

    assert model.calls == {"embed": 1}
    values = np.asarray(aux["value"])
    assert values.shape == (num_samples,)
    assert np.all(np.diff(values) >= 0)  # aux scalars are sorted by loss
    assert np.isclose(value, values.mean())


class _KeyLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        del sequence
        return jnp.zeros(()), {"loss_key": jax.random.key_data(key)}


def test_multisample_loss_gives_the_loss_its_own_key():
    loss = _model().build_loss(loss=_KeyLoss(), features=_features(), num_samples=2)
    key = jax.random.key(6)

    _, aux = eqx.filter_jit(loss)(SEQUENCE, key=key)

    sample_keys = jax.random.split(jax.random.split(key, 3)[2], 2)
    decode_keys, loss_keys = jax.vmap(jax.random.split)(sample_keys).swapaxes(0, 1)
    seen = np.sort(np.asarray(aux["loss_key"]), axis=0)  # aux is sorted by loss value
    np.testing.assert_array_equal(seen, np.sort(np.asarray(jax.random.key_data(loss_keys)), axis=0))
    assert not np.array_equal(seen, np.sort(np.asarray(jax.random.key_data(decode_keys)), axis=0))


def test_unavailable_confidence_gives_nan_plddt_and_pae():
    model = _model(has_confidence=False)
    output = model.model_output(features=_features(), PSSM=SEQUENCE, key=jax.random.key(5))

    assert np.isnan(output.plddt).all() and np.isnan(output.pae).all()
    np.testing.assert_allclose(output.full_sequence[:2], SEQUENCE, atol=1e-6)
    np.testing.assert_allclose(output.backbone_coordinates[:, 1], output.structure_coordinates)
