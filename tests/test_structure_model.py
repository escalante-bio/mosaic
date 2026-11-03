"""Shared trunk + diffusion logic (`mosaic.structure_model`) on a tiny fake model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mosaic.structure_model as sm
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

    @property
    def design_passes(self):
        return 3

    @property
    def inference_passes(self):
        return 3

    @property
    def backprop_all_passes(self):
        return False

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def design_sampler(self, num_steps=None, *, schedule=None):
        sigmas = (schedule or schedules.Karras())(np.linspace(0.0, 1.0, num_steps or 3))
        return Sampler(sigmas=schedules.append_zero(sigmas))

    inference_sampler = design_sampler

    def target_only_features(self, chains):
        raise NotImplementedError

    def binder_features(self, binder_length, chains):
        raise NotImplementedError

    def to_structure(self, features, writer, output, sequence):
        raise NotImplementedError

    def model_output(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def build_loss(self, **kwargs):
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

    def structure_denoiser(self, features, inputs, state):
        del features, inputs
        target = jnp.stack([state.pair[..., 0].sum(-1), state.pair[..., 1].sum(-1), jnp.arange(T)], -1)
        return eqx.Partial(_target, target), jnp.ones(T)

    def distogram(self, features, inputs, state):
        del features, inputs
        return sp.Distogram(logits=state.pair[..., :B], bins=jnp.arange(B, dtype=jnp.float32))

    def confidence(self, features, inputs, state, coords):
        del features, inputs
        if not self.has_confidence:
            return None
        return sp.Confidence(
            plddt=jax.nn.sigmoid(jnp.diagonal(state.pair[..., 0]) + coords[:, 0]),
            pae_logits=state.pair[..., :B],
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


@pytest.mark.parametrize("num_passes", [1, 4])
def test_steps_compose_to_run_trunk_and_trajectory(num_passes):
    model, key = _model(), jax.random.key(2)
    features, inputs = sm.prepare(model, _features(), SEQUENCE, key=key)

    state = sm.run_trunk(model, features, inputs, num_passes=num_passes, key=key)
    trajectory = sm.trunk_trajectory(model, features, inputs, num_passes=num_passes, key=key)

    k_init, k_passes = jax.random.split(key)
    manual = model.init_state(features, inputs, key=k_init)
    for k in jax.random.split(k_passes, num_passes):
        manual = model.step(features, inputs, manual, key=k)

    assert trajectory.pair.shape == (num_passes, T, T, C)
    np.testing.assert_allclose(trajectory.pair[-1], state.pair, atol=1e-6)
    np.testing.assert_allclose(manual.pair, state.pair, atol=1e-6)


def test_backprop_modes_share_the_forward_and_differ_in_gradient():
    model, key = _model(), jax.random.key(3)

    def trunk(sequence, all_passes):
        features, inputs = sm.prepare(model, _features(), sequence, key=key)
        state = sm.run_trunk(
            model, features, inputs, num_passes=4, backprop_all_passes=all_passes, key=key
        )
        return state.pair.sum()

    v_all, g_all = jax.value_and_grad(trunk)(SEQUENCE, True)
    v_last, g_last = jax.value_and_grad(trunk)(SEQUENCE, False)

    assert v_all == v_last
    assert not np.allclose(g_all, g_last)


class _SampleLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        del sequence
        value = output.plddt.mean() + jax.random.uniform(key)
        return value, {"value": value}


@pytest.mark.parametrize("num_samples", [1, 3])
def test_multisample_loss_runs_trunk_once_and_reduces(num_samples):
    model = _model()
    loss = sm.build_loss(model, loss=_SampleLoss(), features=_features(), num_samples=num_samples)

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
    loss = sm.build_loss(_model(), loss=_KeyLoss(), features=_features(), num_samples=2)
    key = jax.random.key(6)

    _, aux = eqx.filter_jit(loss)(SEQUENCE, key=key)

    sample_keys = jax.random.split(jax.random.split(key, 3)[2], 2)
    decode_keys, loss_keys = jax.vmap(jax.random.split)(sample_keys).swapaxes(0, 1)
    seen = np.sort(np.asarray(aux["loss_key"]), axis=0)  # aux is sorted by loss value
    np.testing.assert_array_equal(seen, np.sort(np.asarray(jax.random.key_data(loss_keys)), axis=0))
    assert not np.array_equal(seen, np.sort(np.asarray(jax.random.key_data(decode_keys)), axis=0))


def test_decode_fills_missing_confidence_with_nan():
    model = _model(has_confidence=False)
    output = sm.model_output(model, _features(), sequence=SEQUENCE, key=jax.random.key(5))

    assert np.isnan(output.plddt).all() and np.isnan(output.pae).all()
    np.testing.assert_allclose(output.full_sequence[:2], SEQUENCE, atol=1e-6)
    np.testing.assert_allclose(output.backbone_coordinates[:, 1], output.structure_coordinates)
