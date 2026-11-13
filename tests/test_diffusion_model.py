"""Test `DiffusionStructurePredictionModel`'s shared logic on a tiny fake model."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold.model import StructureModel
from jfold.sampling import UpstreamSampler, masked_centering, rigid_align_stop_gradient
from jfold.slots import (
    AbstractConfidenceModule,
    AbstractDiffusionModule,
    AbstractInputEmbedder,
    AbstractTrunk,
    AbstractTrunkInit,
    ConfidenceLogits,
    TrunkInputs,
    TrunkState,
)

import mosaic.structure_prediction as sp
from mosaic.common import LossTerm
from mosaic.structure_prediction import PAE_BINS

T, C, B = 6, 4, 3  # tokens, pair channels, distogram bins


class FakeFeatures(eqx.Module):
    sequence: jax.Array  # [T, 20]


class _Embedder(AbstractInputEmbedder[FakeFeatures]):
    w_in: jax.Array

    def __call__(self, features):
        return features.sequence @ self.w_in  # [T, C]


class _TrunkInit(AbstractTrunkInit[FakeFeatures]):
    def __call__(self, features, s_inputs, *, key=None):
        z_init = s_inputs[:, None] + s_inputs[None, :]
        return TrunkInputs(s_inputs=s_inputs, s_init=s_inputs, z_init=z_init)


class _Trunk(AbstractTrunk[FakeFeatures]):
    w_rec: jax.Array

    def init_state(self, inputs, *, key=None):
        return TrunkState(s=inputs.s_init, z=0.1 * jax.random.normal(key, inputs.z_init.shape))

    def __call__(self, features, inputs, state, *, key=None):
        noise = 0.01 * jax.random.normal(key, state.z.shape)
        return TrunkState(s=state.s, z=jnp.tanh(inputs.z_init + state.z @ self.w_rec) + noise)


class _Context(eqx.Module):
    target: jax.Array  # [T, 3]
    atom_mask: jax.Array  # [T]


class _Diffusion(AbstractDiffusionModule[FakeFeatures]):
    """A fake denoiser predicting a target read off the pair, one atom per token."""

    def prepare(self, features, inputs, state):
        z = state.z
        target = jnp.stack([z[..., 0].sum(-1), z[..., 1].sum(-1), jnp.arange(T)], -1)
        return _Context(target=target, atom_mask=jnp.ones(T, bool))

    def __call__(self, context, x_noisy, t):
        return context.target


class _Distogram(eqx.Module):
    def __call__(self, z):
        return z[..., :B]


class _Confidence(AbstractConfidenceModule[FakeFeatures]):
    def __call__(self, features, inputs, state, coords, *, key=None):
        z = state.z
        plddt = jnp.diagonal(z[..., 0]) + coords[:, 0]
        return ConfidenceLogits(
            pae=jnp.broadcast_to(z[..., :1], (T, T, len(PAE_BINS))),
            pde=None,
            plddt=jnp.stack([jnp.zeros(T), plddt], -1),
            resolved=None,
        )


class FakeModel(sp.DiffusionStructurePredictionModel[FakeFeatures]):
    model: StructureModel
    calls: dict = eqx.field(static=True, default_factory=dict)
    design_recycling_steps: int = eqx.field(static=True, default=3)
    inference_recycling_steps: int = eqx.field(static=True, default=3)
    backprop_all_passes: bool = eqx.field(static=True, default=False)
    distogram_bins = np.arange(B, dtype=np.float32)

    def target_only_features(self, chains):
        raise NotImplementedError

    def binder_features(self, binder_length, chains):
        raise NotImplementedError

    def to_structure(self, features, writer, output, sequence):
        raise NotImplementedError

    def set_sequence(self, features, sequence, *, key):
        del key
        return FakeFeatures(features.sequence.at[: sequence.shape[0]].set(sequence))

    def jfold_features(self, features):
        return features

    def embed(self, features, *, key):
        self.calls["embed"] = self.calls.get("embed", 0) + 1
        return super().embed(features, key=key)

    def token_plddt(self, plddt, trunk):
        return plddt  # the fake head is per token

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


def _model(has_confidence=True, **kwargs):
    k1, k2 = jax.random.split(jax.random.key(0))
    model = StructureModel(
        input_embedder=_Embedder(jax.random.normal(k1, (20, C))),
        trunk_init=_TrunkInit(),
        trunk=_Trunk(0.5 * jax.random.normal(k2, (C, C))),
        diffusion=_Diffusion(),
        distogram=_Distogram(),
        confidence=_Confidence() if has_confidence else None,
        sampling=UpstreamSampler(
            sigma_data=16.0,
            s_max=160.0,
            s_min=4e-4,
            rho=7.0,
            num_steps=3,
            end="append_zero",
            gamma_0=0.8,
            gamma_min=1.0,
            noise_scale=1.003,
            step_scale=1.5,
            center=masked_centering,
            augment=True,
            augment_after_noise=False,
            align=rigid_align_stop_gradient,
        ),
    )
    return FakeModel(model=model, **kwargs)


def _features():
    return FakeFeatures(jax.nn.one_hot(jnp.arange(T) % 20, 20))


SEQUENCE = jax.nn.softmax(jax.random.normal(jax.random.key(1), (2, 20)))


@pytest.mark.parametrize("recycling_steps", [1, 4])
def test_steps_compose_to_run_trunk(recycling_steps):
    model, key = _model(), jax.random.key(2)
    features, inputs = model.prepare(_features(), SEQUENCE, key=key)

    state = model.run_trunk(features, inputs, recycling_steps=recycling_steps, key=key).state

    k_init, k_passes = jax.random.split(key)
    manual = model.init_state(inputs, key=k_init)
    for k in jax.random.split(k_passes, recycling_steps):
        manual = model.step(inputs, manual, key=k)

    np.testing.assert_allclose(manual.z, state.z, atol=1e-6)


class _DistogramSum(LossTerm):
    """A loss summing the fake model's distogram logits."""

    def __call__(self, sequence, output, *, key):
        del sequence, key
        value = output.distogram_logits.sum()
        return value, {"value": value}


def test_build_loss_runs_the_trunk_from_initial_state():
    model, key = _model(), jax.random.key(8)
    initial = TrunkState(s=jnp.zeros((T, C)), z=jnp.full((T, T, C), 0.3))
    loss = model.build_loss(loss=_DistogramSum(), features=_features(), initial_state=initial)

    value, _ = eqx.filter_jit(loss)(SEQUENCE, key=key)

    k_prepare, k_trunk, _ = jax.random.split(key, 3)
    _, inputs = model.prepare(_features(), SEQUENCE, key=k_prepare)
    manual = initial
    for k in jax.random.split(jax.random.split(k_trunk)[1], model.design_recycling_steps):
        manual = model.step(inputs, manual, key=k)
    np.testing.assert_allclose(value, manual.z[..., :B].sum(), rtol=1e-5)


def test_backprop_modes_share_the_forward_and_differ_in_gradient():
    key = jax.random.key(3)

    def trunk(sequence, all_passes):
        model = dataclasses.replace(_model(), backprop_all_passes=all_passes)
        features, inputs = model.prepare(_features(), sequence, key=key)
        trunk = model.run_trunk(features, inputs, recycling_steps=4, key=key)
        return trunk.state.z.sum()

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
    output_keys, loss_keys = jax.vmap(jax.random.split)(sample_keys).swapaxes(0, 1)
    seen = np.sort(np.asarray(aux["loss_key"]), axis=0)  # aux is sorted by loss value
    np.testing.assert_array_equal(seen, np.sort(np.asarray(jax.random.key_data(loss_keys)), axis=0))
    assert not np.array_equal(seen, np.sort(np.asarray(jax.random.key_data(output_keys)), axis=0))


def test_unavailable_confidence_gives_nan_plddt_and_pae():
    model = _model(has_confidence=False)
    output = model.model_output(features=_features(), PSSM=SEQUENCE, key=jax.random.key(5))

    assert np.isnan(output.plddt).all() and np.isnan(output.pae).all()
    np.testing.assert_allclose(output.full_sequence[:2], SEQUENCE, atol=1e-6)
    np.testing.assert_allclose(output.backbone_coordinates[:, 1], output.structure_coordinates)
