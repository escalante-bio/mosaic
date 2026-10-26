import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from mosaic.sampling import (
    Noise,
    Sampler,
    integrate,
    masked_centering,
    mean_centering,
    no_alignment,
    no_centering,
    rigid_align,
    rigid_align_stop_gradient,
    sample,
    schedules,
)
from mosaic.sampling.core import random_rotation

NUM_ATOMS = 32


def _denoise_to(target: Float[Array, "A 3"], x, sigma):
    """The exact x0-predictor for data concentrated on a single point cloud."""
    del x, sigma
    return target


def _point_denoiser(target):
    return eqx.Partial(_denoise_to, target)


MASK = jnp.ones(NUM_ATOMS)


def _target():
    return 10.0 * jax.random.normal(jax.random.key(0), (NUM_ATOMS, 3))


def _rmsd_after_alignment(x, target):
    aligned = rigid_align_stop_gradient(x, target, jnp.ones(x.shape[0]))
    return float(jnp.sqrt(((aligned - target) ** 2).sum(-1).mean()))


def _grid(num_points):
    return np.linspace(0.0, 1.0, num_points)


def _sigmas(num_steps):
    return schedules.zero_last(schedules.karras(_grid(num_steps + 1)))


ODE = Sampler(
    sigmas=_sigmas(20),
    gamma_0=0.0,
    noise_scale=1.0,
    step_scale=1.0,
    center=no_centering,
    augment=False,
    align=no_alignment,
)


@pytest.mark.parametrize(
    "sigmas, final",
    [
        (schedules.zero_last(schedules.karras(_grid(11))), 0.0),
        (schedules.append_zero(schedules.karras(_grid(10))), 0.0),
        (schedules.karras(_grid(11)), 16.0 * 4e-4),
    ],
    ids=["zero_last", "append_zero", "sigma_min"],
)
def test_karras_schedule_shape_and_ends(sigmas, final):
    assert sigmas.shape == (11,)
    assert sigmas.dtype == np.float32
    assert np.isclose(sigmas[0], 16.0 * 160.0)
    assert np.all(np.diff(sigmas) < 0)
    assert np.isclose(sigmas[-1], final)


def test_karras_single_step_is_finite():
    sigmas = schedules.append_zero(schedules.karras(_grid(1)))

    np.testing.assert_allclose(sigmas, [16.0 * 160.0, 0.0])


def test_dilate_fixes_endpoints_and_stays_monotone():
    t = np.linspace(0.0, 1.0, 101)
    warped = schedules.dilate(t, start=0.6, end=0.8, dilation=2.667)

    np.testing.assert_allclose(warped[[0, -1]], [0.0, 1.0], atol=1e-6)
    assert np.all(np.diff(warped) > 0)


def test_cap_starts_at_cap_and_shortens_schedule():
    sigmas = schedules.cap(schedules.append_zero(schedules.karras(_grid(14))), 256.0)

    assert sigmas[0] == 256.0
    assert np.all(sigmas[1:] <= 256.0)
    assert len(sigmas) - 1 < 14


def test_random_rotation_is_proper_orthogonal():
    rotations = jax.vmap(random_rotation)(jax.random.split(jax.random.key(1), 8))

    eye = jnp.broadcast_to(jnp.eye(3), rotations.shape)
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), eye, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.det(rotations), 1.0, atol=1e-5)


def test_masked_centering_ignores_padding():
    x = jnp.concatenate([jnp.ones((4, 3)), 100.0 * jnp.ones((2, 3))])
    mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    np.testing.assert_allclose(masked_centering(x, mask)[:4], 0.0, atol=1e-6)
    assert not np.allclose(mean_centering(x, mask)[:4], 0.0)


@pytest.mark.parametrize("align", [rigid_align, rigid_align_stop_gradient])
def test_rigid_align_recovers_rigid_motion(align):
    target = _target()
    rotation = random_rotation(jax.random.key(2))
    moved = jnp.matmul(target, rotation, precision="highest") + jnp.array([3.0, -1.0, 2.0])

    aligned = align(moved, target, jnp.ones(NUM_ATOMS))

    np.testing.assert_allclose(aligned, target, atol=1e-3)


def test_ode_recovers_point_target():
    target = _target()
    denoise = _point_denoiser(target)

    x = sample(denoise, MASK, ODE, key=jax.random.key(3))

    np.testing.assert_allclose(x, target, atol=1e-3)


@pytest.mark.parametrize("align", [rigid_align_stop_gradient, rigid_align])
def test_stochastic_sampler_recovers_target_up_to_rigid_motion(align):
    target = _target()
    denoise = _point_denoiser(target)
    sampler = Sampler(sigmas=_sigmas(20), align=align)

    xs = jax.vmap(lambda k: sample(denoise, MASK, sampler, key=k))(
        jax.random.split(jax.random.key(4), 3)
    )

    assert np.isfinite(np.asarray(xs)).all()
    for x in xs:
        assert _rmsd_after_alignment(x, target) < 0.05


def test_integrate_is_deterministic_given_noise():
    denoise = _point_denoiser(_target())
    sampler = Sampler(sigmas=_sigmas(10))
    noise = Noise.draw(jax.random.key(5), num_steps=sampler.num_steps, num_atoms=NUM_ATOMS)

    np.testing.assert_array_equal(
        integrate(denoise, MASK, sampler, noise),
        integrate(denoise, MASK, sampler, noise),
    )


def test_sampler_fields_are_replaceable():
    sampler = dataclasses.replace(ODE, gamma_0=0.8, sigmas=_sigmas(5))

    assert sampler.num_steps == 5
    assert sampler.gamma_0 == 0.8
    assert sampler.align is no_alignment


@pytest.mark.parametrize("align", [no_alignment, rigid_align_stop_gradient, rigid_align])
def test_gradient_flows_through_sampler(align):
    denoise = _point_denoiser(_target())
    sampler = Sampler(sigmas=_sigmas(10), align=align)
    noise = Noise.draw(jax.random.key(6), num_steps=sampler.num_steps, num_atoms=NUM_ATOMS)

    def loss(denoise):
        return (integrate(denoise, MASK, sampler, noise) ** 2).mean()

    (grad,) = eqx.filter_jit(eqx.filter_grad(loss))(denoise).args

    assert np.isfinite(np.asarray(grad)).all()
    assert float(jnp.abs(grad).max()) > 0.0
