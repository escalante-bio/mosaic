import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from mosaic.sampling import Denoiser, Sampler, draw_noise, integrate, sample, schedules
from mosaic.sampling.core import center, random_rotation, rigid_align

NUM_ATOMS = 32


class PointDenoiser(Denoiser):
    """The exact x0-predictor for data concentrated on a single point cloud."""

    target: Float[Array, "A 3"]
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        del x, sigma
        return self.target


def _target():
    return 10.0 * jax.random.normal(jax.random.key(0), (NUM_ATOMS, 3))


def _rmsd_after_alignment(x, target):
    mask = jnp.ones(x.shape[0])
    aligned = rigid_align(x, target, mask, stop_gradient=True)
    return float(jnp.sqrt(((aligned - target) ** 2).sum(-1).mean()))


ODE = Sampler(
    sigmas=schedules.karras(20),
    gamma_0=0.0,
    noise_scale=1.0,
    step_scale=1.0,
    centering="none",
    augment=False,
    alignment="none",
)


@pytest.mark.parametrize("terminal", ["zero", "append_zero", "sigma_min"])
def test_karras_schedule_shape_and_ends(terminal):
    sigmas = schedules.karras(10, terminal=terminal)

    assert sigmas.shape == (11,)
    assert sigmas.dtype == np.float32
    assert np.isclose(sigmas[0], 16.0 * 160.0)
    assert np.all(np.diff(sigmas) < 0)
    if terminal == "sigma_min":
        assert np.isclose(sigmas[-1], 16.0 * 4e-4)
    else:
        assert sigmas[-1] == 0.0


def test_karras_single_step_is_finite():
    sigmas = schedules.karras(1, terminal="append_zero")

    np.testing.assert_allclose(sigmas, [16.0 * 160.0, 0.0])


def test_dilate_fixes_endpoints_and_stays_monotone():
    t = np.linspace(0.0, 1.0, 101)
    warped = schedules.dilate(t, start=0.6, end=0.8, dilation=2.667)

    np.testing.assert_allclose(warped[[0, -1]], [0.0, 1.0], atol=1e-6)
    assert np.all(np.diff(warped) > 0)


def test_cap_starts_at_cap_and_shortens_schedule():
    sigmas = schedules.cap(schedules.karras(14, terminal="append_zero"), 256.0)

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

    np.testing.assert_allclose(center(x, mask, "masked")[:4], 0.0, atol=1e-6)
    assert not np.allclose(center(x, mask, "mean")[:4], 0.0)


@pytest.mark.parametrize("stop_gradient", [True, False])
def test_rigid_align_recovers_rigid_motion(stop_gradient):
    target = _target()
    rotation = random_rotation(jax.random.key(2))
    moved = jnp.matmul(target, rotation, precision="highest") + jnp.array([3.0, -1.0, 2.0])

    aligned = rigid_align(moved, target, jnp.ones(NUM_ATOMS), stop_gradient=stop_gradient)

    np.testing.assert_allclose(aligned, target, atol=1e-3)


def test_ode_recovers_point_target():
    target = _target()
    denoiser = PointDenoiser(target, jnp.ones(NUM_ATOMS))

    x = sample(denoiser, ODE, key=jax.random.key(3))

    np.testing.assert_allclose(x, target, atol=1e-3)


@pytest.mark.parametrize("alignment", ["stop_gradient", "full"])
def test_stochastic_sampler_recovers_target_up_to_rigid_motion(alignment):
    target = _target()
    denoiser = PointDenoiser(target, jnp.ones(NUM_ATOMS))
    sampler = Sampler(sigmas=schedules.karras(20), alignment=alignment)

    xs = jax.vmap(lambda k: sample(denoiser, sampler, key=k))(
        jax.random.split(jax.random.key(4), 3)
    )

    assert np.isfinite(np.asarray(xs)).all()
    for x in xs:
        assert _rmsd_after_alignment(x, target) < 0.05


def test_integrate_is_deterministic_given_noise():
    denoiser = PointDenoiser(_target(), jnp.ones(NUM_ATOMS))
    sampler = Sampler(sigmas=schedules.karras(10))
    noise = draw_noise(jax.random.key(5), num_steps=sampler.num_steps, num_atoms=NUM_ATOMS)

    np.testing.assert_array_equal(
        integrate(denoiser, sampler, noise), integrate(denoiser, sampler, noise)
    )


def test_sampler_fields_are_replaceable():
    sampler = dataclasses.replace(ODE, gamma_0=0.8, sigmas=schedules.karras(5))

    assert sampler.num_steps == 5
    assert sampler.gamma_0 == 0.8
    assert sampler.alignment == "none"


@pytest.mark.parametrize("alignment", ["none", "stop_gradient", "full"])
def test_gradient_flows_through_sampler(alignment):
    denoiser = PointDenoiser(_target(), jnp.ones(NUM_ATOMS))
    sampler = Sampler(sigmas=schedules.karras(10), alignment=alignment)
    noise = draw_noise(jax.random.key(6), num_steps=sampler.num_steps, num_atoms=NUM_ATOMS)

    def loss(denoiser):
        return (integrate(denoiser, sampler, noise) ** 2).mean()

    grad = eqx.filter_jit(eqx.filter_grad(loss))(denoiser)

    assert np.isfinite(np.asarray(grad.target)).all()
    assert float(jnp.abs(grad.target).max()) > 0.0
