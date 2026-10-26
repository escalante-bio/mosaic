"""One AF3-style (Algorithm 18) diffusion sampler for every AF3-class structure module.

Each model exposes its structure module as a `Denoise` function: an EDM
x0-predictor `D(x, sigma)` with its weights and conditioning bound (an
`eqx.Partial`, so they stay pytree leaves). A `Sampler` owns everything
else -- the noise schedule, churn, centring / random augmentation and rigid
alignment -- and `Noise` holds all randomness, so the per-model differences that
used to be baked into each model's loop become fields of a `Sampler`.

Everything is written for a single sample (`x: [A, 3]`, Å); use `jax.vmap` over
keys (or `Noise`) for several.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

Denoise = Callable[[Float[Array, "A 3"], Float[Array, ""]], Float[Array, "A 3"]]
"""EDM x0-predictor for one sample: maps noisy coordinates `x` (Å) at noise level
`sigma` to the denoised estimate, preconditioning (c_in, c_skip, c_out, c_noise)
included. Build it as an `eqx.Partial` binding the weights and conditioning."""

# Frame policies. `Sampler.center` maps `(x, mask) -> x`; `Sampler.align` maps
# `(x, target, mask) -> x`. They are static fields, so they must be module-level
# functions: a fresh `lambda` / `partial` per `Sampler` would miss the jit cache.


def no_centering(x: Float[Array, "A 3"], mask: Float[Array, " A"]) -> Float[Array, "A 3"]:
    return x


def mean_centering(x: Float[Array, "A 3"], mask: Float[Array, " A"]) -> Float[Array, "A 3"]:
    """Subtract the plain atom mean, padding included."""
    return x - x.mean(0)


def masked_centering(x: Float[Array, "A 3"], mask: Float[Array, " A"]) -> Float[Array, "A 3"]:
    """Subtract the mean over real (masked-in) atoms."""
    w = mask[:, None]
    return x - (x * w).sum(0) / jnp.maximum(w.sum(), 1.0)


def no_alignment(
    x: Float[Array, "A 3"], target: Float[Array, "A 3"], mask: Float[Array, " A"]
) -> Float[Array, "A 3"]:
    return x


def _kabsch(x, target, mask):
    """Weighted Kabsch fit of `x` onto `target`: (centred x, rotation, target centroid)."""
    w = mask[:, None]
    total = w.sum()
    x_c = x - (x * w).sum(0) / total
    target_mean = (target * w).sum(0) / total
    cov = jnp.einsum("ni,nj->ij", w * (target - target_mean), x_c, precision="highest")
    u, _, vh = jnp.linalg.svd(cov)
    flip = jnp.eye(3).at[2, 2].set(jnp.linalg.det(u @ vh))
    return x_c, u @ flip @ vh, target_mean


def _apply(x_c, rotation, translation):
    return jnp.einsum("ni,ji->nj", x_c, rotation, precision="highest") + translation


def rigid_align(
    x: Float[Array, "A 3"], target: Float[Array, "A 3"], mask: Float[Array, " A"]
) -> Float[Array, "A 3"]:
    """Rigidly move `x` onto `target`, with gradients through the fit."""
    return _apply(*_kabsch(x, target, mask))


def rigid_align_stop_gradient(
    x: Float[Array, "A 3"], target: Float[Array, "A 3"], mask: Float[Array, " A"]
) -> Float[Array, "A 3"]:
    """Rigidly move `x` onto `target`, without gradients through the fitted
    rotation or target centroid (gradients still flow through `x`'s centring)."""
    x_c, rotation, target_mean = _kabsch(x, target, mask)
    return _apply(x_c, *jax.lax.stop_gradient((rotation, target_mean)))


class Sampler(eqx.Module):
    """Stochastic-churn Euler sampler (AF3 Algorithm 18): schedule, churn and frame policy.

    `sigmas` is the noise schedule, `num_steps + 1` levels from high to low (see
    `mosaic.sampling.schedules`). Each step from `sigma` to `sigma_next` first
    raises the noise level to `t_hat = sigma * (1 + gamma)`, with
    `gamma = gamma_0` while `sigma_next > gamma_min` (else 0), adding
    `noise_scale * sqrt(t_hat² - sigma²)` Gaussian noise; then takes an Euler
    step scaled by `step_scale`. `gamma_0=0, noise_scale=1, step_scale=1` is the
    plain probability-flow ODE.

    Frame policy, applied every step:
      - `center`: `no_centering`, `mean_centering` or `masked_centering`.
      - `augment`: apply the step's random rotation and translation.
      - `augment_after_noise`: centre/augment after the churn noise is added
        rather than before.
      - `align`: before the Euler step, move the noisy sample onto the denoised
        one with `rigid_align` or `rigid_align_stop_gradient`, or `no_alignment`.

    Each model's `<model>_sampler(...)` returns the `Sampler` reproducing that
    model's own loop; adjust it with `dataclasses.replace`.
    """

    sigmas: Float[Array | np.ndarray, " T"]
    gamma_0: float | Float[Array, ""] = 0.8
    gamma_min: float | Float[Array, ""] = 1.0
    noise_scale: float | Float[Array, ""] = 1.003
    step_scale: float | Float[Array, ""] = 1.5
    center: Callable = eqx.field(default=masked_centering, static=True)
    augment: bool = eqx.field(default=True, static=True)
    augment_after_noise: bool = eqx.field(default=False, static=True)
    align: Callable = eqx.field(default=rigid_align_stop_gradient, static=True)

    @property
    def num_steps(self) -> int:
        return self.sigmas.shape[0] - 1


class Noise(eqx.Module):
    """Every random draw one sampling run consumes.

    `init` and `churn` are standard normal (scaled by the sampler), `rotations`
    are Haar-random and applied as `x @ R`, `translations` are standard normal.
    Passing these in explicitly makes `integrate` a deterministic function.
    """

    init: Float[Array, "A 3"]
    churn: Float[Array, "S A 3"]
    rotations: Float[Array, "S 3 3"]
    translations: Float[Array, "S 3"]

    @classmethod
    def draw(cls, key: PRNGKeyArray, *, num_steps: int, num_atoms: int) -> "Noise":
        k_init, k_churn, k_rot, k_trans = jax.random.split(key, 4)
        return cls(
            init=jax.random.normal(k_init, (num_atoms, 3)),
            churn=jax.random.normal(k_churn, (num_steps, num_atoms, 3)),
            rotations=jax.vmap(random_rotation)(jax.random.split(k_rot, num_steps)),
            translations=jax.random.normal(k_trans, (num_steps, 3)),
        )


def random_rotation(key: PRNGKeyArray) -> Float[Array, "3 3"]:
    """Haar-random rotation from a Gaussian quaternion."""
    r, i, j, k = jax.random.normal(key, (4,))
    s = 2.0 / (r * r + i * i + j * j + k * k)  # scale-invariant: no normalisation needed
    return jnp.stack(
        [
            jnp.stack([1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r)]),
            jnp.stack([s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r)]),
            jnp.stack([s * (i * k - j * r), s * (j * k + i * r), 1 - s * (i * i + j * j)]),
        ]
    )


@eqx.filter_jit  # `denoise` (weights + conditioning) is an argument, never a baked-in constant
def integrate(
    denoise: Denoise,
    atom_mask: Float[Array, " A"],
    sampler: Sampler,
    noise: Noise,
) -> Float[Array, "A 3"]:
    """Run `sampler` from `noise` using the given draws (deterministic).

    `atom_mask` marks real (non-padding) atoms for centring and alignment.
    """
    s = sampler
    gammas = jnp.where(s.sigmas[1:] > s.gamma_min, s.gamma_0, 0.0)

    def reframe(x, rotation, translation):
        x = s.center(x, atom_mask)
        return x @ rotation + translation if s.augment else x

    @jax.checkpoint
    def step(x, inputs):
        sigma, sigma_next, gamma, eps, rotation, translation = inputs
        if not s.augment_after_noise:
            x = reframe(x, rotation, translation)
        t_hat = sigma * (1 + gamma)
        x = x + s.noise_scale * jnp.sqrt(jnp.maximum(t_hat**2 - sigma**2, 0.0)) * eps
        if s.augment_after_noise:
            x = reframe(x, rotation, translation)
        x0 = denoise(x, t_hat)
        x = s.align(x, x0, atom_mask)
        x = x + s.step_scale * (sigma_next - t_hat) * (x - x0) / t_hat
        return x, None

    xs = (s.sigmas[:-1], s.sigmas[1:], gammas, noise.churn, noise.rotations, noise.translations)
    x, _ = jax.lax.scan(step, s.sigmas[0] * noise.init, xs)
    return x


def sample(
    denoise: Denoise,
    atom_mask: Float[Array, " A"],
    sampler: Sampler,
    *,
    key: PRNGKeyArray,
) -> Float[Array, "A 3"]:
    """Draw fresh `Noise` from `key` and `integrate`."""
    noise = Noise.draw(key, num_steps=sampler.num_steps, num_atoms=atom_mask.shape[0])
    return integrate(denoise, atom_mask, sampler, noise)
