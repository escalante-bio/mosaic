"""Mol-1 structure module as a `Denoise` function."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from jol1 import Features, InputEmbedding, Mol1, TrunkState

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoise, Sampler, masked_centering, no_alignment


def mol1_denoiser(
    model: Mol1, features: Features, inputs: InputEmbedding, state: TrunkState
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (`Mol1.embed`, `Mol1.trunk`); the step-invariant
    diffusion conditioning is computed here, once.

    Returns the denoiser and the atom mask.
    """
    context = model.diffusion_context(features, inputs, state)
    denoise = eqx.Partial(Mol1.denoise, model, context)
    return denoise, features.atoms.atom_mask.astype(jnp.float32)


def mol1_sampler(model: Mol1, num_steps: int) -> Sampler:
    """The `Sampler` of upstream Mol-1's `sample_diffusion` (AF3 Algorithm 18:
    re-centre, random rotation and translation, churn, Euler step; no alignment),
    with the schedule and churn parameters of the checkpoint's inference config.

    Upstream centres on the plain atom mean; it never pads atoms, so masked centring
    (which ignores the padding of a repacked binder) is the same."""
    settings = model.sampler_settings
    sigmas = schedules.zero_last(
        schedules.karras(
            np.linspace(0.0, 1.0, num_steps + 1),
            sigma_data=settings.sigma_data,
            s_max=settings.s_max,
            s_min=settings.s_min,
            rho=settings.rho,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=settings.gamma0,
        gamma_min=settings.gamma_min,
        noise_scale=settings.noise_scale_lambda,
        step_scale=settings.step_scale_eta,
        center=masked_centering,
        augment=True,
        align=no_alignment,
    )
