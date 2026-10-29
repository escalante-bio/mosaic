"""Protenix structure module as a `Denoise` function."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree
from protenix.protenij import DiffusionModel
from protenix.protenij import Protenix as Protenij

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoise, Sampler, masked_centering, no_alignment


def _denoise(diffusion_module: DiffusionModel, conditioning: dict, x, sigma):
    return diffusion_module(x_noisy=x[None], t_hat_noise_level=sigma[None], **conditioning)[0]


def protenix_denoiser(
    model: Protenij, features: PyTree, initial_embedding, trunk_state
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (see `mosaic.losses.protenix.get_trunk_state`).

    Returns the denoiser and the atom mask.
    """
    conditioning = {
        "input_feature_dict": features,
        "s_inputs": initial_embedding.s_inputs,
        "s_trunk": trunk_state.s,
        "z_trunk": trunk_state.z,
    }
    denoise = eqx.Partial(_denoise, model.diffusion_module, conditioning)
    num_atoms = features["atom_to_token_idx"].shape[-1]
    # Design packs pad the atom axis after refreshing the binder (`atom_pad_mask`).
    return denoise, jnp.asarray(features.get("atom_pad_mask", jnp.ones(num_atoms)), jnp.float32)


def protenix_sampler(model: Protenij, num_steps: int) -> Sampler:
    """The `Sampler` of upstream Protenix's `sample_diffusion`, with the
    schedule and churn parameters read off the model.

    Upstream re-centres and randomly rotates and translates every step; the JAX
    port only re-centres.
    """
    scheduler = model.inference_noise_scheduler
    sigmas = schedules.zero_last(
        schedules.karras(
            np.linspace(0.0, 1.0, num_steps + 1),
            sigma_data=scheduler.sigma_data,
            s_max=scheduler.s_max,
            s_min=scheduler.s_min,
            rho=scheduler.rho,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=model.gamma0,
        gamma_min=model.gamma_min,
        noise_scale=model.noise_scale_lambda,
        step_scale=model.step_scale_eta,
        center=masked_centering,
        augment=True,
        align=no_alignment,
    )
