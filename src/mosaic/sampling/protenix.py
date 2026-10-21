"""Protenix structure module as a `Denoiser`."""

import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from protenix.protenij import DiffusionModel
from protenix.protenij import Protenix as Protenij

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoiser, Sampler


class ProtenixDenoiser(Denoiser):
    diffusion_module: DiffusionModel
    input_feature_dict: dict
    s_inputs: Float[Array, "N Cs_in"]
    s_trunk: Float[Array, "N Cs"]
    z_trunk: Float[Array, "N N Cz"]
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        return self.diffusion_module(
            x_noisy=x[None],
            t_hat_noise_level=sigma[None],
            input_feature_dict=self.input_feature_dict,
            s_inputs=self.s_inputs,
            s_trunk=self.s_trunk,
            z_trunk=self.z_trunk,
        )[0]


def protenix_denoiser(
    model: Protenij, features: PyTree, initial_embedding, trunk_state
) -> ProtenixDenoiser:
    """Bind the trunk outputs (see `mosaic.losses.protenix.get_trunk_state`)."""
    num_atoms = features["atom_to_token_idx"].shape[-1]
    return ProtenixDenoiser(
        diffusion_module=model.diffusion_module,
        input_feature_dict=features,
        s_inputs=initial_embedding.s_inputs,
        s_trunk=trunk_state.s,
        z_trunk=trunk_state.z,
        atom_mask=jnp.ones(num_atoms),  # Protenix featurizes without atom padding
    )


def protenix_sampler(model: Protenij, num_steps: int) -> Sampler:
    """The `Sampler` reproducing `Protenix.sample_structures`.

    Reads the churn parameters off the model, so this reproduces mosaic's
    ODE override (`mosaic.models.protenix.load_model`) when it is applied.
    """
    scheduler = model.inference_noise_scheduler
    sigmas = schedules.karras(
        num_steps,
        sigma_data=scheduler.sigma_data,
        s_max=scheduler.s_max,
        s_min=scheduler.s_min,
        rho=scheduler.rho,
        terminal="zero",
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=model.gamma0,
        gamma_min=model.gamma_min,
        noise_scale=model.noise_scale_lambda,
        step_scale=model.step_scale_eta,
        centering="mean",
        augment=False,
        alignment="none",
    )
