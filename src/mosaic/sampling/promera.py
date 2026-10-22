"""ProMERA structure module as a `Denoiser`."""

import jax.numpy as jnp
from jaxtyping import Array, Float
from jpromera.config import DIFFUSION
from jpromera.diffusion import AtomDiffusion, DiffusionCache
from jpromera.feats import Feats
from jpromera.model import JPromera

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoiser, Sampler


class ProMERADenoiser(Denoiser):
    structure_module: AtomDiffusion
    s_inputs: Float[Array, "1 N Si"]
    s_trunk: Float[Array, "1 N S"]
    z_trunk: Float[Array, "1 N N Z"]
    relative_position_encoding: Float[Array, "1 N N Z"]
    feats: Feats
    cache: DiffusionCache  # step-invariant conditioning (`precompute_conditioning`)
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        conditioning = {
            "s_inputs": self.s_inputs,
            "s_trunk": self.s_trunk,
            "z_trunk": self.z_trunk,
            "relative_position_encoding": self.relative_position_encoding,
            "feats": self.feats,
            "multiplicity": 1,
            "cache": self.cache,
        }
        return self.structure_module.preconditioned_network_forward(
            x[None], sigma, conditioning
        )["denoised_atom_coords"][0]


def promera_denoiser(model: JPromera, feats: Feats, out) -> ProMERADenoiser:
    """Bind the trunk output (`JPromera.fold`), hoisting the step-invariant cache."""
    cache = model.structure_module.score_model.precompute_conditioning(
        s_trunk=out.s,
        z_trunk=out.z,
        relative_position_encoding=out.relpos,
        feats=feats,
    )
    return ProMERADenoiser(
        structure_module=model.structure_module,
        s_inputs=out.s_inputs,
        s_trunk=out.s,
        z_trunk=out.z,
        relative_position_encoding=out.relpos,
        feats=feats,
        cache=cache,
        atom_mask=feats.atom_pad_mask[0].astype(jnp.float32),
    )


def promera_sampler(
    num_steps: int, diffusion_cfg: dict = DIFFUSION
) -> Sampler:
    """The `Sampler` reproducing `JPromera.sample`."""
    cfg = diffusion_cfg
    sigmas = schedules.karras(
        num_steps,
        sigma_data=cfg["sigma_data"],
        s_max=cfg["sigma_max"],
        s_min=cfg["sigma_min"],
        rho=cfg["rho"],
        terminal="sigma_min",
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=cfg["gamma_0"],
        gamma_min=cfg["gamma_min"],
        noise_scale=cfg["noise_scale"],
        step_scale=cfg["step_scale"],
        centering="mean",
        augment_after_noise=True,
        alignment="stop_gradient",
    )
