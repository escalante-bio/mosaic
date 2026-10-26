"""ProMERA structure module as a `Denoise` function."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree
from jpromera.config import DIFFUSION
from jpromera.diffusion import AtomDiffusion
from jpromera.feats import Feats
from jpromera.model import JPromera

from mosaic.sampling import schedules
from mosaic.sampling.core import (
    Denoise,
    Sampler,
    mean_centering,
    rigid_align_stop_gradient,
)


def _denoise(structure_module: AtomDiffusion, conditioning: PyTree, x, sigma):
    return structure_module.preconditioned_network_forward(
        x[None], sigma, conditioning
    )["denoised_atom_coords"][0]


def promera_denoiser(
    model: JPromera, feats: Feats, out
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk output (`JPromera.fold`), hoisting the step-invariant cache.

    Returns the denoiser and the atom mask.
    """
    cache = model.structure_module.score_model.precompute_conditioning(
        s_trunk=out.s,
        z_trunk=out.z,
        relative_position_encoding=out.relpos,
        feats=feats,
    )
    conditioning = {
        "s_inputs": out.s_inputs,
        "s_trunk": out.s,
        "z_trunk": out.z,
        "relative_position_encoding": out.relpos,
        "feats": feats,
        "multiplicity": 1,
        "cache": cache,
    }
    denoise = eqx.Partial(_denoise, model.structure_module, conditioning)
    return denoise, feats.atom_pad_mask[0].astype(jnp.float32)


def promera_sampler(
    num_steps: int, diffusion_cfg: dict = DIFFUSION
) -> Sampler:
    """The `Sampler` reproducing `JPromera.sample`."""
    cfg = diffusion_cfg
    sigmas = schedules.karras(
        np.linspace(0.0, 1.0, num_steps + 1),
        sigma_data=cfg["sigma_data"],
        s_max=cfg["sigma_max"],
        s_min=cfg["sigma_min"],
        rho=cfg["rho"],
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=cfg["gamma_0"],
        gamma_min=cfg["gamma_min"],
        noise_scale=cfg["noise_scale"],
        step_scale=cfg["step_scale"],
        center=mean_centering,
        augment_after_noise=True,
        align=rigid_align_stop_gradient,
    )
