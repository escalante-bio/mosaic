"""BioEmu2 structure module as a `Denoise` function."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from biojemu2 import BioEmu2, Dynamics, Features, StaticContext, TrunkEmbeddings
from jaxtyping import Array, Float

from mosaic.sampling import schedules
from mosaic.sampling.core import (
    Denoise,
    Sampler,
    masked_centering,
    rigid_align_stop_gradient,
)

# `EDMSampler.structure_prediction` / `EDMSampler.equilibrium`.
STEP_SCALE = {"structure": 1.5, "equilibrium": 1.0}


def _denoise(model: BioEmu2, context: StaticContext, dynamics: Dynamics, x, sigma):
    return model.denoise(context, x, sigma, dynamics)


def bioemu2_denoiser(
    model: BioEmu2, features: Features, trunk: TrunkEmbeddings
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk output (`BioEmu2.embed`); the step-invariant conditioning is
    computed here, once.

    Only unconditioned dynamics (equilibrium sampling, structure prediction) are
    supported: upstream moves the companion frame `coords_cond` with every random
    augmentation, which a `Denoise` cannot see.

    Returns the denoiser and the atom mask.
    """
    context = model.static_context(features, trunk)
    denoise = eqx.Partial(_denoise, model, context, features.dynamics)
    return denoise, features.structure.atom_mask.astype(jnp.float32)


def bioemu2_sampler(
    model: BioEmu2,
    num_steps: int,
    mode: str = "structure",
    *,
    schedule: schedules.Schedule | None = None,
) -> Sampler:
    """The `Sampler` reproducing `biojemu2.sample_from_noise` with upstream's preset for
    `mode` (``"structure"`` or ``"equilibrium"``). `schedule` defaults to upstream's
    `Karras`."""
    if schedule is None:
        schedule = schedules.Karras(rho=7.0)
    sigmas = schedules.append_zero(
        schedule(
            np.linspace(0.0, 1.0, num_steps),
            sigma_data=model.sigma_data,
            s_max=model.sigma_max,
            s_min=model.sigma_min,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.0,
        step_scale=STEP_SCALE[mode],
        center=masked_centering,
        align=rigid_align_stop_gradient,
    )
