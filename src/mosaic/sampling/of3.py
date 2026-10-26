"""OpenFold3 structure module as a `Denoise` function."""

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float
from jopenfold3.batch import Batch
from jopenfold3.model import InitialEmbedding, OpenFold3, TrunkEmbedding
from jopenfold3.structure.diffusion_module import DiffusionModule

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoise, Sampler, masked_centering, no_alignment


def _denoise(
    diffusion_module: DiffusionModule,
    batch: Batch,
    si_input: Float[Array, "1 1 N Si"],
    si_trunk: Float[Array, "1 1 N S"],
    zij_trunk: Float[Array, "1 1 N N Z"],
    x,
    sigma,
):
    return diffusion_module(
        batch=batch,
        xl_noisy=x[None, None],
        token_mask=batch.token_mask,
        atom_mask=batch.atom_mask,
        t=sigma,
        si_input=si_input,
        si_trunk=si_trunk,
        zij_trunk=zij_trunk,
        use_conditioning=True,
        # Only split and forwarded: the atom/token transformers are
        # deterministic, so any fixed key gives the same output.
        key=jax.random.key(0),
    )[0, 0]


def of3_denoiser(
    model: OpenFold3, batch: Batch, init_emb: InitialEmbedding, trunk_emb: TrunkEmbedding
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (`OpenFold3.run_trunk`).

    Returns the denoiser and the atom mask.
    """
    denoise = eqx.Partial(
        _denoise,
        # The copy `sample_structures` uses (the top-level one is a duplicate).
        model.sample_diffusion.diffusion_module,
        batch.expand_sample_dim(),  # leaves are [1, 1, ...]
        init_emb.s_input[:, None],
        trunk_emb.s[:, None],
        trunk_emb.z[:, None],
    )
    return denoise, batch.atom_mask[0]


def of3_sampler(model: OpenFold3, num_steps: int) -> Sampler:
    """The `Sampler` of upstream OpenFold3's `SampleDiffusion`, with the
    schedule and churn parameters read off the model.

    Upstream re-centres (masked) and randomly rotates and translates every step;
    the JAX port does neither.
    """
    sampler = model.sample_diffusion
    sigmas = schedules.karras(
        np.linspace(0.0, 1.0, num_steps + 1),
        sigma_data=model.noise_sigma_data,
        s_max=model.noise_s_max,
        s_min=model.noise_s_min,
        rho=model.noise_p,
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=sampler.gamma_0,
        gamma_min=sampler.gamma_min,
        noise_scale=sampler.noise_scale,
        step_scale=sampler.step_scale,
        center=masked_centering,
        augment=True,
        align=no_alignment,
    )
