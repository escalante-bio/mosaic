"""ESMFold2 structure head as a `Denoise` function (release and Experimental share it)."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from esmjfold2.diffusion import DiffusionCache, DiffusionModule, DiffusionStructureHead
from jaxtyping import Array, Float

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoise, Sampler, masked_centering, rigid_align


def _denoise(diffusion_module: DiffusionModule, cache: DiffusionCache, x, sigma):
    return diffusion_module.denoise(
        cache, x[None], jnp.full((1,), sigma, dtype=jnp.float32)
    )[0]


def esmfold2_denoiser(
    structure_head: DiffusionStructureHead, ctx: eqx.Module, z
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk output (see `mosaic.losses.esmfold2.esmfold2_trunk`).

    Returns the denoiser and the atom mask.
    """
    module = structure_head.diffusion_module
    # Step-invariant conditioning, computed once.
    cache = module.prepare(
        ref_pos=ctx.ref_pos,
        ref_charge=ctx.ref_charge,
        ref_mask=ctx.atm_mask,
        ref_element_oh=ctx.ref_element_oh,
        ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh,
        ref_space_uid=ctx.ref_space_uid,
        tok_idx=ctx.atom_to_token,
        s_inputs=ctx.x_inputs,
        z_trunk=z,
        relative_position_encoding=ctx.relative_position_encoding,
        n_tokens=ctx.n_tokens,
        token_attention_mask=ctx.tok_mask,
    )
    denoise = eqx.Partial(_denoise, module, cache)
    return denoise, ctx.atm_mask[0].astype(jnp.float32)


def esmfold2_sampler(
    structure_head: DiffusionStructureHead,
    num_steps: int,
    *,
    max_sigma: float | None = 256.0,
) -> Sampler:
    """The `Sampler` reproducing `DiffusionStructureHead.sample`.

    The default `max_sigma` cap drops the high-noise steps, so fewer than
    `num_steps` denoiser calls run (e.g. 34 for 50).
    """
    h = structure_head
    sigmas = schedules.append_zero(
        schedules.karras(
            np.linspace(0.0, 1.0, num_steps),
            sigma_data=h.sigma_data,
            s_max=h.inference_s_max,
            s_min=h.inference_s_min,
            rho=h.inference_p,
        )
    )
    if max_sigma is not None:
        sigmas = schedules.cap(sigmas, max_sigma)
    return Sampler(
        sigmas=sigmas,
        gamma_0=h.gamma_0,
        gamma_min=h.gamma_min,
        noise_scale=h.noise_scale,
        step_scale=h.step_scale,
        center=masked_centering,
        align=rigid_align,
    )
