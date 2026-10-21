"""ESMFold2 structure head as a `Denoiser` (release and Experimental share it)."""

import equinox as eqx
import jax.numpy as jnp
from esmjfold2.diffusion import DiffusionCache, DiffusionModule, DiffusionStructureHead
from jaxtyping import Array, Float

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoiser, Sampler


class ESMFold2Denoiser(Denoiser):
    diffusion_module: DiffusionModule
    cache: DiffusionCache  # step-invariant conditioning, computed once by `prepare`
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        return self.diffusion_module.denoise(
            self.cache, x[None], jnp.full((1,), sigma, dtype=jnp.float32)
        )[0]


def esmfold2_denoiser(
    structure_head: DiffusionStructureHead, ctx: eqx.Module, z
) -> ESMFold2Denoiser:
    """Bind the trunk output (see `mosaic.losses.esmfold2.esmfold2_trunk`)."""
    module = structure_head.diffusion_module
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
    return ESMFold2Denoiser(
        diffusion_module=module,
        cache=cache,
        atom_mask=ctx.atm_mask[0].astype(jnp.float32),
    )


def esmfold2_sampler(
    structure_head: DiffusionStructureHead,
    num_steps: int,
    *,
    max_sigma: float | None = 256.0,
) -> Sampler:
    """The `Sampler` reproducing `DiffusionStructureHead.sample`.

    The default `max_sigma` cap drops the high-noise steps, so fewer than
    `num_steps` denoiser calls run (10 for the default 14).
    """
    h = structure_head
    sigmas = schedules.karras(
        num_steps,
        sigma_data=h.sigma_data,
        s_max=h.inference_s_max,
        s_min=h.inference_s_min,
        rho=h.inference_p,
        terminal="append_zero",
    )
    if max_sigma is not None:
        sigmas = schedules.cap(sigmas, max_sigma)
    return Sampler(
        sigmas=sigmas,
        gamma_0=h.gamma_0,
        gamma_min=h.gamma_min,
        noise_scale=h.noise_scale,
        step_scale=h.step_scale,
        centering="masked",
        alignment="full",
    )
