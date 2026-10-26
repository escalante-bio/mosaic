"""OpenDDE structure module as a `Denoise` function.

OpenDDE denoises on its structural-token branch, so building the denoiser runs
`expand_to_structural_tokens` and hoists the step-invariant caches exactly as
`jopendde.diffusion.sample_diffusion` does.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from jopendde.diffusion import DiffusionModule
from jopendde.features import Features
from jopendde.model import OpenDDE

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoise, Sampler, mean_centering, no_alignment


def _denoise(diffusion_module: DiffusionModule, conditioning: dict, x, sigma):
    return diffusion_module(
        x_noisy=x[None], t_hat_noise_level=sigma[None].astype(x.dtype), **conditioning
    )[0]


def opendde_denoiser(
    model: OpenDDE, feat: Features, s_inputs, s, z
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the residue-branch trunk output (`OpenDDE.get_pairformer_output`).

    Returns the denoiser and the structural-token atom mask.
    """
    struct_feat, s_inputs, s, z, attn_bias = model.expand_to_structural_tokens(
        feat, s_inputs, s, z
    )
    module = model.diffusion_module
    pair_z = module.diffusion_conditioning.prepare_cache(struct_feat.relp, z)
    token_pair_bias = module.diffusion_transformer.precompute_pair_bias(
        pair_z, attn_bias
    )
    p_lm, c_l = module.atom_attention_encoder.prepare_cache(
        ref_pos=struct_feat.ref_pos,
        ref_charge=struct_feat.ref_charge,
        ref_mask=struct_feat.ref_mask,
        ref_atom_name_chars=struct_feat.ref_atom_name_chars,
        ref_element=struct_feat.ref_element,
        atom_to_token_idx=struct_feat.atom_to_token_idx,
        d_lm=struct_feat.d_lm,
        v_lm=struct_feat.v_lm,
        pad_info=struct_feat.pad_info,
        r_l=True,  # non-None gate for the token-pair-context term; value unused
        z=pair_z,
    )
    conditioning = {
        "feat": struct_feat,
        "s_inputs": s_inputs,
        "s_trunk": s,
        "z_trunk": z,
        "extra_attn_bias": attn_bias,
        # Step-invariant caches hoisted out of the loop, as in `sample_diffusion`.
        "pair_z": pair_z,
        "p_lm": p_lm,
        "c_l": c_l,
        "token_pair_bias": token_pair_bias,
    }
    denoise = eqx.Partial(_denoise, module, conditioning)
    return denoise, struct_feat.ref_mask.astype(jnp.float32)


def opendde_sampler(model: OpenDDE, num_steps: int) -> Sampler:
    """The `Sampler` reproducing `OpenDDE.sample_coordinates`."""
    sigmas = schedules.zero_last(
        schedules.karras(
            np.linspace(0.0, 1.0, num_steps + 1),
            sigma_data=model.sched_sigma_data,
            s_max=model.sched_s_max,
            s_min=model.sched_s_min,
            rho=model.sched_rho,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=model.gamma0,
        gamma_min=model.gamma_min,
        noise_scale=model.noise_scale_lambda,
        step_scale=model.step_scale_eta,
        center=mean_centering,
        align=no_alignment,
    )
