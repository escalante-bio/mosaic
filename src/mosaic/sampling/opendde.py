"""OpenDDE structure module as a `Denoiser`.

OpenDDE denoises on its structural-token branch, so building the denoiser runs
`expand_to_structural_tokens` and hoists the step-invariant caches exactly as
`jopendde.diffusion.sample_diffusion` does.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float
from jopendde.diffusion import DiffusionModule
from jopendde.features import Features
from jopendde.model import OpenDDE

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoiser, Sampler


class OpenDDEDenoiser(Denoiser):
    diffusion_module: DiffusionModule
    feat: Features  # structural-token branch
    s_inputs: Float[Array, "M Cs_in"]
    s_trunk: Float[Array, "M Cs"]
    z_trunk: Float[Array, "M M Cz"]
    extra_attn_bias: Float[Array, "M M"]
    # Step-invariant caches hoisted out of the loop, as in `sample_diffusion`.
    pair_z: Float[Array, "M M Cp"]
    p_lm: Float[Array, "1 K 32 128 P"]
    c_l: Float[Array, "A C"]
    token_pair_bias: Float[Array, "NB H M M"]
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        return self.diffusion_module(
            x_noisy=x[None],
            t_hat_noise_level=sigma[None].astype(x.dtype),
            feat=self.feat,
            s_inputs=self.s_inputs,
            s_trunk=self.s_trunk,
            z_trunk=self.z_trunk,
            pair_z=self.pair_z,
            p_lm=self.p_lm,
            c_l=self.c_l,
            extra_attn_bias=self.extra_attn_bias,
            token_pair_bias=self.token_pair_bias,
        )[0]


def opendde_denoiser(
    model: OpenDDE, feat: Features, s_inputs, s, z
) -> OpenDDEDenoiser:
    """Bind the residue-branch trunk output (`OpenDDE.get_pairformer_output`)."""
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
    return OpenDDEDenoiser(
        diffusion_module=module,
        feat=struct_feat,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        extra_attn_bias=attn_bias,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        token_pair_bias=token_pair_bias,
        atom_mask=struct_feat.ref_mask.astype(jnp.float32),
    )


def opendde_sampler(model: OpenDDE, num_steps: int) -> Sampler:
    """The `Sampler` reproducing `OpenDDE.sample_coordinates`."""
    sigmas = schedules.karras(
        num_steps,
        sigma_data=model.sched_sigma_data,
        s_max=model.sched_s_max,
        s_min=model.sched_s_min,
        rho=model.sched_rho,
        terminal="zero",
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=model.gamma0,
        gamma_min=model.gamma_min,
        noise_scale=model.noise_scale_lambda,
        step_scale=model.step_scale_eta,
        centering="mean",
        alignment="none",
    )
