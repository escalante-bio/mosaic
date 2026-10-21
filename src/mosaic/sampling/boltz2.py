"""Boltz-2 structure module as a `Denoiser`."""

import jax
import joltz
from jaxtyping import Array, Float, PyTree

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoiser, Sampler


class Boltz2Denoiser(Denoiser):
    structure_module: joltz.AtomDiffusion2
    conditioning: dict
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        with jax.default_matmul_precision("float32"):
            # `key` is unused: the score model has no dropout.
            return self.structure_module.preconditioned_network_forward(
                x[None], sigma, self.conditioning, key=None
            )[0]


def boltz2_denoiser(
    model: joltz.Joltz2, features: PyTree, initial_embedding, trunk_state
) -> Boltz2Denoiser:
    """Bind the trunk outputs (see `mosaic.losses.boltz2.boltz2_trunk`)."""
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            trunk_state.s,
            trunk_state.z,
            initial_embedding.relative_position_encoding,
            features,
        )
    )
    return Boltz2Denoiser(
        structure_module=model.structure_module,
        conditioning={
            "s_trunk": trunk_state.s,
            "s_inputs": initial_embedding.s_inputs,
            "feats": features,
            "multiplicity": 1,
            "diffusion_conditioning": {
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            },
        },
        atom_mask=features["atom_pad_mask"][0],
    )


def boltz2_sampler(
    structure_module: joltz.AtomDiffusion2, num_steps: int
) -> Sampler:
    """The `Sampler` reproducing `AtomDiffusion2.sample`."""
    m = structure_module
    sigmas = schedules.karras(
        num_steps,
        sigma_data=m.sigma_data,
        s_max=m.sigma_max,
        s_min=m.sigma_min,
        rho=m.rho,
        terminal="append_zero",
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=m.gamma_0,
        gamma_min=m.gamma_min,
        noise_scale=m.noise_scale,
        step_scale=m.step_scale,
        centering="mean",
        alignment="stop_gradient" if m.alignment_reverse_diff else "none",
    )

