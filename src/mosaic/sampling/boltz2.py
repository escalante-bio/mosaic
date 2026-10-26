"""Boltz-2 structure module as a `Denoise` function."""

import equinox as eqx
import jax
import joltz
import numpy as np
from jaxtyping import Array, Float, PyTree

from mosaic.sampling import schedules
from mosaic.sampling.core import (
    Denoise,
    Sampler,
    mean_centering,
    no_alignment,
    rigid_align_stop_gradient,
)


def _denoise(structure_module: joltz.AtomDiffusion2, conditioning: PyTree, x, sigma):
    with jax.default_matmul_precision("float32"):
        # `key` is unused: the score model has no dropout.
        return structure_module.preconditioned_network_forward(
            x[None], sigma, conditioning, key=None
        )[0]


def boltz2_denoiser(
    model: joltz.Joltz2, features: PyTree, initial_embedding, trunk_state
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (see `mosaic.losses.boltz2.boltz2_trunk`).

    Returns the denoiser and the atom mask.
    """
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            trunk_state.s,
            trunk_state.z,
            initial_embedding.relative_position_encoding,
            features,
        )
    )
    conditioning = {
        "s_trunk": trunk_state.s,
        "s_inputs": initial_embedding.s_inputs,
        "feats": features,
        "multiplicity": 1,
        # Step-invariant, hoisted out of the loop.
        "diffusion_conditioning": {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        },
    }
    denoise = eqx.Partial(_denoise, model.structure_module, conditioning)
    return denoise, features["atom_pad_mask"][0]


def boltz2_sampler(
    structure_module: joltz.AtomDiffusion2, num_steps: int
) -> Sampler:
    """The `Sampler` reproducing `AtomDiffusion2.sample`."""
    m = structure_module
    sigmas = schedules.append_zero(
        schedules.karras(
            np.linspace(0.0, 1.0, num_steps),
            sigma_data=m.sigma_data,
            s_max=m.sigma_max,
            s_min=m.sigma_min,
            rho=m.rho,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=m.gamma_0,
        gamma_min=m.gamma_min,
        noise_scale=m.noise_scale,
        step_scale=m.step_scale,
        center=mean_centering,
        align=rigid_align_stop_gradient if m.alignment_reverse_diff else no_alignment,
    )

