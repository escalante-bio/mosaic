"""BoltzGen structure module as a `Denoise` function."""

import equinox as eqx
import joltzgen
import numpy as np
from jaxtyping import Array, Float, PyTree

from mosaic.sampling import schedules
from mosaic.sampling.core import (
    Denoise,
    Sampler,
    masked_centering,
    no_alignment,
    rigid_align_stop_gradient,
)


def _denoise(structure_module: joltzgen.AtomDiffusion, conditioning: PyTree, x, sigma):
    # `key` is unused: the score model has no dropout.
    return structure_module.preconditioned_network_forward(
        x[None], sigma, conditioning, key=None
    )[0]


def boltzgen_denoiser(
    model: joltzgen.JoltzGen, features: PyTree, initial_embedding, trunk_state
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (see `mosaic.models.boltzgen.boltzgen_trunk`).

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


def boltzgen_sampler(
    structure_module: joltzgen.AtomDiffusion,
    num_steps: int,
    *,
    step_scale: float,
    noise_scale: float,
    dilate: bool = True,
) -> Sampler:
    """The `Sampler` reproducing `AtomDiffusion.sample`.

    `dilate` warps the time grid with BoltzGen's time dilation; without it the
    schedule is plain Karras (BoltzGen's `"af3"` schedule).

    BoltzGen's `noise_scale` multiplies the churn noise twice (faithful to the
    torch reference), so the equivalent `Sampler.noise_scale` is its square.
    """
    m = structure_module
    t = np.linspace(0.0, 1.0, num_steps)
    if dilate:
        t = schedules.dilate(
            t,
            start=m.time_dilation_start,
            end=m.time_dilation_end,
            dilation=m.time_dilation,
        )
    sigmas = schedules.append_zero(
        schedules.karras(
            t,
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
        noise_scale=noise_scale**2,
        step_scale=step_scale,
        center=masked_centering,
        align=rigid_align_stop_gradient if m.alignment_reverse_diff else no_alignment,
    )
