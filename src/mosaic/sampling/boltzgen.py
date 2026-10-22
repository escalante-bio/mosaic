"""BoltzGen structure module as a `Denoiser`."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import joltzgen
from jaxtyping import Array, Float, PyTree

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoiser, Sampler


class BoltzGenDenoiser(Denoiser):
    structure_module: joltzgen.AtomDiffusion
    s_trunk: Float[Array, "1 N S"]
    s_inputs: Float[Array, "1 N Si"]
    feats: PyTree
    # Step-invariant conditioning from `joltzgen.JoltzGen.diffusion_conditioning`.
    q: Float[Array, "1 A C"]
    c: Float[Array, "1 A C"]
    to_keys: Callable
    atom_enc_bias: Float[Array, "1 K 32 128 H"]
    atom_dec_bias: Float[Array, "1 K 32 128 H"]
    token_trans_bias: Float[Array, "1 N N T"]
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        conditioning = {
            "s_trunk": self.s_trunk,
            "s_inputs": self.s_inputs,
            "feats": self.feats,
            "multiplicity": 1,
            "diffusion_conditioning": {
                "q": self.q,
                "c": self.c,
                "to_keys": self.to_keys,
                "atom_enc_bias": self.atom_enc_bias,
                "atom_dec_bias": self.atom_dec_bias,
                "token_trans_bias": self.token_trans_bias,
            },
        }
        # `key` is unused: the score model has no dropout.
        return self.structure_module.preconditioned_network_forward(
            x[None], sigma, conditioning, key=None
        )[0]


def boltzgen_denoiser(
    model: joltzgen.JoltzGen, features: PyTree, initial_embedding, trunk_state
) -> BoltzGenDenoiser:
    """Bind the trunk outputs (`model.embed_inputs` + `model.recycle`)."""
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            trunk_state.s,
            trunk_state.z,
            initial_embedding.relative_position_encoding,
            features,
        )
    )
    return BoltzGenDenoiser(
        structure_module=model.structure_module,
        s_trunk=trunk_state.s,
        s_inputs=initial_embedding.s_inputs,
        feats=features,
        q=q,
        c=c,
        to_keys=to_keys,
        atom_enc_bias=atom_enc_bias,
        atom_dec_bias=atom_dec_bias,
        token_trans_bias=token_trans_bias,
        atom_mask=features["atom_pad_mask"][0],
    )


def boltzgen_sampler(
    structure_module: joltzgen.AtomDiffusion,
    num_steps: int,
    *,
    step_scale: float,
    noise_scale: float,
    schedule: Literal["dilated", "af3"] = "dilated",
) -> Sampler:
    """The `Sampler` reproducing `AtomDiffusion.sample`.

    BoltzGen's `noise_scale` multiplies the churn noise twice (faithful to the
    torch reference), so the equivalent `Sampler.noise_scale` is its square.
    """
    m = structure_module
    warp = (
        partial(
            schedules.dilate,
            start=m.time_dilation_start,
            end=m.time_dilation_end,
            dilation=m.time_dilation,
        )
        if schedule == "dilated"
        else None
    )
    sigmas = schedules.karras(
        num_steps,
        sigma_data=m.sigma_data,
        s_max=m.sigma_max,
        s_min=m.sigma_min,
        rho=m.rho,
        terminal="append_zero",
        warp=warp,
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=m.gamma_0,
        gamma_min=m.gamma_min,
        noise_scale=noise_scale**2,
        step_scale=step_scale,
        centering="masked",
        alignment="stop_gradient" if m.alignment_reverse_diff else "none",
    )
