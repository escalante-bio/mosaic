"""OpenFold3 structure module as a `Denoiser`."""

import jax
from jaxtyping import Array, Float
from jopenfold3.batch import Batch
from jopenfold3.model import InitialEmbedding, OpenFold3, TrunkEmbedding
from jopenfold3.structure.diffusion_module import DiffusionModule

from mosaic.sampling import schedules
from mosaic.sampling.core import Denoiser, Sampler


class OF3Denoiser(Denoiser):
    diffusion_module: DiffusionModule
    batch: Batch  # with the sample axis inserted: leaves are [1, 1, ...]
    si_input: Float[Array, "1 1 N Si"]
    si_trunk: Float[Array, "1 1 N S"]
    zij_trunk: Float[Array, "1 1 N N Z"]
    atom_mask: Float[Array, " A"]

    def __call__(self, x, sigma):
        return self.diffusion_module(
            batch=self.batch,
            xl_noisy=x[None, None],
            token_mask=self.batch.token_mask,
            atom_mask=self.batch.atom_mask,
            t=sigma,
            si_input=self.si_input,
            si_trunk=self.si_trunk,
            zij_trunk=self.zij_trunk,
            use_conditioning=True,
            # Only split and forwarded: the atom/token transformers are
            # deterministic, so any fixed key gives the same output.
            key=jax.random.key(0),
        )[0, 0]


def of3_denoiser(
    model: OpenFold3, batch: Batch, init_emb: InitialEmbedding, trunk_emb: TrunkEmbedding
) -> OF3Denoiser:
    """Bind the trunk outputs (`OpenFold3.run_trunk`)."""
    return OF3Denoiser(
        # The copy `sample_structures` uses (the top-level one is a duplicate).
        diffusion_module=model.sample_diffusion.diffusion_module,
        batch=batch.expand_sample_dim(),
        si_input=init_emb.s_input[:, None],
        si_trunk=trunk_emb.s[:, None],
        zij_trunk=trunk_emb.z[:, None],
        atom_mask=batch.atom_mask[0],
    )


def of3_sampler(model: OpenFold3, num_steps: int) -> Sampler:
    """The `Sampler` of upstream OpenFold3's `SampleDiffusion`, with the
    schedule and churn parameters read off the model.

    Upstream re-centres (masked) and randomly rotates and translates every step;
    the JAX port does neither.
    """
    sampler = model.sample_diffusion
    sigmas = schedules.karras(
        num_steps,
        sigma_data=model.noise_sigma_data,
        s_max=model.noise_s_max,
        s_min=model.noise_s_min,
        rho=model.noise_p,
        terminal="sigma_min",
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=sampler.gamma_0,
        gamma_min=sampler.gamma_min,
        noise_scale=sampler.noise_scale,
        step_scale=sampler.step_scale,
        centering="masked",
        augment=True,
        alignment="none",
    )
