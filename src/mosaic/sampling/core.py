"""The AF3-style (Algorithm 18) diffusion sampler: jfold's (`jfold.sampling`), with its
settings under mosaic's name `Sampler`.

A structure module enters as a `Denoise` function, an EDM x0-predictor `D(x, sigma)` with
its weights and conditioning bound; a `Sampler` holds the noise schedule, churn, frame
policy and update rule; `Noise` holds every random draw. A backend's
`structure_sampler(trunk, sampler)` binds the three for one trunk output.
"""

from jfold.sampling.core import (
    Denoise,
    Noise,
    euler_update,
    exponential_update,
    integrate,
    masked_centering,
    mean_centering,
    no_alignment,
    no_centering,
    random_rotation,
    rigid_align,
    rigid_align_stop_gradient,
    sample,
)
from jfold.sampling.core import SamplerSettings as Sampler
from jfold.sampling.structure import StructureSampler

__all__ = [
    "Denoise",
    "Noise",
    "Sampler",
    "StructureSampler",
    "euler_update",
    "exponential_update",
    "integrate",
    "masked_centering",
    "mean_centering",
    "no_alignment",
    "no_centering",
    "random_rotation",
    "rigid_align",
    "rigid_align_stop_gradient",
    "sample",
]
