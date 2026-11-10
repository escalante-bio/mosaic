"""Model-agnostic diffusion sampling for AF3-class structure modules (jfold's sampler).

Each backend binds its structure module, conditioned on a trunk output, into a
`StructureSampler` (`structure_sampler(trunk, sampler)`); `sample(key)` draws a structure.
"""

from mosaic.sampling import schedules
from mosaic.sampling.core import (
    Denoise,
    Noise,
    Sampler,
    StructureSampler,
    euler_update,
    exponential_update,
    integrate,
    masked_centering,
    mean_centering,
    no_alignment,
    no_centering,
    rigid_align,
    rigid_align_stop_gradient,
    sample,
)

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
    "rigid_align",
    "rigid_align_stop_gradient",
    "sample",
    "schedules",
]
