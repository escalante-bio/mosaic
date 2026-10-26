"""Noise schedules, built on the host as static float32 arrays of sigmas.

All AF3-class models use the Karras schedule
`sigma(t) = sigma_data * (s_max^(1/rho) + t * (s_min^(1/rho) - s_max^(1/rho)))^rho`
on a time grid `t ∈ [0, 1]`. They differ only in the grid (`np.linspace`,
optionally warped by BoltzGen's `dilate`) and in how the schedule ends:

- `append_zero(karras(np.linspace(0, 1, n)))`: EDM / Boltz.
- `zero_last(karras(np.linspace(0, 1, n + 1)))`: the AF3 paper.
- `karras(np.linspace(0, 1, n + 1))`: end at `sigma_data * s_min`.

`cap` (ESMFold2) drops the high-noise steps.
"""

import numpy as np


def karras(
    t: np.ndarray,
    *,
    sigma_data: float = 16.0,
    s_max: float = 160.0,
    s_min: float = 4e-4,
    rho: float = 7.0,
) -> np.ndarray:
    """Karras sigmas at times `t`: `t=0` gives `sigma_data * s_max`, `t=1` gives `sigma_data * s_min`."""
    inv_rho = 1.0 / rho
    sigmas = sigma_data * (s_max**inv_rho + t * (s_min**inv_rho - s_max**inv_rho)) ** rho
    return sigmas.astype(np.float32)


def append_zero(sigmas: np.ndarray) -> np.ndarray:
    """End with an extra final step to sigma 0 (EDM)."""
    return np.append(sigmas, 0.0).astype(np.float32)


def zero_last(sigmas: np.ndarray) -> np.ndarray:
    """Replace the last sigma with 0 (AF3 paper)."""
    return np.append(sigmas[:-1], 0.0).astype(np.float32)


def dilate(t: np.ndarray, *, start: float, end: float, dilation: float) -> np.ndarray:
    """Piecewise-linear time warp spending `dilation`x more of the grid on [start, end]."""
    width, lower, upper = end - start, start, 1 - end
    assert (dilation - 1) * width <= lower + upper, "dilation too large"
    ratio = (lower + upper + (1 - dilation) * width) / (lower + upper)
    lower_prime, width_prime = lower * ratio, width * dilation
    return np.select(
        [t < lower_prime, t < lower_prime + width_prime],
        [t / ratio, (t - lower_prime) / dilation + lower],
        (t - (lower_prime + width_prime)) / ratio + lower + width,
    )


def cap(sigmas: np.ndarray, max_sigma: float) -> np.ndarray:
    """Drop sigmas above `max_sigma` and start from `max_sigma` instead.

    Shortens the schedule: the step count is whatever survives the cap.
    """
    return np.concatenate([[max_sigma], sigmas[sigmas <= max_sigma]]).astype(np.float32)
