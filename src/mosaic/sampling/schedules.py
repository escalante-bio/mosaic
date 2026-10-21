"""Noise schedules, built on the host as static float32 arrays of `num_steps + 1` sigmas.

All AF3-class models use the Karras schedule
`sigma(t) = sigma_data * (s_max^(1/rho) + t * (s_min^(1/rho) - s_max^(1/rho)))^rho`
on a grid `t ∈ [0, 1]`; they differ only in how the grid ends (`terminal`),
optional time warping (BoltzGen's dilation) and capping (ESMFold2).
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

Terminal = Literal["zero", "append_zero", "sigma_min"]


def karras(
    num_steps: int,
    *,
    sigma_data: float = 16.0,
    s_max: float = 160.0,
    s_min: float = 4e-4,
    rho: float = 7.0,
    terminal: Terminal = "zero",
    warp: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Karras schedule ending in one of three conventions:

    - `"zero"`: `num_steps + 1` grid points, last sigma replaced by 0 (AF3 paper).
    - `"append_zero"`: `num_steps` grid points, then 0 appended (EDM).
    - `"sigma_min"`: `num_steps + 1` grid points, ending at `sigma_data * s_min`.

    `warp` remaps the time grid before it is mapped to sigmas.
    """
    num_points = num_steps if terminal == "append_zero" else num_steps + 1
    t = np.linspace(0.0, 1.0, num_points)
    if warp is not None:
        t = warp(t)
    inv_rho = 1.0 / rho
    sigmas = sigma_data * (s_max**inv_rho + t * (s_min**inv_rho - s_max**inv_rho)) ** rho
    if terminal == "zero":
        sigmas[-1] = 0.0
    elif terminal == "append_zero":
        sigmas = np.append(sigmas, 0.0)
    return sigmas.astype(np.float32)


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
