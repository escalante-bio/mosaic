"""Featurization through each model's upstream (torch) package, for the inputs jfold's views do not take."""

import contextlib
import random
from collections.abc import Iterator

import jax
import numpy as np
from jaxtyping import PRNGKeyArray


@contextlib.contextmanager
def global_rngs(
    key: PRNGKeyArray | None, *, seed_torch: bool = False
) -> Iterator[int | None]:
    """Python's and numpy's global RNGs (and torch's, if `seed_torch`) seeded from `key`,
    and restored on exit; with `key` None they are only restored. Upstream featurizers pose
    each reference conformer randomly from these process-wide RNGs and take no seed, so
    this makes their features a function of `key` without disturbing the caller's RNGs.
    Yields the seed (None without `key`)."""
    seed = None if key is None else int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with contextlib.ExitStack() as stack:
        if seed_torch:
            import torch

            stack.enter_context(torch.random.fork_rng(devices=[]))
            if seed is not None:
                torch.manual_seed(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        try:
            yield seed
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
