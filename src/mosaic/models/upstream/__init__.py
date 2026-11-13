"""Featurize inputs that jfold cannot handle with each model's upstream package."""

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
    """Seed the Python and numpy global RNGs from `key` and restore them on exit.

    Upstream featurizers randomly place reference conformers using these global RNGs and
    take no seed, so this makes their output depend only on `key`. If `seed_torch` is true,
    torch's RNG is seeded too. If `key` is None, the RNGs are not seeded but are still
    restored. Yield the seed, or None if `key` is None."""
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
