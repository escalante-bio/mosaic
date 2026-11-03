"""ESMFold2 feature pack, in-loop binder refresh and ESMC-sharing loss stacks."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from esmjfold2.esmc import ESMC, ESMCForMaskedLM
from jaxtyping import Array, Float

from mosaic.common import LossTerm
from mosaic.losses.transformations import norm_gradient
from mosaic.structure_prediction import MultiSampleLoss


def _with_shared_esmc(loss: MultiSampleLoss, esmc: ESMC) -> MultiSampleLoss:
    return eqx.tree_at(lambda l: l.model.esmc, loss, esmc, is_leaf=lambda x: x is None)


class StackedEsmFold2(LossTerm):
    """ Share the same ESMC model across multiple ESMFold models to save VRAM."""
    stacked: MultiSampleLoss
    static: MultiSampleLoss
    n: int = eqx.field(static=True)
    esmc: ESMCForMaskedLM  # only for its backbone (esmc.esmc); no PLL/lm_head use

    def __init__(self, structure_losses: list, esmc):
        if not structure_losses:
            raise ValueError("StackedEsmFold2 needs at least one structure loss.")
        for sl in structure_losses:
            if sl.model.esmc is not None:
                raise ValueError("loss.model.esmc must be None for StackedEsmFold2")

        dyn, stat = zip(*(eqx.partition(l, eqx.is_inexact_array) for l in structure_losses))
        self.stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *dyn)
        self.static = stat[0]
        self.n = len(structure_losses)
        self.esmc = esmc

    def __call__(self, sequence, *, key):
        k_pick, k_struct = jax.random.split(key, 2)
        i = jax.random.randint(k_pick, (), 0, self.n)
        model = eqx.combine(
            jax.tree_util.tree_map(lambda x: x[i], self.stacked), self.static
        )
        model = _with_shared_esmc(model, self.esmc.esmc)
        v_s, aux_s = model(norm_gradient(1.0, sequence), key=k_struct)
        return v_s, {"model_index": i, "": aux_s}


class StackedEsmFold2PLL(LossTerm):
    """One ESMC shared across a stack of (esmc-less) ESMFold2 structure losses
    plus an ESMC pseudo-perplexity term — a single ESMC buffer for the whole
    loss, to save VRAM.
    """

    # Stacked esmc-less structure losses (each leaf gains a leading axis n) and
    # the shared non-array part — same split as `RandomChoice`.
    stacked: MultiSampleLoss
    static: MultiSampleLoss
    n: int = eqx.field(static=True)
    # The single shared ESMC (backbone + MLM head): the only ESMC in the tree.
    esmc: ESMCForMaskedLM
    # PLL term built with `esm=None`; the shared ESMC is spliced in at call time.
    pll: LossTerm
    pll_weight: float

    def __init__(self, structure_losses, esmc, pll, pll_weight=0.15):
        if not structure_losses:
            raise ValueError("StackedEsmFold2PLL needs at least one structure loss.")
        struct = jax.tree_util.tree_structure(structure_losses[0])
        for i, l in enumerate(structure_losses[1:], 1):
            if jax.tree_util.tree_structure(l) != struct:
                raise ValueError(
                    f"structure_losses[{i}] differs in pytree structure from [0]."
                )
        dyn, stat = zip(
            *(eqx.partition(l, eqx.is_inexact_array) for l in structure_losses)
        )
        self.stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *dyn)
        self.static = stat[0]
        self.n = len(structure_losses)
        self.esmc = esmc
        self.pll = pll
        self.pll_weight = pll_weight

    def __call__(self, sequence: Float[Array, "N 20"], *, key):
        k_pick, k_struct, k_pll = jax.random.split(key, 3)
        i = jax.random.randint(k_pick, (), 0, self.n)
        model = eqx.combine(
            jax.tree_util.tree_map(lambda x: x[i], self.stacked), self.static
        )
        # Splice the shared backbone into the chosen structure loss
        # and the full shared ESMC into the (esm=None) PLL. Same object both
        # places → one buffer. `norm_gradient` norms each sub-term's gradient.
        model = _with_shared_esmc(model, self.esmc.esmc)
        v_s, aux_s = model(norm_gradient(1.0, sequence), key=k_struct)
        pll = eqx.tree_at(
            lambda p: p.esm, self.pll, self.esmc, is_leaf=lambda x: x is None
        )
        v_p, aux_p = pll(norm_gradient(self.pll_weight, sequence), key=k_pll)
        return v_s + self.pll_weight * v_p, {
            "model_index": i,
            "structure": aux_s,
            "pll": aux_p,
        }
