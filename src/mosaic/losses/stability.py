# Very simple *absolute* (delta G, ~not~ delta-delta G) stability model trained on top of frozen ESMC on the Megascale dataset
# Specifically this is trained to minimize MSE on the split described here: https://github.com/SimonKitSangChu/EsmTherm?tab=readme-ov-file
# Could almost certainly be improved but seems to work fine.


from pathlib import Path

import equinox as eqx
import jax
from jaxtyping import Array, Float
import jax.numpy as jnp

from ..common import LossTerm
from .esmc import boltz_to_esmc_matrix
from esmj import ESMC


class StabilityModel(LossTerm):
    esm: ESMC
    head: eqx.nn.MLP

    def __call__(
        self,
        seq_standard_tokens: Float[Array, "N 20"],
        *,
        key,
    ):
        # convert from standard tokenization to ESM tokenization
        # add cls and eos tokens
        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([self.esm.vocab["<cls>"]], 64),
                seq_standard_tokens @ boltz_to_esmc_matrix(self.esm),
                jax.nn.one_hot([self.esm.vocab["<eos>"]], 64),
            ]
        )
        # this isn't very accurate, so let's clip it
        x = esm_toks @ self.esm.embed.embedding.weight
        x, _ = self.esm.transformer(x[None])
        estimated_delta_g = self.head(x[0].mean(axis=0))
        estimated_delta_g = estimated_delta_g.clip(-10, 3)
        return -estimated_delta_g, {"delta_g": estimated_delta_g}  # sign error?

    @staticmethod
    def from_pretrained(esm: ESMC, path: Path = Path("stability.eqx")):
        head = eqx.nn.MLP(
            in_size=960,
            out_size="scalar",
            width_size=2 * 960,
            activation=jax.nn.elu,
            depth=1,
            key=jax.random.key(0),
        )
        return StabilityModel(esm, eqx.tree_deserialise_leaves(path, head))
