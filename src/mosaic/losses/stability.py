# Very simple *absolute* (delta G, ~not~ delta-delta G) stability model trained on top of frozen ESMC on the Megascale dataset
# Specifically this is trained to minimize MSE on the split described here: https://github.com/SimonKitSangChu/EsmTherm?tab=readme-ov-file
# Could almost certainly be improved but seems to work fine.


from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jfold.models.esmfold2 import ESMCForMaskedLM

from ..common import LossTerm
from .esmc import ESMC_VOCAB_SIZE, boltz_to_esmc_matrix, esmc_vocab


def _head(key) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=960,
        out_size="scalar",
        width_size=2 * 960,
        activation=jax.nn.elu,
        depth=1,
        key=key,
    )


class StabilityModel(LossTerm):
    esm: ESMCForMaskedLM  # ESMC-300M, from `load_esmc("esmc_300m")`
    head: eqx.nn.MLP

    def __call__(
        self,
        seq_standard_tokens: Float[Array, "N 20"],
        *,
        key,
    ):
        # convert from standard tokenization to ESM tokenization
        # add cls and eos tokens
        vocab = esmc_vocab()
        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([vocab["<cls>"]], ESMC_VOCAB_SIZE),
                seq_standard_tokens @ boltz_to_esmc_matrix(vocab),
                jax.nn.one_hot([vocab["<eos>"]], ESMC_VOCAB_SIZE),
            ]
        )
        n = esm_toks.shape[0]
        # The final normalized hidden state, pooled over every position.
        x = self.esm.esmc(esm_toks, jnp.zeros(n, int), jnp.ones(n, bool))[-1]
        # this isn't very accurate, so let's clip it
        estimated_delta_g = self.head(x.mean(axis=0)).clip(-10, 3)
        return -estimated_delta_g, {"delta_g": estimated_delta_g}  # sign error?

    @staticmethod
    def from_pretrained(esm: ESMCForMaskedLM, path: Path = Path("stability.eqx")):
        # Only the head's structure is needed to read its weights.
        skeleton = eqx.filter_eval_shape(_head, jax.eval_shape(jax.random.key, 0))
        return StabilityModel(esm, eqx.tree_deserialise_leaves(path, skeleton))
