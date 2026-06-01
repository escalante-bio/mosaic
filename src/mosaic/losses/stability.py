# Very simple *absolute* (delta G, ~not~ delta-delta G) stability model trained on top of frozen ESMC on the Megascale dataset
# Specifically this is trained to minimize MSE on the split described here: https://github.com/SimonKitSangChu/EsmTherm?tab=readme-ov-file
# Could almost certainly be improved but seems to work fine.


from pathlib import Path

import equinox as eqx
import jax
from jaxtyping import Array, Float
import jax.numpy as jnp

from ..common import LossTerm
from .esmc import ESMC_VOCAB_SIZE, boltz_to_esmc_matrix, esmc_vocab
from esmjfold2.esmc import ESMCForMaskedLM


class StabilityModel(LossTerm):
    esm: ESMCForMaskedLM
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
        # this isn't very accurate, so let's clip it
        x = esm_toks @ self.esm.esmc.embed.weight
        x, _ = self.esm.esmc.transformer(x[None], None, collect_hidden_states=False)
        estimated_delta_g = self.head(x[0].mean(axis=0))
        estimated_delta_g = estimated_delta_g.clip(-10, 3)
        return -estimated_delta_g, {"delta_g": estimated_delta_g}  # sign error?

    @staticmethod
    def from_pretrained(esm: ESMCForMaskedLM, path: Path = Path("stability.eqx")):
        head = eqx.nn.MLP(
            in_size=960,
            out_size="scalar",
            width_size=2 * 960,
            activation=jax.nn.elu,
            depth=1,
            key=jax.random.key(0),
        )
        return StabilityModel(esm, eqx.tree_deserialise_leaves(path, head))
