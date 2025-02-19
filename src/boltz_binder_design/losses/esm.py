import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..esm import ESM2
from esm2quinox._esm2 import _alphabet as ESM_TOKENS
from ..common import LossTerm, TOKENS


def boltz_to_esm_matrix():
    """Converts from standard tokenization (Boltz ... plus two???) to ESM2QUINOX tokenization"""
    T = np.zeros((len(TOKENS), len(ESM_TOKENS)))
    for i, tok in enumerate(TOKENS):
        esm_idx = ESM_TOKENS[tok]
        T[i, esm_idx] = 1
    return T

class ESM2PseudoLikelihood(LossTerm):
    esm: ESM2
    stop_grad: bool = True

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]
        # convert from standard tokenization to ESM tokenization
        esm_toks_unpadded = seq_standard_tokens @ boltz_to_esm_matrix()
        # add cls and eos tokens
        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([0], 33),
                esm_toks_unpadded,
                jax.nn.one_hot([2], 33),
            ]
        )

        def single_ll(index: int):
            # replace token at index with mask
            masked_tokens = esm_toks.at[index].set(jax.nn.one_hot(ESM_TOKENS["m"], 33))
            # embed and run ESM
            embedding = masked_tokens @ self.esm.embedding.weight
            # set masked token embedding to zero
            embedding = embedding.at[index].set(0.0)
            # rescale to account for masking during ESM training
            mask_ratio_train = 0.15 * 0.8
            embedding = embedding * ((1 - mask_ratio_train) / (1 - 1/(n+2)))
            # apply ESM trunk and LM head
            embedding = self.esm._apply_trunk(embedding, np.zeros(n + 2))
            return jax.nn.log_softmax(self.esm.logit_head(embedding[index]))

        masked_log_likelihoods = jax.vmap(single_ll)(jnp.arange(start = 1, stop = n+1))
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        pll =  (masked_log_likelihoods * esm_toks_unpadded).sum(-1).mean()
        return -pll, {"esm_pll": pll}



