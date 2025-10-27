import jax
import numpy as np
from esm.models.esmc import ESMC as TORCH_ESMC
from esmj import ESMC, from_torch
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..common import TOKENS, LossTerm


def load_esmc(model_name: str = "esmc_300m"):
    return from_torch(TORCH_ESMC.from_pretrained(model_name).to("cpu"))


def boltz_to_esmc_matrix(esmc: ESMC):
    """Converts from standard tokenization (Boltz ... plus two???) to ESM-C tokenization"""
    T = np.zeros((len(TOKENS), 64))
    for i, tok in enumerate(TOKENS):
        esm_idx = esmc.vocab[tok]
        T[i, esm_idx] = 1
    return T


class ESMCPseudoLikelihood(LossTerm):
    """
    Pseudo-likelihood for the ESM-C masked language model

    Usage:
        from esmj import from_torch
        from esm.models.esmc import ESMC as TORCH_ESMC

        esm = from_torch(TORCH_ESMC.from_pretrained("esmc_300m").to("cpu"))

        ESMCPLL = ESMCPseudoLikelihood(esm)
    """

    esm: ESMC
    stop_grad: bool = True

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]
        # convert from standard tokenization to ESM tokenization
        esm_toks_unpadded = seq_standard_tokens @ boltz_to_esmc_matrix(self.esm)
        # add cls and eos tokens
        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([self.esm.vocab["<cls>"]], 64),
                esm_toks_unpadded,
                jax.nn.one_hot([self.esm.vocab["<eos>"]], 64),
            ]
        )

        def single_ll(index: int):
            # replace token at index with mask
            masked_tokens = esm_toks.at[index].set(
                jax.nn.one_hot(self.esm.vocab["<mask>"], 64)
            )
            # embed and run ESM
            x = masked_tokens @ self.esm.embed.embedding.weight
            x, _ = self.esm.transformer(x[None])
            logits = self.esm.sequence_head(x)[0]
            return jax.nn.log_softmax(logits[index])

        masked_log_likelihoods = jax.vmap(single_ll)(jnp.arange(start=1, stop=n + 1))
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        pll = (masked_log_likelihoods * esm_toks_unpadded).sum(-1).mean()
        return -pll, {"esmc_pll": pll}
