# uses precomputed n-gram frequencies from uniref50 to compute log-likelihoods under a trigram model.
# inspired by the excellent paper `Language models generalize beyond natural proteins`: https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1
# practically this tends to discourage the homopolymer stretches that ESM2 pseudo-likelihood really likes.

from pathlib import Path
import pickle
import numpy as onp

from jaxtyping import Array, Float
from jax import numpy as jnp, vmap
import jax

from tqdm import tqdm

from boltz.data.const import prot_letter_to_token
from ..common import LossTerm, TOKENS


def load_trigram_frequencies(path: Path = "../../../trigram_seg.pkl"):
    with open(path, "rb") as f:
        ngram_dict = pickle.load(f)

    N_TOKENS = len(TOKENS)
    P = onp.zeros((N_TOKENS, N_TOKENS, N_TOKENS))

    for trimer, freq in tqdm(ngram_dict.items()):
        one_letter_codes = list(trimer)
        if all(
            c in TOKENS
            for c in one_letter_codes
        ):
            i, j, k = [TOKENS.index(c) for c in one_letter_codes]
            P[i, j, k] = freq


    return P


class TrigramLL(LossTerm):
    log_probabilities: Float[Array, "20 20 20"]
    stop_grad: bool = False

    def __call__(self, soft_sequence: Float[Array, "N 20"], key):
        # I *think* this is the expected log likelihood of the soft sequence (under the trigram model) if each position is independent
        # e.g. if s_i ~ Categorical(p = soft_sequence[i]), then this should be E_s[\sum_i log p(s_i | s_{i-1}, s_{i-2})]
        def eval_single_position(i: int):
            x_i = soft_sequence[i]
            x_j = soft_sequence[i + 1]
            x_k = soft_sequence[i + 2]
            if self.stop_grad:
                x_i = jax.lax.stop_gradient(x_i)
                x_j = jax.lax.stop_gradient(x_j)

            return jnp.einsum(
                "i,j,k,ijk->",
                x_i,
                x_j,
                x_k,
                self.log_probabilities,
            )

        ave_log_prob = (
            vmap(eval_single_position)(jnp.arange(soft_sequence.shape[0] - 2))
        ).mean()

        return -ave_log_prob, {"trigram_ll": ave_log_prob}

    @staticmethod
    def from_pkl(path: Path = "trigram_seg.pkl", stop_grad = False):
        frequencies = load_trigram_frequencies(path)
        frequencies = onp.clip(frequencies, 1E-5, 1.0)
        trigram_conditional_probabilities = frequencies / frequencies.sum(-1, keepdims=True)
        return TrigramLL(onp.log(trigram_conditional_probabilities), stop_grad=stop_grad)
