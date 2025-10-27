# This loss is not well tested; use at your own risk
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float
from ..common import LossTerm, TOKENS

from jablang import AbLang, from_torch
import ablang

def boltz_to_ablang_matrix(tokenizer):
    T = np.zeros((len(TOKENS), 24))
    for i, tok in enumerate(TOKENS):
        idx = tokenizer.vocab_to_token[tok]
        T[i, idx] = 1
    return T


def load_ablang(model_name: str = "heavy"):
    model = ablang.pretrained(model_name)
    model.freeze()
    return from_torch(model.AbLang), model.tokenizer


class AbLangPseudoLikelihood(LossTerm):
    model: AbLang
    tokenizer: ablang.tokenizers.ABtokenizer
    stop_grad: bool = True


    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]
        # convert from standard tokenization to ablang tokenization
        ablang_toks_unpadded = seq_standard_tokens @ boltz_to_ablang_matrix(self.tokenizer)

        # add cls and eos tokens (0 and 22)
        toks = jnp.concatenate(
            [
                jax.nn.one_hot([self.tokenizer.vocab_to_token["<"]], 24),
                ablang_toks_unpadded,
                jax.nn.one_hot([self.tokenizer.vocab_to_token[">"]], 24),
            ]
        )


        def single_ll(index: int):
            rep = self.model.rep
            head = self.model.head
            # replace token at index with mask
            masked_tokens = toks.at[index].set(jax.nn.one_hot(self.tokenizer.vocab_to_token["*"], 24))
            # embed and run ablang
            x = masked_tokens @ rep.embedding.AAEmbeddings.weight
            position_embeddings = rep.embedding.PositionEmbeddings.weight[:x.shape[0]]
            x = rep.embedding.LayerNorm(x + position_embeddings)

            x = self.model.rep.encoder(x[None])
            logits = head(x)[0]


            return jax.nn.log_softmax(logits[index])

        masked_log_likelihoods = jax.vmap(single_ll)(jnp.arange(start = 1, stop = n+1))
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        pll =  (masked_log_likelihoods * ablang_toks_unpadded).sum(-1).mean()
        return -pll, {"ablang_pll": pll}




        
