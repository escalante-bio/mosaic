"""ESM-C pseudo-likelihood loss, backed by esmjfold2's ESMC port.

`Biohub/ESMC-300M` matches EvolutionaryScale's `esmc_300m` to fp tolerance, so
this reuses esmjfold2's ESMC instead of a second port. Loading goes through
transformers + esmjfold2's `from_torch`, i.e. torch + the Biohub transformers
fork (both are dependencies) are needed at load time.
"""

import functools
import gc

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float

import esmjfold2
from esmjfold2.esmc import ESMCForMaskedLM

from ..common import TOKENS, LossTerm

ESMC_VOCAB_SIZE = 64
_CHECKPOINTS = {
    "esmc_300m": "Biohub/ESMC-300M",
    "esmc_600m": "Biohub/ESMC-600M",
    "esmc_6b": "Biohub/ESMC-6B",
}


def esmc_vocab() -> dict[str, int]:
    """ESM-C token → embedding-row index, from the esm tokenizer."""
    from esm.tokenization import EsmSequenceTokenizer

    return EsmSequenceTokenizer().get_vocab()


def load_esmc(model_name: str = "esmc_300m", *, dtype=None) -> ESMCForMaskedLM:
    """Load ESM-C (backbone + MLM head) as an Equinox `ESMCForMaskedLM`.

    `model_name` is an alias (`esmc_300m` / `esmc_600m` / `esmc_6b`) or a raw
    HuggingFace id. `dtype` is the torch load precision (default fp32); the
    converter upcasts bf16 → fp32, so only fp16 actually shrinks the JAX model.
    Loads via torch + the Biohub transformers fork (both are dependencies).
    """
    import torch
    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM as TorchESMC

    checkpoint = _CHECKPOINTS.get(model_name, model_name)
    torch_model = TorchESMC.from_pretrained(
        checkpoint, dtype=dtype or torch.float32
    ).eval()
    model = esmjfold2.from_torch(torch_model)
    del torch_model
    gc.collect()
    return model


def boltz_to_esmc_matrix(vocab: dict[str, int]) -> np.ndarray:
    """One-hot map from standard (mosaic `TOKENS`) order to ESM-C tokenization."""
    T = np.zeros((len(TOKENS), ESMC_VOCAB_SIZE), dtype=np.float32)
    for i, tok in enumerate(TOKENS):
        T[i, vocab[tok]] = 1.0
    return T


class ESMCPseudoLikelihood(LossTerm):
    """Pseudo-likelihood for the ESM-C masked language model.

    Usage:
        from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc

        ESMCPLL = ESMCPseudoLikelihood(load_esmc())
    """

    esm: ESMCForMaskedLM
    stop_grad: bool = True

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]
        vocab = esmc_vocab()
        # standard tokenization → ESM tokenization, then add cls/eos
        esm_toks_unpadded = seq_standard_tokens @ boltz_to_esmc_matrix(vocab)
        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([vocab["<cls>"]], ESMC_VOCAB_SIZE),
                esm_toks_unpadded,
                jax.nn.one_hot([vocab["<eos>"]], ESMC_VOCAB_SIZE),
            ]
        )

        def single_ll(index):
            masked_tokens = esm_toks.at[index].set(
                jax.nn.one_hot(vocab["<mask>"], ESMC_VOCAB_SIZE)
            )
            # Embed by matmul rather than the integer lookup in
            # `ESMCForMaskedLM.__call__`, so gradients reach the soft sequence.
            x = masked_tokens @ self.esm.esmc.embed.weight
            x, _ = self.esm.esmc.transformer(
                x[None], None, collect_hidden_states=False
            )
            logits = self.esm.lm_head(x)[0]
            return jax.nn.log_softmax(logits[index])

        masked_log_likelihoods = jax.vmap(single_ll)(jnp.arange(start=1, stop=n + 1))
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        pll = (masked_log_likelihoods * esm_toks_unpadded).sum(-1).mean()
        return -pll, {"esmc_pll": pll}



class ESMCPseudoPerplexity(LossTerm):
    """Pseudoperplexity loss from Algorithm 14 of ESMfold2 paper.

    `design_idx` restricts which positions may be masked (and scored): the
    masked positions for each sample are drawn from `design_idx`, and
    `masking_fraction` is taken relative to its length. If None, every
    position in `seq_standard_tokens` is maskable.
    """

    # The ESMC (backbone + MLM head). May be None when a shared one is spliced
    # in at call time (see `StackedEsmFold2PLL` in losses/esmfold2.py) — built
    # that way the term holds no ESMC copy of its own.
    esm: ESMCForMaskedLM | None = None
    masking_fraction: float = 0.15
    num_samples: int = 4
    design_idx: jax.Array | None = eqx.field(
        default=None,
        converter=lambda x: None if x is None else jnp.asarray(x),
    )

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]
        vocab = esmc_vocab()
        # standard tokenization → ESM tokenization, then add cls/eos
        esm_toks_unpadded = seq_standard_tokens @ boltz_to_esmc_matrix(vocab)

        esm_toks_unpadded_one_hot_ste = jax.nn.one_hot(esm_toks_unpadded.argmax(-1), esm_toks_unpadded.shape[-1])
        esm_toks_unpadded_one_hot_ste = esm_toks_unpadded_one_hot_ste + esm_toks_unpadded - jax.lax.stop_gradient(esm_toks_unpadded)

        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([vocab["<cls>"]], ESMC_VOCAB_SIZE),
                esm_toks_unpadded_one_hot_ste,
                jax.nn.one_hot([vocab["<eos>"]], ESMC_VOCAB_SIZE),
            ]
        )

        # Positions eligible for masking (default: all of them). `idx_pool` is
        # passed straight to `jax.random.choice`: an int n samples arange(n),
        # a 1-D array samples its element values (the actual position indices).
        idx_pool = n if self.design_idx is None else self.design_idx
        pool_size = n if self.design_idx is None else self.design_idx.shape[0]

        def single_sample(key):
            n_masked = int(self.masking_fraction * pool_size)
            masked_idx = jax.random.choice(key, idx_pool, shape = (n_masked,), replace=False)


            masked_tokens = esm_toks.at[masked_idx+1].set( #offset for <cls>
                jax.nn.one_hot(vocab["<mask>"], ESMC_VOCAB_SIZE)
            )
            # Embed by matmul rather than the integer lookup in
            # `ESMCForMaskedLM.__call__`, so gradients reach the soft sequence.
            x = masked_tokens @ self.esm.esmc.embed.weight
            x, _ = self.esm.esmc.transformer(
                x[None], None, collect_hidden_states=False
            )
            logits = self.esm.lm_head(x)[0]
            return -(esm_toks_unpadded[masked_idx] * jax.nn.log_softmax(logits[masked_idx+1])).sum()/n_masked

        masked_perplexities = jax.vmap(single_sample)(jax.random.split(key, self.num_samples))
        
        per = masked_perplexities.mean()
        # `per` is the mean masked NLL (pseudo-log-perplexity, lower = more
        # likely); minimizing it favors plausible sequences. Also report the
        # actual pseudo-perplexity exp(per) (interpretable: ~1 perfect,
        # ~|vocab| for random).
        return per, {
            "esmc_perplexity": jnp.exp(per),
            "esmc_log_perplexity": per,
        }
