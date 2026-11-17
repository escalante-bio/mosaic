"""ESM-C pseudo-likelihood losses, using jfold's ESMC models with masked-LM heads."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float
from jfold import pretrained
from jfold.models.esmfold2 import ESMC_VOCAB, ESMCForMaskedLM

from ..cache import cache_dir
from ..common import TOKENS, LossTerm

ESMC_VOCAB_SIZE = 64
# jfold checkpoint names, keyed by short name and by Hugging Face repository.
_RELEASES = {
    "esmc_300m": "esmc_300m_mlm",
    "esmc_600m": "esmc_600m_mlm",
    "esmc_6b": "esmc_6b_mlm",
    "biohub/esmc-300m": "esmc_300m_mlm",
    "biohub/esmc-600m": "esmc_600m_mlm",
    "biohub/esmc-6b": "esmc_6b_mlm",
}


def esmc_vocab() -> dict[str, int]:
    """Return a map from ESM-C token to its embedding row."""
    return dict(ESMC_VOCAB)


def load_esmc(
    model_name: str = "esmc_300m", *, compute_dtype: DTypeLike = "bfloat16"
) -> ESMCForMaskedLM:
    """Load ESM-C with its masked-LM head, downloading it on first use.

    `model_name` is `esmc_300m`, `esmc_600m`, `esmc_6b` or a Biohub repository name
    such as `Biohub/ESMC-6B`. With `compute_dtype="bfloat16"` the weights are stored
    in bfloat16 to halve memory, and computation still runs in float32.
    """
    if jnp.dtype(compute_dtype) not in (jnp.dtype("float32"), jnp.dtype("bfloat16")):
        raise ValueError("compute_dtype must be float32 or bfloat16")
    release = _RELEASES.get(model_name.lower())
    if release is None:
        raise ValueError(f"unknown ESMC {model_name!r}; one of {sorted(_RELEASES)}")
    with jax.default_device(jax.devices("cpu")[0]):
        model = pretrained.load(release, cache=cache_dir("jfold"))
        if jnp.dtype(compute_dtype) == jnp.bfloat16:
            model = jax.tree.map(
                lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, model
            )
    return eqx.filter_shard(model, jax.sharding.SingleDeviceSharding(jax.devices()[0]))


def boltz_to_esmc_matrix(vocab: dict[str, int]) -> np.ndarray:
    """Return the one-hot matrix mapping mosaic `TOKENS` order to ESM-C tokens."""
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
        seq_standard_tokens = seq_standard_tokens.astype(jnp.float32)
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
            # ESMC accepts probability vectors, so gradients flow to the soft sequence.
            logits = self.esm(masked_tokens).astype(jnp.float32)
            return jax.nn.log_softmax(logits[index])

        masked_log_likelihoods = jax.vmap(single_ll)(jnp.arange(start=1, stop=n + 1))
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        pll = (masked_log_likelihoods * esm_toks_unpadded).sum(-1).mean()
        return -pll, {"esmc_pll": pll}


class ESMCPseudoPerplexity(LossTerm):
    """Pseudoperplexity loss from Algorithm 14 of ESMfold2 paper.

    `design_idx` lists the positions that may be masked and scored, and
    `masking_fraction` is relative to its length. If None, every position can be
    masked.
    """

    # None when `mosaic.biohub.BiohubObjective` supplies a shared ESMC at call time.
    esm: ESMCForMaskedLM | None = None
    masking_fraction: float = 0.15
    num_samples: int = 4
    design_idx: jax.Array | None = eqx.field(
        default=None,
        converter=lambda x: None if x is None else jnp.asarray(x),
    )

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        esm = self.esm
        if esm is None:
            raise ValueError("Attach an ESMC model before evaluating the PLL loss")
        seq_standard_tokens = seq_standard_tokens.astype(jnp.float32)
        n = seq_standard_tokens.shape[0]
        vocab = esmc_vocab()
        esm_toks_unpadded = seq_standard_tokens @ boltz_to_esmc_matrix(vocab)

        esm_toks_unpadded_one_hot_ste = jax.nn.one_hot(
            esm_toks_unpadded.argmax(-1), esm_toks_unpadded.shape[-1]
        )
        esm_toks_unpadded_one_hot_ste = (
            esm_toks_unpadded_one_hot_ste
            + esm_toks_unpadded
            - jax.lax.stop_gradient(esm_toks_unpadded)
        )

        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([vocab["<cls>"]], ESMC_VOCAB_SIZE),
                esm_toks_unpadded_one_hot_ste,
                jax.nn.one_hot([vocab["<eos>"]], ESMC_VOCAB_SIZE),
            ]
        )

        # jax.random.choice samples arange(n) when given an int n, and the array's
        # values when given an array.
        idx_pool = n if self.design_idx is None else self.design_idx
        pool_size = n if self.design_idx is None else self.design_idx.shape[0]

        def single_sample(key):
            n_masked = int(self.masking_fraction * pool_size)
            masked_idx = jax.random.choice(
                key, idx_pool, shape=(n_masked,), replace=False
            )

            masked_tokens = esm_toks.at[masked_idx + 1].set(  # offset for <cls>
                jax.nn.one_hot(vocab["<mask>"], ESMC_VOCAB_SIZE)
            )
            logits = esm(masked_tokens).astype(jnp.float32)
            return (
                -(
                    esm_toks_unpadded[masked_idx]
                    * jax.nn.log_softmax(logits[masked_idx + 1])
                ).sum()
                / n_masked
            )

        masked_perplexities = jax.vmap(single_sample)(
            jax.random.split(key, self.num_samples)
        )

        per = masked_perplexities.mean()
        # `per` is the mean masked negative log-likelihood. Its exponential, the
        # pseudo-perplexity, is about 1 for a perfect model and about the vocabulary
        # size for a random one.
        return per, {
            "esmc_perplexity": jnp.exp(per),
            "esmc_log_perplexity": per,
        }
