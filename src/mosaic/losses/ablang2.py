from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic.common import LossTerm, TOKENS
from ablang2.load_model import load_model
from jablang import from_torch


def boltz_to_ablang2_matrix(tokenizer):
    T = np.zeros((len(TOKENS), len(tokenizer.aa_to_token)))
    for i, tok in enumerate(TOKENS):
        idx = tokenizer.aa_to_token[tok]
        T[i, idx] = 1
    return T


def load_ablang2():
    model_pt, tokenizer, _hparams = load_model("ablang2-paired")
    model_pt.eval()
    return from_torch(model_pt), tokenizer


class Ablang2PseudoLikelihood(LossTerm):
    """Pseudo-likelihood loss using the AbLang2 paired model.

    Formats the concatenated binder sequence as ``<H>|<L>`` (or ``<H>|`` /
    ``|<L>`` for single-chain) to match ablang2's input convention, and masks
    special-token logits before log-softmax.

    ``heavy_len`` specifies how many leading residues belong to the heavy chain;
    the remainder are the light chain.  Use ``heavy_len=len(seq)`` for heavy-only
    and ``heavy_len=0`` for light-only.
    """

    model: Any
    tokenizer: Any
    heavy_len: int
    designable_positions: jax.Array | None
    token_mapping: jax.Array
    special_mask: jax.Array
    mask_onehot: jax.Array
    vocab_size: int
    stop_grad: bool = True
    aux_name: str = "ablang2_ppl"

    def __init__(
        self,
        model,
        tokenizer,
        heavy_len,
        designable_positions=None,
        stop_grad=True,
        aux_name="ablang2_ppl",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.heavy_len = heavy_len
        self.designable_positions = designable_positions
        self.stop_grad = stop_grad
        self.aux_name = aux_name
        self.token_mapping = jnp.array(boltz_to_ablang2_matrix(tokenizer))
        self.vocab_size = len(tokenizer.aa_to_token)
        special_indices = jnp.array(tokenizer.all_special_tokens, dtype=jnp.int32)
        self.special_mask = (
            jnp.zeros(self.vocab_size, dtype=bool).at[special_indices].set(True)
        )
        self.mask_onehot = jax.nn.one_hot(tokenizer.aa_to_token["*"], self.vocab_size)

    def __call__(self, seq_standard_tokens, *, key):
        del key
        n = seq_standard_tokens.shape[0]
        designable_positions = (
            self.designable_positions
            if self.designable_positions is not None
            else jnp.arange(n, dtype=jnp.int32)
        )

        ablang2_toks = seq_standard_tokens @ self.token_mapping
        at = self.tokenizer.aa_to_token

        def special(token):
            return jax.nn.one_hot(jnp.array([at[token]]), self.vocab_size)

        parts: list[jax.Array] = []
        sequence_token_indices = jnp.full(n, -1, dtype=jnp.int32)
        offset = 0

        if self.heavy_len > 0:
            parts += [special("<"), ablang2_toks[: self.heavy_len], special(">")]
            sequence_token_indices = sequence_token_indices.at[: self.heavy_len].set(
                jnp.arange(offset + 1, offset + 1 + self.heavy_len, dtype=jnp.int32)
            )
            offset += self.heavy_len + 2

        parts.append(special("|"))
        offset += 1

        if self.heavy_len < n:
            parts += [special("<"), ablang2_toks[self.heavy_len :], special(">")]
            sequence_token_indices = sequence_token_indices.at[self.heavy_len :].set(
                jnp.arange(offset + 1, offset + 1 + n - self.heavy_len, dtype=jnp.int32)
            )

        toks = jnp.concatenate(parts)
        residue_indices = sequence_token_indices[designable_positions]
        designable_toks = ablang2_toks[designable_positions]
        num_designable = designable_positions.shape[0]

        def single_ll(token_index):
            masked_tokens = toks.at[token_index].set(self.mask_onehot)
            x = masked_tokens @ self.model.rep.aa_embed_layer.weight
            x = self.model.rep.encoder_blocks(x[None])
            x = self.model.rep.layer_norm(x)
            logits = self.model.head(x)[0]
            logits = jnp.where(self.special_mask, -1e9, logits[token_index])
            return jax.nn.log_softmax(logits)

        masked_log_likelihoods = jax.vmap(single_ll)(residue_indices)
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        per_position_pll = (masked_log_likelihoods * designable_toks).sum(-1)
        pll = jnp.sum(per_position_pll) / jnp.maximum(
            jnp.array(num_designable, dtype=per_position_pll.dtype), 1.0
        )
        return -pll, {self.aux_name: jnp.exp(-pll)}
