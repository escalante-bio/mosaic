from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic.common import LossTerm, TOKENS


def _boltz_to_ablang2_matrix(tokenizer):
    T = np.zeros((len(TOKENS), len(tokenizer.aa_to_token)))
    for i, tok in enumerate(TOKENS):
        idx = tokenizer.aa_to_token[tok]
        T[i, idx] = 1
    return T


def _load_ablang2():
    from ablang2.load_model import load_model
    from jablang import from_torch

    model_pt, tokenizer, _hparams = load_model("ablang2-paired")
    model_pt.eval()
    return from_torch(model_pt), tokenizer


@lru_cache(maxsize=1)
def _cached_ablang2_model():
    return _load_ablang2()


class Ablang2PseudoLikelihood(LossTerm):
    """Pseudo-likelihood loss using the AbLang2 paired model.
    Formats the concatenated binder sequence as ``<H>|<L>`` (or ``<H>|`` /
    ``|<L>`` for single-chain) to match ablang2's input convention, and masks
    special-token logits before log-softmax.
    """

    model: Any
    tokenizer: Any
    chain_slices: tuple[tuple[str, int, int], ...]  # (part_id, start, stop); part_id must be "H" or "L"
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
        chain_slices,
        designable_positions=None,
        stop_grad=True,
        aux_name="ablang2_ppl",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chain_slices = chain_slices
        self.designable_positions = designable_positions
        self.stop_grad = stop_grad
        self.aux_name = aux_name
        self.token_mapping = jnp.array(_boltz_to_ablang2_matrix(tokenizer))
        self.vocab_size = len(tokenizer.aa_to_token)
        special_indices = jnp.array(tokenizer.all_special_tokens, dtype=jnp.int32)
        self.special_mask = jnp.zeros(self.vocab_size, dtype=bool).at[special_indices].set(True)
        self.mask_onehot = jax.nn.one_hot(tokenizer.aa_to_token["*"], self.vocab_size)

    def __call__(self, seq_standard_tokens, *, key):
        del key
        n = seq_standard_tokens.shape[0]
        designable_positions = (
            self.designable_positions if self.designable_positions is not None
            else jnp.arange(n, dtype=jnp.int32)
        )

        ablang2_toks = seq_standard_tokens @ self.token_mapping
        at = self.tokenizer.aa_to_token

        def special(token):
            return jax.nn.one_hot(jnp.array([at[token]]), self.vocab_size)

        chain_bounds = {cid: (start, stop) for cid, start, stop in self.chain_slices}
        parts: list[jax.Array] = []
        sequence_token_indices = jnp.full(n, -1, dtype=jnp.int32)
        offset = 0

        for part_id in ("H", "|", "L"):
            if part_id == "|":
                parts.append(special("|"))
                offset += 1
            elif part_id in chain_bounds:
                start, stop = chain_bounds[part_id]
                parts.append(special("<"))
                offset += 1
                parts.append(ablang2_toks[start:stop])
                sequence_token_indices = sequence_token_indices.at[start:stop].set(
                    jnp.arange(offset, offset + stop - start, dtype=jnp.int32)
                )
                offset += stop - start
                parts.append(special(">"))
                offset += 1

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
