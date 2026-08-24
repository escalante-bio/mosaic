"""Autoregressive sampling for ProteinMPNN.

`mosaic.losses.protein_mpnn.inverse_fold` uses Jacobi iteration, which is fast
and differentiable but only approximates the true autoregressive distribution
and offers no way to hold positions fixed.  Binder design pipelines need exact
sampling with fixed positions (target chain + interface hotspots) and amino
acid omission, so this module implements the real thing.

The sampler walks the decoding order one designable position at a time,
re-running :meth:`ProteinMPNN.decode` with the partially decoded sequence.  The
autoregressive mask guarantees the logits at the position being decoded depend
only on positions decoded earlier, so this is mathematically identical to the
cached incremental decoder in the reference implementation -- just expressed
with the ``decode`` primitive mosaic already has.  Fixed positions are placed
first in the decoding order, so the scan only runs over designable positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from mosaic.proteinmpnn.mpnn import MPNN_ALPHABET, ProteinMPNN

__all__ = [
    "MPNN_ALPHABET",
    "MPNNSample",
    "encode_sequence",
    "decode_sequence",
    "make_bias",
    "sample",
    "sample_batch",
    "score_sequence",
]

_N_AA = 20
_N_TOKENS = len(MPNN_ALPHABET)
_AA_TO_IDX = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}


def encode_sequence(seq: str) -> np.ndarray:
    """Amino acid string -> indices in the ProteinMPNN alphabet."""
    return np.array([_AA_TO_IDX.get(aa, _AA_TO_IDX["X"]) for aa in seq], dtype=np.int32)


def decode_sequence(indices) -> str:
    """Indices in the ProteinMPNN alphabet -> amino acid string."""
    return "".join(MPNN_ALPHABET[int(i)] for i in np.asarray(indices))


def make_bias(
    length: int,
    *,
    omit_AAs: str | None = None,
    per_position_bias: dict[int, dict[str, float]] | None = None,
) -> np.ndarray:
    """Build an additive logit bias of shape ``(length, len(MPNN_ALPHABET))``.

    ``omit_AAs`` accepts either ``"CM"`` or ``"C,M"``.  ``X`` is always omitted.
    """
    bias = np.zeros((length, _N_TOKENS), dtype=np.float32)
    bias[:, _N_AA:] = -1e6

    if omit_AAs:
        for aa in omit_AAs.replace(",", "").strip():
            if aa in _AA_TO_IDX:
                bias[:, _AA_TO_IDX[aa]] -= 1e6

    if per_position_bias:
        for pos, entries in per_position_bias.items():
            for aa, value in entries.items():
                if aa in _AA_TO_IDX:
                    bias[pos, _AA_TO_IDX[aa]] += value

    return bias


@dataclass
class MPNNSample:
    """One sampled sequence plus the statistics the design pipeline reports."""

    sequence: str
    S: np.ndarray
    logits: np.ndarray
    score: float
    seqid: float
    decoding_order: np.ndarray

    def chains(self, chain_index) -> list[str]:
        chain_index = np.asarray(chain_index)
        return [
            decode_sequence(self.S[chain_index == c])
            for c in np.unique(chain_index)
        ]


def _decoding_scores(key, mask, fix_pos_mask):
    """Random decoding order: fixed first, then valid, then padding."""
    scores = jax.random.uniform(key, shape=mask.shape)
    scores = jnp.where(mask, scores, scores + 1.0)
    return jnp.where(fix_pos_mask, scores - 1.0, scores)


def _sample_single(
    mpnn: ProteinMPNN,
    *,
    X: Float[Array, "N 4 3"],
    S_native: Int[Array, " N"],
    mask: Bool[Array, " N"],
    residue_idx: Int[Array, " N"],
    chain_encoding_all: Int[Array, " N"],
    bias: Float[Array, "N T"],
    fix_pos_mask: Bool[Array, " N"],
    n_fixed: int,
    temperature: float,
    key,
):
    encode_key, order_key, gumbel_key = jax.random.split(key, 3)

    h_V, h_E, E_idx = mpnn.encode(
        X=X,
        mask=mask,
        residue_idx=residue_idx,
        chain_encoding_all=chain_encoding_all,
        key=encode_key,
    )

    order_scores = _decoding_scores(order_key, mask, fix_pos_mask)
    perm = jnp.argsort(order_scores)

    native_onehot = jax.nn.one_hot(S_native, _N_TOKENS)
    S_init = jnp.where(fix_pos_mask[:, None], native_onehot, 0.0)

    n_steps = mask.shape[0] - n_fixed
    gumbel = jax.random.gumbel(gumbel_key, (n_steps, _N_TOKENS))

    def step(carry, xs):
        S, logits_acc = carry
        t, noise = xs

        logits = mpnn.decode(
            S=S,
            h_V=h_V,
            h_E=h_E,
            E_idx=E_idx,
            mask=mask,
            decoding_order=order_scores,
        )[0]
        logits_t = logits[t]
        logits_acc = logits_acc.at[t].set(logits_t)

        biased = (logits_t + bias[t]) / temperature + noise
        # `X` (and any padding tokens) are never sampled.
        aa = jnp.argmax(biased[:_N_AA])
        S = S.at[t].set(jax.nn.one_hot(aa, _N_TOKENS))
        return (S, logits_acc), None

    logits_init = jnp.zeros((mask.shape[0], _N_TOKENS))
    (S, logits), _ = jax.lax.scan(
        step, (S_init, logits_init), (perm[n_fixed:], gumbel)
    )

    return {"S": S, "logits": logits, "decoding_order": perm}


def _score_from_logits(logits, S, S_native, mask, fix_pos_mask):
    score_mask = jnp.logical_and(mask, jnp.logical_not(fix_pos_mask)).astype(jnp.float32)
    log_q = jax.nn.log_softmax(logits, axis=-1)[..., :_N_AA]
    per_pos = -(S[..., :_N_AA] * log_q).sum(-1)
    recovered = (S[..., :_N_AA].argmax(-1) == S_native).astype(jnp.float32)
    denom = score_mask.sum() + 1e-8
    return (
        (per_pos * score_mask).sum(-1) / denom,
        (recovered * score_mask).sum(-1) / denom,
    )


def _prepare(
    *,
    length: int,
    fix_pos,
    omit_AAs,
    bias,
    mask,
):
    fix_pos_mask = np.zeros(length, dtype=bool)
    if fix_pos is not None:
        fix_pos = np.asarray(fix_pos)
        if fix_pos.dtype == bool:
            fix_pos_mask = fix_pos.copy()
        else:
            fix_pos_mask[fix_pos.astype(int)] = True

    mask = np.ones(length, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    # Padding positions carry no information and must not be sampled.
    fix_pos_mask = np.logical_or(fix_pos_mask, ~mask)

    total_bias = make_bias(length, omit_AAs=omit_AAs)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape[-1] == _N_AA:
            total_bias[:, :_N_AA] += bias
        else:
            total_bias += bias

    return fix_pos_mask, mask, total_bias


def sample_batch(
    mpnn: ProteinMPNN,
    *,
    X,
    S_native,
    residue_idx,
    chain_encoding_all,
    key,
    num_seqs: int = 1,
    temperature: float = 0.1,
    fix_pos=None,
    omit_AAs: str | None = None,
    bias=None,
    mask=None,
    batch_size: int | None = None,
) -> list[MPNNSample]:
    """Sample ``num_seqs`` sequences for one backbone.

    Positions listed in ``fix_pos`` keep their residue from ``S_native`` and are
    decoded first so every designed position is conditioned on them.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    S_native_np = np.asarray(S_native, dtype=np.int32)
    length = X.shape[0]

    fix_pos_mask, mask_np, total_bias = _prepare(
        length=length, fix_pos=fix_pos, omit_AAs=omit_AAs, bias=bias, mask=mask
    )
    n_fixed = int(fix_pos_mask.sum())
    if n_fixed == length:
        raise ValueError("every position is fixed - nothing to design")

    S_native_j = jnp.asarray(S_native_np)
    mask_j = jnp.asarray(mask_np)
    fix_j = jnp.asarray(fix_pos_mask)
    bias_j = jnp.asarray(total_bias)
    residue_idx_j = jnp.asarray(residue_idx, dtype=jnp.int32)
    chain_j = jnp.asarray(chain_encoding_all, dtype=jnp.int32)

    def run(k):
        out = _sample_single(
            mpnn,
            X=X,
            S_native=S_native_j,
            mask=mask_j,
            residue_idx=residue_idx_j,
            chain_encoding_all=chain_j,
            bias=bias_j,
            fix_pos_mask=fix_j,
            n_fixed=n_fixed,
            temperature=temperature,
            key=k,
        )
        score, seqid = _score_from_logits(
            out["logits"], out["S"], S_native_j, mask_j, fix_j
        )
        return {**out, "score": score, "seqid": seqid}

    run_batch = jax.jit(jax.vmap(run))

    batch_size = num_seqs if batch_size is None else min(batch_size, num_seqs)
    samples: list[MPNNSample] = []
    remaining = num_seqs
    while remaining > 0:
        n = min(batch_size, remaining)
        key, sub = jax.random.split(key)
        out = jax.tree.map(np.asarray, run_batch(jax.random.split(sub, n)))
        for i in range(n):
            S = out["S"][i].argmax(-1)
            samples.append(
                MPNNSample(
                    sequence=decode_sequence(S),
                    S=S,
                    logits=out["logits"][i],
                    score=float(out["score"][i]),
                    seqid=float(out["seqid"][i]),
                    decoding_order=out["decoding_order"][i],
                )
            )
        remaining -= n

    return samples


def sample(mpnn: ProteinMPNN, **kwargs) -> MPNNSample:
    """Sample a single sequence.  See :func:`sample_batch`."""
    return sample_batch(mpnn, num_seqs=1, **kwargs)[0]


def score_sequence(
    mpnn: ProteinMPNN,
    *,
    X,
    S,
    residue_idx,
    chain_encoding_all,
    key,
    mask=None,
    fix_pos=None,
    num_decoding_orders: int = 1,
) -> tuple[float, np.ndarray]:
    """Average negative log-likelihood of ``S`` under ``mpnn``.

    This is the teacher-forced score used to rank designs; it averages over
    ``num_decoding_orders`` random autoregressive orders.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    S_np = np.asarray(S, dtype=np.int32)
    length = X.shape[0]

    fix_pos_mask, mask_np, _ = _prepare(
        length=length, fix_pos=fix_pos, omit_AAs=None, bias=None, mask=mask
    )
    S_j = jnp.asarray(S_np)
    mask_j = jnp.asarray(mask_np)
    fix_j = jnp.asarray(fix_pos_mask)
    onehot = jax.nn.one_hot(S_j, _N_TOKENS)

    @jax.jit
    def one(k):
        encode_key, order_key = jax.random.split(k)
        h_V, h_E, E_idx = mpnn.encode(
            X=X,
            mask=mask_j,
            residue_idx=jnp.asarray(residue_idx, dtype=jnp.int32),
            chain_encoding_all=jnp.asarray(chain_encoding_all, dtype=jnp.int32),
            key=encode_key,
        )
        order_scores = _decoding_scores(order_key, mask_j, fix_j)
        return mpnn.decode(
            S=onehot,
            h_V=h_V,
            h_E=h_E,
            E_idx=E_idx,
            mask=mask_j,
            decoding_order=order_scores,
        )[0]

    logits = jnp.mean(
        jnp.stack([one(k) for k in jax.random.split(key, num_decoding_orders)]), 0
    )
    score, _ = _score_from_logits(logits, onehot, S_j, mask_j, fix_j)
    return float(score), np.asarray(logits)
