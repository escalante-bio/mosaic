import numpy as np
import jax
import jax.numpy as jnp
from typing import List


def aa_vocab() -> str:
    return "ARNDCQEGHILKMFPSTWYV"


def sanitize_sequence(seq: str) -> str:
    v = set(aa_vocab())
    return "".join([c for c in (seq or "").strip().upper() if c in v])


def sequence_to_one_hot(seq: str, binder_len: int) -> np.ndarray:
    seq = sanitize_sequence(seq)
    if len(seq) != binder_len:
        raise ValueError(f"Sequence length {len(seq)} != binder_len {binder_len}")
    vocab = aa_vocab()
    idx = np.array([vocab.index(c) for c in seq], dtype=np.int32)
    return np.eye(20, dtype=np.float32)[idx]


def one_hot_to_crisp_logits(one_hot: np.ndarray) -> np.ndarray:
    # map 1 -> +10.0, 0 -> -10.0
    return one_hot * 10.0 + (1.0 - one_hot) * -10.0


def build_reward_fn_from_loss(loss_function, binder_len: int, seed: int = 0):
    """
    Wrap a Mosaic loss_function(probs, key) into a Python reward(seq)->float.
    Reward = -loss. no JIT here to keep it simple and robust.
    """

    vocab = aa_vocab()

    def _reward_for_seq(seq: str) -> float:
        oh = sequence_to_one_hot(seq, binder_len)
        probs = jnp.asarray(oh)
        key = jax.random.key(seed)
        v, _ = loss_function(probs, key=key)
        return -float(v)

    def _batched_reward(seqs: List[str]) -> List[float]:
        return [_reward_for_seq(s) for s in seqs]

    return _batched_reward


