"""Lightweight helpers shared across mosaic_rl modules."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

AA_VOCAB = "ARNDCQEGHILKMFPSTWYV"
VOCAB_INDEX = {aa: idx for idx, aa in enumerate(AA_VOCAB)}


def sanitize_sequence(seq: str, binder_len: int, pad_token: str = "G") -> str:
    """Uppercase, drop unknown tokens, and pad/trim to ``binder_len``.

    The function keeps the logic intentionally simple (and JAX-friendly):
    - non alphabetic characters are ignored
    - unknown letters are replaced with the fallback token (glycine)
    - sequences shorter than ``binder_len`` are padded with the fallback token
    - longer sequences are truncated
    """

    if binder_len <= 0:
        raise ValueError("binder_len must be positive")

    allowed = set(AA_VOCAB)
    pad_token = pad_token.upper()
    if pad_token not in allowed:
        raise ValueError(f"pad_token must be an amino acid, got {pad_token}")

    cleaned: list[str] = []
    for ch in (seq or "").upper():
        if ch in allowed:
            cleaned.append(ch)
        elif ch.isalpha():
            cleaned.append(pad_token)

    if not cleaned:
        cleaned = [pad_token] * binder_len

    if len(cleaned) < binder_len:
        cleaned.extend([pad_token] * (binder_len - len(cleaned)))
    else:
        cleaned = cleaned[:binder_len]

    return "".join(cleaned)


def sequence_to_one_hot(seq: str, binder_len: int) -> np.ndarray:
    """Return a NumPy array with shape ``(binder_len, len(AA_VOCAB))``."""
    sanitized = sanitize_sequence(seq, binder_len)
    rows = np.zeros((binder_len, len(AA_VOCAB)), dtype=np.float32)
    for position, ch in enumerate(sanitized):
        rows[position, VOCAB_INDEX[ch]] = 1.0
    return rows


def decode_argmax(logits: Sequence[Sequence[float]]) -> str:
    """Convert argmax logits into a sequence string."""
    letters: list[str] = []
    for row in logits:
        if not row:
            letters.append("G")
            continue
        idx = int(np.argmax(np.asarray(row)))
        letters.append(AA_VOCAB[idx])
    return "".join(letters)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax over NumPy arrays (used outside JAX contexts)."""
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def ensure_shape(arr: np.ndarray, binder_len: int) -> np.ndarray:
    if arr.shape != (binder_len, len(AA_VOCAB)):
        raise ValueError(f"expected logits shape {(binder_len, len(AA_VOCAB))}, got {arr.shape}")
    return arr


def iter_prompts(prompts: Iterable[str], count: int) -> list[str]:
    prompts = list(prompts)
    if not prompts:
        return [""] * count
    return [prompts[min(i, len(prompts) - 1)] for i in range(count)]

