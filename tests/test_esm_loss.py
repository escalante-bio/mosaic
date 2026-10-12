from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from esm2quinox._esm2 import _alphabet as ESM_TOKENS
from mosaic.losses.esm import embed_esm2_tokens

ESM_VOCAB_SIZE = max(ESM_TOKENS.values()) + 1


@pytest.fixture(autouse=True)
def _esm2quinox_legacy_mask_alias(monkeypatch):
    if "m" not in ESM_TOKENS:
        monkeypatch.setitem(ESM_TOKENS, "m", ESM_TOKENS["#"])


def _esm(*, token_dropout=True):
    return SimpleNamespace(
        embedding=SimpleNamespace(weight=jnp.eye(ESM_VOCAB_SIZE)),
        token_dropout=token_dropout,
    )


def _tokens(*indices):
    return jax.nn.one_hot(jnp.asarray(indices), ESM_VOCAB_SIZE)


def test_embed_esm2_tokens_scales_unmasked_embeddings():
    tokens = _tokens(ESM_TOKENS["A"], ESM_TOKENS["G"], ESM_TOKENS["L"])

    embedded = embed_esm2_tokens(_esm(), tokens, np.zeros(3, dtype=bool))

    np.testing.assert_allclose(embedded, tokens * 0.88)


def test_embed_esm2_tokens_accounts_for_observed_masks():
    tokens = _tokens(ESM_TOKENS["A"], ESM_TOKENS["m"], ESM_TOKENS["L"])

    embedded = embed_esm2_tokens(_esm(), tokens, np.zeros(3, dtype=bool))

    expected = tokens.at[1].set(0.0) * (0.88 / (1 - 1 / 3))
    np.testing.assert_allclose(embedded, expected)


def test_embed_esm2_tokens_respects_padding_and_disabled_token_dropout():
    tokens = _tokens(ESM_TOKENS["A"], ESM_TOKENS["G"], ESM_TOKENS["L"])

    embedded = embed_esm2_tokens(
        _esm(token_dropout=False), tokens, [False, False, True]
    )

    expected = tokens.at[2].set(0.0)
    np.testing.assert_allclose(embedded, expected)
