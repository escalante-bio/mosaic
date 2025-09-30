"""Utilities for reproducing CLEAN embeddings with the PyTorch reference stack.

The CLEAN loss expects a fixed 1280-dimensional embedding per sequence. The
original implementation uses ESM-1b to obtain this representation and then feeds
it through a small MLP (``LayerNormNet``). This module provides a lightweight
wrapper around ``fair-esm`` so JAX code can reuse the same embedding pipeline
without depending on the full PyTorch training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class CleanESMEmbedder:
    """Callable wrapper that reproduces CLEAN's ESM embeddings.

    Parameters
    ----------
    model_name
        Name of the pretrained ESM checkpoint to load. CLEAN defaults to
        ``esm1b_t33_650M_UR50S``.
    device
        Optional PyTorch device string (``"cpu"`` by default). Passing
        ``"cuda"`` is supported if the environment has GPU support.
    layer
        Representation layer to extract. Layer 33 matches the reference repo.
    """

    model_name: str = "esm1b_t33_650M_UR50S"
    device: str = "cpu"
    layer: int = 33

    def __post_init__(self) -> None:
        try:
            import torch
            from esm import pretrained
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "fair-esm and torch are required to build the CLEAN embedder.\n"
                "Install them via `pip install fair-esm torch`."
            ) from exc

        loader = getattr(pretrained, self.model_name, None)
        if loader is None:  # pragma: no cover - user misconfiguration
            raise ValueError(f"Unknown ESM model '{self.model_name}'")

        model, alphabet = loader()
        self._torch = torch
        self._model = model.eval().to(self.device)
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()

    def __call__(self, sequence: str) -> np.ndarray:
        torch = self._torch
        data = [("seq", sequence)]
        _, _, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            result = self._model(batch_tokens, repr_layers=[self.layer], return_contacts=False)
            representations = result["representations"][self.layer]
            # Drop CLS (0) and EOS tokens (last) to match the training code
            token_vecs = representations[0, 1 : len(sequence) + 1]
            emb = token_vecs.mean(dim=0)

        return emb.detach().cpu().numpy().astype(np.float32, copy=False)


def build_clean_embed_fn(
    *,
    model_name: str = "esm1b_t33_650M_UR50S",
    device: str = "cpu",
    layer: int = 33,
) -> Tuple[CleanESMEmbedder, Callable[[str], np.ndarray]]:
    """Convenience helper returning the embedder instance and callable.

    The embedder instance is returned to keep the underlying ESM model alive; the
    callable can be passed directly to :class:`CleanCosineSimilarityLoss`.
    """

    embedder = CleanESMEmbedder(model_name=model_name, device=device, layer=layer)
    return embedder, embedder.__call__
