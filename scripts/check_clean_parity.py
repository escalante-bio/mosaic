#!/usr/bin/env python3
"""Validate parity between PyTorch CLEAN head and JAX reimplementation.

The reference CLEAN project ships a ``LayerNormNet`` module implemented in
PyTorch. Mosaic mirrors that module using :class:`mosaic.losses.clean.CleanHead`
so that the loss can run inside JAX. This script acts as a quick regression
check: it loads a PyTorch checkpoint, transfers the weights into an Equinox
``CleanHead`` via :func:`load_clean_head_from_torch`, and then compares the
outputs on random inputs.

Example
-------
```
python scripts/check_clean_parity.py \
    --weights _external/tiny-clean-test/CLEAN/app/data/pretrained/split100.pth
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():  # allow running without installing the package
    sys.path.insert(0, str(_SRC_ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover - script level guard
    raise SystemExit("PyTorch is required to run this parity check") from exc


def _load_torch_clean_head(weights: Path, hidden_dim: int, out_dim: int):
    """Instantiate the PyTorch reference model with checkpoint weights."""

    torch_root = Path(__file__).resolve().parent.parent / "_external" / "tiny-clean-test" / "CLEAN" / "app" / "src"
    sys.path.append(str(torch_root))

    from CLEAN.model import LayerNormNet  # type: ignore

    device = torch.device("cpu")
    model = LayerNormNet(hidden_dim, out_dim, device=device, dtype=torch.float32)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Path to PyTorch CLEAN LayerNormNet checkpoint (.pth)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension used during training")
    parser.add_argument("--out-dim", type=int, default=128, help="Output dimension used during training")
    parser.add_argument("--samples", type=int, default=16, help="Number of random vectors to test")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for reproducibility")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()

    torch.manual_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    torch_model = _load_torch_clean_head(args.weights, args.hidden_dim, args.out_dim)

    # Random test vectors (match dtype/shape expected by both implementations)
    x_torch = torch.randn(args.samples, 1280, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_model(x_torch).cpu().numpy()

    # Load the JAX module and evaluate under vmap for batch processing
    from mosaic.losses.clean import load_clean_head_from_torch

    clean_head = load_clean_head_from_torch(
        args.weights,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        key=key,
    )

    x_jax = jnp.asarray(x_torch.numpy(), dtype=jnp.float32)
    batched_apply = jax.vmap(clean_head)
    jax_out = np.asarray(batched_apply(x_jax))

    abs_diff = np.max(np.abs(jax_out - torch_out))
    rel_diff = np.max(np.abs(jax_out - torch_out) / (np.maximum(np.abs(torch_out), 1e-6)))

    print({
        "max_abs_diff": float(abs_diff),
        "max_rel_diff": float(rel_diff),
        "torch_out_shape": tuple(torch_out.shape),
        "jax_out_shape": tuple(jax_out.shape),
    })

    tol = 5e-6  # allow small numerical drift from casting
    if not np.allclose(jax_out, torch_out, atol=tol, rtol=tol):  # pragma: no cover - CLI side-effect
        raise SystemExit("Mismatch detected; see printed metrics above.")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
