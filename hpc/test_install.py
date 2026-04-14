"""
test_install.py — quick sanity check for a mosaic installation on HPC.

Run with:
    uv run python hpc/test_install.py

This script does NOT run any structure prediction (no model weights needed).
It just checks that:
  1. JAX can see at least one GPU / accelerator
  2. Core mosaic modules can be imported
  3. A trivial JAX jit-compiled function works on the GPU

Takes < 30 seconds.
"""

import sys

# ---------------------------------------------------------------------------
# 1. JAX device check
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. JAX device check")
print("=" * 60)
import jax
import jax.numpy as jnp

devices = jax.devices()
print(f"  JAX version  : {jax.__version__}")
print(f"  Devices      : {devices}")

gpu_devices = [d for d in devices if d.platform == "gpu"]
if gpu_devices:
    print(f"  GPU(s) found : {gpu_devices}  ✓")
else:
    print("  WARNING: No GPU found. Falling back to CPU.")
    print("  Structure prediction will be impractically slow on CPU.")
    print("  Make sure you have allocated a GPU node in your SLURM job.")

# ---------------------------------------------------------------------------
# 2. Simple JAX computation on the default device
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. Trivial JAX computation")
print("=" * 60)

@jax.jit
def matmul_test(a, b):
    return jnp.dot(a, b)

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (512, 512))
b = jax.random.normal(key, (512, 512))
c = matmul_test(a, b)
c.block_until_ready()
print(f"  512x512 matmul result shape : {c.shape}  ✓")
print(f"  Computation ran on          : {c.devices()}")

# ---------------------------------------------------------------------------
# 3. Core mosaic imports
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. Core mosaic imports")
print("=" * 60)

try:
    from mosaic.optimizers import simplex_APGM, gradient_MCMC
    print("  mosaic.optimizers            ✓")
except Exception as e:
    print(f"  mosaic.optimizers  FAILED: {e}")
    sys.exit(1)

try:
    from mosaic.common import LossTerm, TOKENS
    print("  mosaic.common                ✓")
except Exception as e:
    print(f"  mosaic.common      FAILED: {e}")
    sys.exit(1)

try:
    from mosaic.structure_prediction import TargetChain
    print("  mosaic.structure_prediction  ✓")
except Exception as e:
    print(f"  mosaic.structure_prediction  FAILED: {e}")
    sys.exit(1)

try:
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    print("  mosaic.proteinmpnn           ✓")
except Exception as e:
    print(f"  mosaic.proteinmpnn FAILED: {e}")
    sys.exit(1)

try:
    from mosaic.losses.stability import StabilityModel
    print("  mosaic.losses.stability      ✓")
except Exception as e:
    print(f"  mosaic.losses.stability  FAILED: {e}")
    sys.exit(1)

try:
    from mosaic.losses.trigram import TrigramLL
    print("  mosaic.losses.trigram        ✓")
except Exception as e:
    print(f"  mosaic.losses.trigram  FAILED: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Structure prediction model import (no weight download, just import)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. Structure model imports (no weights downloaded yet)")
print("=" * 60)

try:
    from mosaic.models.boltz1 import Boltz1
    print("  Boltz1 class importable      ✓")
except Exception as e:
    print(f"  Boltz1 import FAILED: {e}")

try:
    from mosaic.models.protenix import Protenix2025
    print("  Protenix2025 class importable ✓")
except Exception as e:
    print(f"  Protenix2025 import FAILED: {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Installation check complete.")
print()
print("Next steps:")
print("  • Run a structure prediction  : uv run python hpc/run_prediction.py")
print("  • Run a binder design job     : sbatch hpc/job.slurm")
print("=" * 60)
