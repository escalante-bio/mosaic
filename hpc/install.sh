#!/usr/bin/env bash
# Installation script for mosaic on Isambard-AI (NVIDIA GH200)
# Run this once from your home directory or a suitable project directory.
#
# Usage:
#   cd /path/to/mosaic   (your forked clone)
#   bash hpc/install.sh
#
# The script will:
#   1. Ensure uv is installed (to $HOME/.local/bin/uv)
#   2. Install all Python dependencies via uv sync with CUDA support
#   3. Print a verification command to check the GPU is visible

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Isambard-AI module setup
#    Load the CUDA and Python modules that are available on your partition.
#    Run `module avail` to see what is installed; adjust the names below.
#    GH200 nodes typically expose CUDA 12.x; JAX bundles its own CUDA libs
#    so the exact system version matters less, but Python 3.12 is required.
# ---------------------------------------------------------------------------
if command -v module &>/dev/null; then
    # Adjust these module names to match what 'module avail' shows on Isambard.
    # Common patterns for Isambard-AI:
    #   module load cudatoolkit/12.x.x   or   module load cuda/12.x
    #   module load python/3.12.x        or   use the uv-managed Python below
    echo "[install] Loading environment modules (edit this section if names differ)..."
    # module load cudatoolkit/12.3.0   # uncomment and adjust as needed
    # module load gcc/13               # uv may need a recent compiler
fi

# ---------------------------------------------------------------------------
# 1. Install uv (fast Python package manager, replaces pip/conda here)
#    uv is installed to ~/.local/bin and does NOT need root.
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[install] uv not found — installing to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in the current shell session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[install] uv already installed: $(uv --version)"
fi

# Ensure uv is on PATH for the rest of this script
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# 2. Sync dependencies with CUDA support
#
#    uv reads pyproject.toml and installs everything into a local .venv.
#    --group jax-cuda installs jax[cuda] (bundles its own CUDA 12 libraries;
#    no system CUDA match needed).
#
#    NOTE: The pyproject.toml pins nvidia-cublas-cu12 and nvidia-cusolver-cu12
#    as Python packages. These are the bundled NVIDIA wheels that JAX needs and
#    are separate from the system CUDA modules.  If you see conflicts, try the
#    override section below.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."    # ensure we're in the repo root
echo "[install] Installing dependencies (this will take a few minutes the first time)..."
uv sync --group jax-cuda

# ---------------------------------------------------------------------------
# 3. Optional: override if the bundled NVIDIA wheels conflict with Isambard's
#    existing CUDA installation.  Uncomment if 'uv sync' fails with CUDA errors.
# ---------------------------------------------------------------------------
# uv sync --group jax-cuda \
#     --override "nvidia-cublas-cu12>=12.0,<13" \
#     --override "nvidia-cusolver-cu12>=11,<12"

# ---------------------------------------------------------------------------
# 4. Quick smoke-test: check JAX can see the GPU
# ---------------------------------------------------------------------------
echo ""
echo "[install] Running GPU visibility check..."
uv run python - <<'EOF'
import jax
devices = jax.devices()
print(f"  JAX version      : {jax.__version__}")
print(f"  Visible devices  : {devices}")
if any(str(d).startswith("gpu") or "cuda" in str(d).lower() or "GH" in str(d) for d in devices):
    print("  GPU found — installation looks good!")
else:
    print("  WARNING: No GPU detected. Check your SLURM allocation and module setup.")
    print("  JAX defaults to CPU; structure prediction will be extremely slow.")
EOF

echo ""
echo "[install] Done. To run examples:"
echo "   uv run python hpc/test_install.py"
echo "   uv run python hpc/run_prediction.py"
echo "   sbatch hpc/job.slurm"
