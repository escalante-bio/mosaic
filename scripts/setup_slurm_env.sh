#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/setup_slurm_env.sh
# Assumes CUDA 12.4 toolchain available on node (module load cuda/12.4 recommended)

python -m pip install -U pip setuptools wheel

# PyTorch with CUDA 12.4
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch --upgrade

# JAX CUDA stack compatible with Joltz/Boltz
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin || true
python -m pip install --no-cache-dir jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Ensure libdevice and NVCC are present for XLA
python -m pip install nvidia-cuda-nvcc-cu12

# cuDNN 9.8 (works with jaxlib 0.6.2)
python -m pip install nvidia-cudnn-cu12==9.8.0.87 --no-deps

# Numpy/Numba compatible with Boltz
python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4 numba==0.61.0

# Core deps
python -m pip install transformers datasets trl accelerate
python -m pip install gemmi fair-esm

# Joltz and Boltz
python -m pip install --no-cache-dir git+https://github.com/adaptyvbio/joltz.git
python -m pip install --no-cache-dir git+https://github.com/jwohlwend/boltz.git

echo "Done. Set XLA_FLAGS to point to libdevice if needed, e.g.:"
echo "export XLA_FLAGS=\"--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_cuda_data_dir=/usr/local/cuda\""


