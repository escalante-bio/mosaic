#!/bin/bash
#SBATCH --job-name=mosaic
#SBATCH --account=YOUR_PROJECT_CODE      # <-- replace with your Isambard project code
#SBATCH --partition=gpu                  # check: sinfo | grep gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 1 GH200 GPU per task is enough
#SBATCH --cpus-per-task=16              # GH200 nodes have many CPU cores; 16 is generous
#SBATCH --mem=64G
#SBATCH --time=04:00:00                  # 4 h — plenty for 150 design steps
#SBATCH --output=logs/%x_%j.out         # stdout  →  logs/mosaic_<jobid>.out
#SBATCH --error=logs/%x_%j.err          # stderr  →  logs/mosaic_<jobid>.err
#
# ---- Array job (one SLURM task per random seed) --------------------------
# Uncomment to run 10 independent design trajectories in parallel:
#SBATCH --array=0-9
#
# Usage (single run):
#   sbatch hpc/job.slurm
#
# Usage (array — 10 seeds):
#   # uncomment the --array line above, then:
#   sbatch hpc/job.slurm
#
# To override the task (see JOB_TASK below):
#   sbatch hpc/job.slurm prediction      # just run a structure prediction
# --------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Environment setup
# ---------------------------------------------------------------------------
echo "Job started : $(date)"
echo "Host        : $(hostname)"
echo "Working dir : $PWD"
echo "SLURM job   : ${SLURM_JOB_ID:-none}  array task: ${SLURM_ARRAY_TASK_ID:-0}"

# Load system modules (adjust names to match 'module avail' on Isambard-AI)
if command -v module &>/dev/null; then
    # module load cudatoolkit/12.3.0   # uncomment + adjust as needed
    echo "Skipping module load (add your site modules here)"
fi

# Ensure uv is on PATH (installed by hpc/install.sh)
export PATH="$HOME/.local/bin:$PATH"

# Change to repo root (so relative paths in scripts work)
cd "$(dirname "$0")/.."

# Create logs directory if missing
mkdir -p logs results

# ---------------------------------------------------------------------------
# 2. Decide which task to run
# ---------------------------------------------------------------------------
# Default: binder design.  Pass "prediction" or "test" as $1 to override.
JOB_TASK="${1:-design}"

# Seed from array task ID (0 if not an array job)
SEED="${SLURM_ARRAY_TASK_ID:-0}"

case "$JOB_TASK" in

    test)
        # Quick install check — no GPU-intensive work
        echo "=== Running install test ==="
        uv run python hpc/test_install.py
        ;;

    prediction)
        # Single structure prediction (good first real test)
        echo "=== Running structure prediction ==="
        uv run python hpc/run_prediction.py --out-dir "results/prediction"
        ;;

    design|*)
        # Full binder design pipeline
        echo "=== Running binder design (seed=${SEED}) ==="
        uv run python hpc/run_binder_design.py \
            --target PDL1.pdb \
            --binder-length 80 \
            --n-steps 150 \
            --seed "${SEED}" \
            --out-dir "results/binder_design"
        ;;

esac

echo "Job finished: $(date)"
