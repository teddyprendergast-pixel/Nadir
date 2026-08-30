#!/bin/bash
# PPO training for Nadir on CINECA Leonardo (boost partition, A100 64GB).
#
# Replaces the previous scripts/train.sh, which could not have run here:
# it asked for "--partition=gpu" (no such partition on Leonardo), activated a
# conda env that does not exist, and invoked "python -m nadir.training.train"
# — lowercase, which is an ImportError on a case-sensitive filesystem.
#
# Submit from the repository root:
#     sbatch scripts/train_leonardo.sbatch
#
# One A100 is enough: this is a 10-DOF model and MJX keeps the whole batch
# resident on the GPU. Leonardo bills a boost node as 4 GPUs / 32 cores, so
# asking for 1 GPU and 8 cores takes a clean quarter node.

#SBATCH --job-name=nadir-ppo
#SBATCH --account=INF26_npqcd
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nadir-ppo-%j.out
#SBATCH --error=logs/nadir-ppo-%j.out

set -euo pipefail

REPO=/leonardo_scratch/large/userexternal/rdionisi/fun/Nadir
VENV=/leonardo_scratch/large/userexternal/rdionisi/fun/nadir-venv

module purge
module load python/3.11.7
# Deliberately NO "module load cuda". The jax[cuda12] wheels bundle their own
# CUDA runtime (cuBLAS, cuSPARSE, cuDNN, ...) under site-packages/nvidia/, and
# only need libcuda.so from the node's driver. Loading Leonardo's cuda module
# puts the system CUDA ahead of them on LD_LIBRARY_PATH, and JAX then fails its
# version check with "Unable to load cuSPARSE" and silently falls back to CPU.

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs

# Do not preallocate the whole card — MJX sizes its buffers from num_envs and
# a hard preallocation hides genuine OOMs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_PLATFORMS=cuda

echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "git: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

srun python -u -m Nadir.training.train \
    --num-envs "${NADIR_NUM_ENVS:-4096}" \
    --num-steps "${NADIR_NUM_STEPS:-24}" \
    --total-timesteps "${NADIR_TOTAL_STEPS:-300000000}" \
    --seed "${NADIR_SEED:-42}" \
    --run-name "${NADIR_RUN_NAME:-run-${SLURM_JOB_ID:-local}}" \
    --checkpoint-dir "${NADIR_CKPT_DIR:-$REPO/runs}" \
    "$@"

echo "=== $(date -Is) finished ==="
