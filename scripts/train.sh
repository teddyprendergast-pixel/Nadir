#!/bin/bash
#SBATCH --job-name=nadir-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out

# Activate environment
source ~/.bashrc
conda activate nadir

# Run training
python -m nadir.training.train \
    --num-envs 4096 \
    --total-timesteps 100000000 \
    --seed 42 \
    --checkpoint-dir checkpoints/ \
    "$@"
