#!/bin/bash
# Post-training evaluation: export the policy to ONNX, validate it in vanilla
# MuJoCo, and render a video.
#
# This needs a GPU node even though none of the work is heavy: MuJoCo's
# offscreen renderer needs an EGL device, and Leonardo's login nodes have
# neither a GPU (EGL) nor libOSMesa. Trying to render on a login node fails
# with "'GLContext' object has no attribute '_context'".
#
#     sbatch --export=ALL,NADIR_RUN=leonardo-300M scripts/evaluate_leonardo.sbatch

#SBATCH --job-name=nadir-eval
#SBATCH --account=INF26_npqcd
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/nadir-eval-%j.out
#SBATCH --error=logs/nadir-eval-%j.out

set -euo pipefail

REPO=/leonardo_scratch/large/userexternal/rdionisi/fun/Nadir
VENV=/leonardo_scratch/large/userexternal/rdionisi/fun/nadir-venv
RUN="${NADIR_RUN:?set NADIR_RUN to the run directory name under runs/}"
CKPT="${NADIR_CKPT:-$REPO/runs/$RUN/best}"

module purge
module load python/3.11.7          # no cuda module — see train_leonardo.sbatch
source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs exported_models docs

export JAX_PLATFORMS=cpu           # evaluation is single-env; CPU is plenty
export MUJOCO_GL=egl               # this is what the GPU is actually for

echo "=== $(date -Is) evaluating $CKPT ==="

python -u -m Nadir.export.export_onnx \
    --checkpoint "$CKPT" \
    --output "exported_models/nadir_policy.onnx"

python -u -m Nadir.sim.sim2sim \
    --onnx exported_models/nadir_policy.onnx \
    --seconds 20 --seeds 5 \
    --video "docs/nadir_walk.mp4"

python -u - <<'PY'
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = os.environ["NADIR_RUN"]
path = f"runs/{run}/metrics.csv"
rows = list(csv.DictReader(open(path)))
x = [float(r["env_steps"]) / 1e6 for r in rows]

# episode_return is deliberately not plotted: runs launched before a7cc29f
# carry a version that accumulates across episodes. mean_step_reward is the
# equivalent signal and is correct in every run. See docs/audit-2026-08-30.md.
panels = [
    ("episode_length", "mean episode length (steps)", None),
    ("termination_rate", "fraction of episodes ending in a fall", (0, 1.05)),
    ("mean_step_reward", "mean reward per step", None),
    ("value_loss", "value loss", None),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
for ax, (key, label, ylim) in zip(axes.flat, panels):
    ax.plot(x, [float(r[key]) for r in rows], lw=1.2)
    ax.set_ylabel(label); ax.grid(alpha=0.3)
    if key == "episode_length":
        ax.axhline(1000, ls="--", lw=0.8, c="grey")
        ax.annotate("20 s time limit", (x[-1], 1000), ha="right", va="bottom",
                    fontsize=8, color="grey")
    if ylim: ax.set_ylim(*ylim)
for ax in axes[1]:
    ax.set_xlabel("environment steps (millions)")
fig.suptitle(f"Nadir PPO — {run} (A100, MJX)")
fig.tight_layout()
fig.savefig("docs/training_curve.png", dpi=140)
print("wrote docs/training_curve.png")
PY

echo "=== $(date -Is) done ==="
