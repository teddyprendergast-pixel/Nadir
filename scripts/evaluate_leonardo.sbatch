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
# Default to `best`, but allow pointing at a specific update_N. Prefer a
# completed update_N while training is still live: `best` is rewritten in
# place by the running job, so reading it can catch a partial write.
CKPT="${NADIR_CKPT:-$REPO/runs/$RUN/best}"
TAG="${NADIR_TAG:-$RUN}"
ONNX="${NADIR_ONNX:-exported_models/nadir_policy_$TAG.onnx}"
VIDEO="${NADIR_VIDEO:-docs/nadir_walk_$TAG.mp4}"

module purge
module load python/3.11.7          # no cuda module — see train_leonardo.sbatch
source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs exported_models docs

# JAX must run on the same platform the checkpoint was written on. orbax
# stores device topology, and restoring a GPU-written checkpoint under
# JAX_PLATFORMS=cpu fails with "Topology mismatch detected" unless you hand
# it a target tree — which we cannot do cheaply here, because the saved
# opt_state belongs to whichever optimiser that run used.
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl               # offscreen rendering needs an EGL device

echo "=== $(date -Is) evaluating $CKPT ==="

python -u -m Nadir.export.export_onnx \
    --checkpoint "$CKPT" \
    --output "$ONNX"

python -u -m Nadir.sim.sim2sim \
    --onnx "$ONNX" \
    --seconds "${NADIR_SECONDS:-20}" --seeds "${NADIR_SEEDS:-5}" \
    --video "$VIDEO"

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
fig.suptitle(f"Nadir PPO — {os.environ.get('NADIR_TAG', run)} (A100, MJX)")
fig.tight_layout()
out = f"docs/training_curve_{os.environ.get('NADIR_TAG', run)}.png"
fig.savefig(out, dpi=140)
print("wrote", out)
PY

echo "=== $(date -Is) done ==="
