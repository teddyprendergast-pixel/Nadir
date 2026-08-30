"""Sim2sim validation: run the exported ONNX policy in vanilla MuJoCo.

This is the middle step of the pipeline CLAUDE.md specifies —
train in MJX -> validate sim2sim in vanilla MuJoCo -> export ONNX -> deploy.
Running it here catches two whole classes of bug before any of it reaches a Pi:

1. MJX and the reference MuJoCo engine disagreeing about the model, and
2. the deployed observation vector being assembled differently from the one the
   policy was trained on.

(2) is why this module imports the training env's own `_get_obs` and feeds it a
shim over `MjData` rather than rebuilding the observation by hand. If the
layout changes in training, it changes here too, and cannot silently drift.
"""

import argparse
import os
from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np

from .env_mjx import NadirEnv


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # the Pi runs it single-threaded at 50 Hz
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, obs):
        out = self.sess.run(
            ["action", "goal_position"],
            {self.input_name: np.asarray(obs, dtype=np.float32)[None, :]},
        )
        return out[0][0], out[1][0]


def rollout(policy, env, seconds=20.0, command=(0.4, 0.0, 0.0), seed=0,
            video_path=None, width=640, height=480, fps=50):
    """Run one episode in vanilla MuJoCo. Returns a metrics dict."""
    m, d = env.mj_model, mujoco.MjData(env.mj_model)
    rng = np.random.default_rng(seed)

    d.qpos[0:3] = [0.0, 0.0, env.STANDING_HEIGHT + 0.01]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:] = np.array(env.DEFAULT_POSE) + rng.uniform(-0.05, 0.05, size=env.nu)
    d.ctrl[:] = env.DEFAULT_POSE
    mujoco.mj_forward(m, d)

    cmd = jnp.array(command)
    prev_action = jnp.zeros(env.nu)
    gait_phase = jnp.zeros(1)

    renderer = None
    frames = []
    if video_path:
        renderer = mujoco.Renderer(m, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        cam.distance, cam.azimuth, cam.elevation = 1.2, 120.0, -15.0

    n_steps = int(seconds / env.dt)
    vel_err, heights, uprights = [], [], []
    vx_b, vy_b, wz_b = [], [], []
    path_len = 0.0
    prev_xy = np.array(d.qpos[0:2])
    survived = n_steps

    for i in range(n_steps):
        # Identical observation construction to training.
        shim = SimpleNamespace(qpos=jnp.array(d.qpos), qvel=jnp.array(d.qvel))
        obs = env._get_obs(shim, cmd, prev_action, gait_phase)

        action, goal = policy(np.asarray(obs))
        d.ctrl[:] = goal

        for _ in range(env.decimation):
            mujoco.mj_step(m, d)

        prev_action = jnp.array(action)
        gait_phase = (gait_phase + env.dt) % 1.0

        quat = jnp.array(d.qpos[3:7])
        pg = np.asarray(env._quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0])))
        v_body = np.asarray(env._quat_rotate_inv(quat, jnp.array(d.qvel[0:3])))

        vel_err.append(float(np.linalg.norm(v_body[:2] - np.array(command[:2]))))
        heights.append(float(d.qpos[2]))
        uprights.append(float(-pg[2]))
        vx_b.append(float(v_body[0])); vy_b.append(float(v_body[1]))
        wz_b.append(float(d.qvel[5]))
        xy = np.array(d.qpos[0:2])
        path_len += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy

        if renderer is not None and i % max(1, int(1 / (fps * env.dt))) == 0:
            cam.lookat[:] = d.qpos[0:3]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())

        if pg[2] > 0.0 or d.qpos[2] < 0.15:
            survived = i + 1
            break

    if renderer is not None:
        renderer.close()
        if frames:
            import imageio
            os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
            imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)

    m = lambda a: float(np.mean(a)) if a else float("nan")
    return {
        "command": tuple(float(c) for c in command),
        "survived_steps": survived,
        "survived_seconds": survived * env.dt,
        "completed": survived == n_steps,
        # Net world displacement says nothing on its own: a robot walking in a
        # circle nets ~0 while moving the whole time. Report the path actually
        # walked, and the achieved body-frame velocity against the command,
        # which together distinguish walking from circling from standing.
        "net_displacement_m": float(np.linalg.norm(d.qpos[0:2])),
        "path_length_m": path_len,
        "mean_vx_body": m(vx_b),
        "mean_vy_body": m(vy_b),
        "mean_yaw_rate": m(wz_b),
        "mean_vel_error_ms": m(vel_err),
        "mean_height_m": m(heights),
        "mean_upright": m(uprights),
        "video": video_path,
    }


def main():
    p = argparse.ArgumentParser(description="Validate an ONNX policy in vanilla MuJoCo")
    p.add_argument("--onnx", required=True)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--video", default=None, help="write an mp4 of the first command")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    env = NadirEnv(num_envs=1)
    policy = OnnxPolicy(args.onnx)

    commands = [
        (0.0, 0.0, 0.0),    # stand
        (0.3, 0.0, 0.0),    # walk forward, slow
        (0.5, 0.0, 0.0),    # walk forward, faster
        (-0.2, 0.0, 0.0),   # walk backward
        (0.0, 0.0, 0.8),    # turn in place
        (0.3, 0.15, 0.0),   # diagonal
    ]

    hdr = (f"{'command (vx,vy,wz)':>22} {'done':>6} {'path':>7} {'net':>7} "
           f"{'vx':>16} {'vy':>13} {'wz':>15} {'height':>7}")
    print(hdr)
    print("-" * len(hdr))
    all_completed = 0
    for ci, cmd in enumerate(commands):
        rs = [
            rollout(policy, env, seconds=args.seconds, command=cmd, seed=s,
                    video_path=(args.video if (ci == 1 and s == 0) else None))
            for s in range(args.seeds)
        ]
        comp = sum(r["completed"] for r in rs)
        all_completed += comp
        avg = lambda k: float(np.mean([r[k] for r in rs]))
        print(f"{str(cmd):>22} {comp:>3}/{args.seeds} "
              f"{avg('path_length_m'):>7.2f} {avg('net_displacement_m'):>7.2f} "
              f"{avg('mean_vx_body'):>8.3f}/{cmd[0]:<7.2f} "
              f"{avg('mean_vy_body'):>6.3f}/{cmd[1]:<6.2f} "
              f"{avg('mean_yaw_rate'):>7.3f}/{cmd[2]:<6.2f} "
              f"{avg('mean_height_m'):>6.3f}")

    total = len(commands) * args.seeds
    print("-" * len(hdr))
    print(f"episodes completing the full {args.seconds:.0f}s: {all_completed}/{total}")
    if args.video:
        print(f"video: {args.video}")


if __name__ == "__main__":
    main()
