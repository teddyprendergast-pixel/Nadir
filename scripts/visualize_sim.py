import os
import sys

# Auto-relaunch using the project's virtual environment if invoked with a different Python
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
venv_python = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([venv_python] + sys.argv))

import mujoco
import mujoco.viewer
import numpy as np
import argparse
import time

def quat_rotate_inv(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion (wxyz format)."""
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return vec.copy()
    q = quat / norm
    w, x, y, z = q[0], q[1], q[2], q[3]
    xyz = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(xyz, vec)
    return (vec - w * t + np.cross(xyz, t)).astype(np.float32)

def reset_robot(model: mujoco.MjModel, data: mujoco.MjData, default_pos: np.ndarray):
    """Reset the robot to the default standing pose with feet gently touching the floor."""
    mujoco.mj_resetData(model, data)
    data.qpos[7:17] = default_pos
    data.qpos[2] = 0.27  # Lower torso height so feet start directly on the ground
    data.ctrl[:] = default_pos
    mujoco.mj_forward(model, data)

def get_observation(data: mujoco.MjData, default_pos: np.ndarray,
                    command: np.ndarray, previous_action: np.ndarray,
                    gait_phase: float) -> np.ndarray:
    """Build the 41-dim observation vector matching NadirEnv in env_mjx.py."""
    # 1. Project world gravity [0, 0, -1] into the robot body frame
    quat = data.qpos[3:7]  # torso quaternion (w, x, y, z)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    proj_gravity = quat_rotate_inv(quat, gravity_world)

    # 2. Base angular velocity in body frame
    ang_vel = data.qvel[3:6].astype(np.float32)

    # 3. Joint positions relative to default pose
    joint_pos_rel = (data.qpos[7:17] - default_pos).astype(np.float32)

    # 4. Joint velocities scaled by 0.05
    joint_vel_scaled = (data.qvel[6:16] * 0.05).astype(np.float32)

    # 5. Gait phase signal
    phase_signal = np.array([
        np.sin(2.0 * np.pi * gait_phase),
        np.cos(2.0 * np.pi * gait_phase)
    ], dtype=np.float32)

    return np.concatenate([
        proj_gravity,       # 3
        ang_vel,            # 3
        joint_pos_rel,      # 10
        joint_vel_scaled,   # 10
        previous_action,    # 10
        command,            # 3
        phase_signal        # 2
    ]).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Nadir Simulation Viewer")
    parser.add_argument('--policy', type=str, default=None, help="Path to ONNX policy file")
    parser.add_argument('--xml', type=str, default='nadir/sim/nadir.xml', help="Path to MJCF xml")
    parser.add_argument('--vx', type=float, default=0.5, help="Target forward velocity (m/s, default: 0.5)")
    parser.add_argument('--vy', type=float, default=0.0, help="Target lateral velocity (m/s, default: 0.0)")
    parser.add_argument('--yaw', type=float, default=0.0, help="Target yaw rate (rad/s, default: 0.0)")
    args = parser.parse_args()

    print(f"Loading model: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    policy = None
    input_name, output_name = None, None
    if args.policy:
        import onnxruntime as ort
        print(f"Loading policy: {args.policy}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        policy = ort.InferenceSession(args.policy, sess_options)
        input_name = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name

    # Control timing and configuration (50 Hz policy loop)
    policy_dt = 0.02
    decimation = max(1, int(round(policy_dt / model.opt.timestep)))
    print(f"Simulation timestep: {model.opt.timestep * 1000:.1f}ms | Policy rate: 50Hz (decimation: {decimation})")

    default_pos = np.array([0.0, 0.0, 0.5, -0.3, 0.0,
                            0.0, 0.0, 0.5, -0.3, 0.0], dtype=np.float32)
    action_scale = np.array([0.4, 0.2, 0.6, 0.4, 0.2,
                             0.4, 0.2, 0.6, 0.4, 0.2], dtype=np.float32)
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    # Initial standing pose
    reset_robot(model, data, default_pos)
    previous_action = np.zeros(model.nu, dtype=np.float32)
    gait_phase = 0.0

    print("Launching viewer (Close the window or press ESC to exit)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Stability check: auto-reset if robot falls or numerical instability occurs
            if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)) or data.qpos[2] < 0.12 or data.qpos[2] > 1.2:
                print("Robot fell or simulation destabilized. Resetting to standing pose...")
                reset_robot(model, data, default_pos)
                previous_action[:] = 0.0
                gait_phase = 0.0
                viewer.sync()
                continue

            if policy is not None:
                obs = get_observation(data, default_pos, command, previous_action, gait_phase)
                obs_batch = obs.reshape(1, -1)
                raw_action = policy.run([output_name], {input_name: obs_batch})[0][0]
                action = np.clip(raw_action, -1.0, 1.0)
                previous_action = action.copy()

                target_pos = default_pos + action * action_scale
                data.ctrl[:] = target_pos
                gait_phase = (gait_phase + 0.02) % 1.0
            else:
                # Hold default standing pose if no policy loaded
                data.ctrl[:] = default_pos

            # Step physics forward by decimation sub-steps
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Sync with real time (50 Hz)
            elapsed = time.time() - step_start
            sleep_time = policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == '__main__':
    main()
