"""
===============================================================================
Nadir Robot — 3D Interactive Simulation & Policy Visualizer
===============================================================================
What this script does:
  This script opens an interactive 3D window showing the Nadir bipedal robot
  inside the MuJoCo physics simulator. You can:
  1. Watch the robot stand in its neutral pose to inspect the 3D model.
  2. Load a trained AI "brain" (an ONNX neural network policy) and watch it walk!
  3. Change the walking speed and turning rate using command-line arguments.

For non-coders / beginners:
  - "Simulation": A virtual video-game-like world where gravity, friction, and
    electric motors follow real-world physics laws.
  - "Policy": The trained AI neural network that looks at the robot's balance
    and decides how to move its leg joints.
  - "MuJoCo": A fast, high-accuracy physics engine built by Google DeepMind.
===============================================================================
"""

import os
import sys

# ---------------------------------------------------------------------------
# Virtual Environment Auto-Relauncher:
# If you run `python scripts/visualize_sim.py` using your system's default Python,
# this code automatically finds and relaunches using the project's `.venv`
# (virtual environment) so all installed packages (MuJoCo, NumPy, ONNX) work!
# ---------------------------------------------------------------------------
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


# ===========================================================================
# 1. 3D Math Helper: Quaternion Inverse Rotation
# ===========================================================================
def quat_rotate_inv(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Rotate a 3D vector by the inverse of a quaternion (orientation).

    Why this matters for non-coders:
      Imagine the robot has an "inner ear" for balance (just like humans do).
      In the world, gravity always pulls straight down towards the floor: [0, 0, -1].
      A "quaternion" is a 4-number math way to describe which way the robot is
      tilted (pitch, roll, yaw) in 3D space without getting stuck in math errors.
      
      This function takes the world gravity vector and calculates which direction
      gravity feels like it is pulling *from the robot's own body perspective*.
      If the robot tilts forward by 30 degrees, gravity will feel like it is
      pulling diagonally backwards through its chest!
    """
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        # If quaternion is invalid or zero, leave vector unchanged
        return vec.copy()
    
    # Normalize quaternion so its length is exactly 1.0
    q = quat / norm
    w, x, y, z = q[0], q[1], q[2], q[3]
    xyz = np.array([x, y, z], dtype=np.float32)
    
    # Standard quaternion-vector inverse rotation formula
    t = 2.0 * np.cross(xyz, vec)
    return (vec - w * t + np.cross(xyz, t)).astype(np.float32)


# ===========================================================================
# 2. Reset Robot to Standing Position
# ===========================================================================
def reset_robot(model: mujoco.MjModel, data: mujoco.MjData, default_pos: np.ndarray):
    """
    Resets the robot to a stable standing position with feet touching the floor.

    Explanation of MuJoCo data structures:
      - `data.qpos` (Generalized Coordinates):
          Contains the 3D position and rotation of the robot body, PLUS the
          current angles of all 10 leg joints.
          Indices 0 to 2:   Body position in the room (X, Y, Z in meters).
          Indices 3 to 6:   Body rotation as a 4D quaternion (w, x, y, z).
          Indices 7 to 16:  The 10 leg joint angles (hips, knees, ankles in radians).
      - `data.ctrl` (Motor Controls):
          The target angle we are asking each electric servo motor to turn to.
      - `mujoco.mj_forward`:
          Recalculates all physics forces, collision points, and graphics positions.
    """
    # Reset all physics states (velocities to 0, timers to 0)
    mujoco.mj_resetData(model, data)
    
    # Set the 10 leg joints to their neutral standing angles
    data.qpos[7:17] = default_pos
    
    # Set the torso height (Z = 0.27m) so the feet rest gently on the ground
    data.qpos[2] = 0.27
    
    # Tell the motors to hold this standing position
    data.ctrl[:] = default_pos
    
    # Compute the new physical positions and contacts
    mujoco.mj_forward(model, data)


# ===========================================================================
# 3. Build the AI's "Senses" (Observation Vector)
# ===========================================================================
def get_observation(data: mujoco.MjData, default_pos: np.ndarray,
                    command: np.ndarray, previous_action: np.ndarray,
                    gait_phase: float) -> np.ndarray:
    """
    Builds the 41-number "Observation Vector" that feeds into the AI neural network.

    Just like a human needs senses (inner ear balance, muscle feelings, eyes) to walk,
    the robot's AI brain needs 41 specific measurements at every tick to decide its next move:

    Breakdown of the 41 numbers:
      1. Projected Gravity (3 numbers):
         Which way is "down" relative to the robot's body? Tells the robot if it's leaning.
      2. Base Angular Velocity (3 numbers):
         How fast is the body tilting, twisting, or rolling right now? (From the gyro).
      3. Relative Joint Positions (10 numbers):
         Where is each leg motor currently located compared to neutral standing?
      4. Scaled Joint Velocities (10 numbers):
         How fast is each motor currently spinning? (Multiplied by 0.05 so numbers aren't huge).
      5. Previous Action (10 numbers):
         What motor adjustments did the AI request in the last step? This helps the AI
         make smooth, continuous motions rather than jerky, vibrating twitches.
      6. User Command (3 numbers):
         What does the human want the robot to do? [forward_speed, sideways_speed, turning_rate].
      7. Gait Phase Clock (2 numbers):
         A repeating rhythmic clock signal [sin(2*pi*phase), cos(2*pi*phase)] ticking like
         a musical metronome. This helps the robot coordinate left-foot / right-foot stepping.

    Total: 3 + 3 + 10 + 10 + 10 + 3 + 2 = 41 numbers.
    """
    # 1. Project world gravity [0, 0, -1] into the robot's local body frame
    quat = data.qpos[3:7]  # Torso orientation quaternion (w, x, y, z)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    proj_gravity = quat_rotate_inv(quat, gravity_world)

    # 2. Angular velocity of the body in the body frame (gyroscope sensor)
    ang_vel = data.qvel[3:6].astype(np.float32)

    # 3. Leg joint positions relative to the neutral standing pose
    joint_pos_rel = (data.qpos[7:17] - default_pos).astype(np.float32)

    # 4. Joint speeds scaled by 0.05 to keep values in a nice range for neural networks
    joint_vel_scaled = (data.qvel[6:16] * 0.05).astype(np.float32)

    # 5. Rhythmic walking clock (metronome)
    phase_signal = np.array([
        np.sin(2.0 * np.pi * gait_phase),
        np.cos(2.0 * np.pi * gait_phase)
    ], dtype=np.float32)

    # Combine all 41 sensory inputs into one single array
    return np.concatenate([
        proj_gravity,       # 3 numbers
        ang_vel,            # 3 numbers
        joint_pos_rel,      # 10 numbers
        joint_vel_scaled,   # 10 numbers
        previous_action,    # 10 numbers
        command,            # 3 numbers
        phase_signal        # 2 numbers
    ]).astype(np.float32)


# ===========================================================================
# 4. Main Simulation Loop
# ===========================================================================
def main():
    # Set up command-line arguments so the user can easily customize the simulation
    parser = argparse.ArgumentParser(description="Nadir 3D Simulation & Policy Visualizer")
    parser.add_argument('--policy', type=str, default=None,
                        help="Path to an ONNX policy file (e.g. exported_models/nadir_policy.onnx). If omitted, robot stands in place.")
    parser.add_argument('--xml', type=str, default='hardware/nadir.xml',
                        help="Path to the MuJoCo robot model XML file (describes robot 3D shapes, motors, and joints)")
    parser.add_argument('--vx', type=float, default=0.5,
                        help="Desired forward walking speed in meters/second (default: 0.5 m/s)")
    parser.add_argument('--vy', type=float, default=0.0,
                        help="Desired sideways walking speed in meters/second (default: 0.0 m/s)")
    parser.add_argument('--yaw', type=float, default=0.0,
                        help="Desired turning speed in radians/second (default: 0.0 rad/s)")
    args = parser.parse_args()

    # Load the 3D robot model into MuJoCo
    print(f"Loading 3D robot model from: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Optionally load the AI policy (neural network brain)
    policy = None
    input_name, output_name = None, None
    if args.policy:
        import onnxruntime as ort
        print(f"Loading trained AI brain (ONNX policy): {args.policy}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        policy = ort.InferenceSession(args.policy, sess_options)
        input_name = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name

    # -----------------------------------------------------------------------
    # Timing & Decimation:
    # - The AI "brain" thinks at 50 Hz (50 times per second -> every 20 milliseconds: policy_dt = 0.02s).
    # - But physics collisions (feet hitting floor) must be calculated much faster
    #   (e.g., every 2 milliseconds) so feet don't pass through the floor!
    # - "Decimation" is the ratio: how many physics mini-steps MuJoCo calculates
    #   between each single AI brain thought.
    # -----------------------------------------------------------------------
    policy_dt = 0.02  # 50 Hz = 20 milliseconds per thought
    decimation = max(1, int(round(policy_dt / model.opt.timestep)))
    print(f"Physics timestep: {model.opt.timestep * 1000:.1f}ms | AI Decision rate: 50Hz (Decimation: {decimation} physics steps per AI step)")

    # The neutral standing position for the 10 leg joints (in radians)
    # [Left hip yaw, roll, pitch, knee, ankle,  Right hip yaw, roll, pitch, knee, ankle]
    default_pos = np.array([0.0, 0.0, 0.5, -0.3, 0.0,
                            0.0, 0.0, 0.5, -0.3, 0.0], dtype=np.float32)

    # How far the AI is allowed to move each joint from its default standing angle
    # Prevents the neural network from commanding impossible or self-colliding angles!
    action_scale = np.array([0.4, 0.2, 0.6, 0.4, 0.2,
                             0.4, 0.2, 0.6, 0.4, 0.2], dtype=np.float32)
    
    # Human driving command: [forward_speed, sideways_speed, turning_rate]
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    # Reset robot to standing pose and initialize memory
    reset_robot(model, data, default_pos)
    previous_action = np.zeros(model.nu, dtype=np.float32)
    gait_phase = 0.0

    print("\nLaunching 3D Interactive Viewer...")
    print("Controls:")
    print("  - Left click + drag: Rotate camera")
    print("  - Right click + drag: Zoom in/out")
    print("  - Scroll wheel: Zoom")
    print("  - Spacebar: Pause / Resume simulation")
    print("  - Close the window or press ESC to exit.\n")

    # Open the MuJoCo 3D passive graphics window
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # ---------------------------------------------------------------
            # Safety Watchdog & Auto-Reset:
            # If the robot falls over (torso height drops below 0.12m) or flies away,
            # automatically reset it back onto its feet so you don't have to restart the script!
            # ---------------------------------------------------------------
            if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)) or data.qpos[2] < 0.12 or data.qpos[2] > 1.2:
                print("Robot fell or simulation destabilized! Automatically resetting to standing pose...")
                reset_robot(model, data, default_pos)
                previous_action[:] = 0.0
                gait_phase = 0.0
                viewer.sync()
                continue

            # ---------------------------------------------------------------
            # AI Brain Decision Step (runs at 50 Hz):
            # ---------------------------------------------------------------
            if policy is not None:
                # 1. Gather sensory observations (41 numbers)
                obs = get_observation(data, default_pos, command, previous_action, gait_phase)
                obs_batch = obs.reshape(1, -1)
                
                # 2. Ask the neural network what to do
                raw_action = policy.run([output_name], {input_name: obs_batch})[0][0]
                
                # 3. Clip actions between -1.0 and +1.0 for safety
                action = np.clip(raw_action, -1.0, 1.0)
                previous_action = action.copy()

                # 4. Calculate final motor target angles:
                #    Target = Default Standing Angle + (AI Output * Safe Scale)
                target_pos = default_pos + action * action_scale
                data.ctrl[:] = target_pos
                
                # 5. Advance the rhythmic stepping clock (metronome)
                gait_phase = (gait_phase + 0.02) % 1.0
            else:
                # If no AI brain was loaded, hold the default standing pose
                data.ctrl[:] = default_pos

            # ---------------------------------------------------------------
            # Physics Sub-stepping:
            # Advance MuJoCo physics by 'decimation' micro-steps
            # ---------------------------------------------------------------
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            # Redraw the 3D graphics on screen
            viewer.sync()

            # ---------------------------------------------------------------
            # Real-Time Clock Synchronization:
            # Pause slightly if our computer finished the physics faster than
            # real life (20 milliseconds), so the robot moves at true 1.0x speed!
            # ---------------------------------------------------------------
            elapsed = time.time() - step_start
            sleep_time = policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == '__main__':
    main()
