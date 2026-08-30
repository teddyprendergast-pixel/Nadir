import mujoco
import mujoco.viewer
import numpy as np
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Nadir Simulation Viewer")
    parser.add_argument('--policy', type=str, default=None, help="Path to ONNX policy file")
    parser.add_argument('--xml', type=str, default='Nadir/sim/nadir.xml', help="Path to MJCF xml")
    args = parser.parse_args()
    
    print(f"Loading model: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    
    policy = None
    if args.policy:
        import onnxruntime as ort
        print(f"Loading policy: {args.policy}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        policy = ort.InferenceSession(args.policy, sess_options)
        input_name = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name
        
    # Variables for control
    dt = model.opt.timestep
    gait_phase = 0.0
    previous_action = np.zeros(model.nu, dtype=np.float32)
    default_pos = np.array([0.0, 0.0, 0.5, -0.3, 0.0, 0.0, 0.0, 0.5, -0.3, 0.0], dtype=np.float32)
    action_scale = np.array([0.4, 0.2, 0.6, 0.4, 0.2, 0.4, 0.2, 0.6, 0.4, 0.2], dtype=np.float32)
    command = np.array([0.0, 0.0, 0.0], dtype=np.float32) # vx, vy, yaw_rate
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            if policy is not None:
                # Extract observations from mujoco data
                # Simplified dummy values for missing state estimators
                proj_gravity = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                ang_vel = np.zeros(3, dtype=np.float32)
                
                joint_pos = data.qpos[7:7+model.nu].astype(np.float32)
                joint_vel = data.qvel[6:6+model.nu].astype(np.float32)
                
                phase_signal = np.array([np.sin(2 * np.pi * gait_phase),
                                         np.cos(2 * np.pi * gait_phase)], dtype=np.float32)
                
                obs = np.concatenate([
                    proj_gravity,
                    ang_vel,
                    joint_pos - default_pos,
                    joint_vel * 0.05,
                    previous_action,
                    command,
                    phase_signal
                ])
                
                obs_batch = obs.reshape(1, -1)
                action = policy.run([output_name], {input_name: obs_batch})[0][0]
                action = np.clip(action, -1.0, 1.0)
                previous_action = action.copy()
                gait_phase = (gait_phase + 1.5 * dt) % 1.0
                
                target_pos = default_pos + action * action_scale
                data.ctrl[:] = target_pos
                
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
