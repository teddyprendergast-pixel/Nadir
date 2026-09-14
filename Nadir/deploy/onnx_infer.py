import numpy as np
import onnxruntime as ort
from typing import Optional

class PolicyInference:
    """Loads and runs the trained ONNX locomotion policy at 50 Hz with action smoothing."""
    
    def __init__(self, model_path: str, num_joints: int = 10, filter_alpha: float = 0.7):
        # Load ONNX model with optimizations for ARM
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # State & Smoothing
        self.filter_alpha = filter_alpha  # 0.7 = 70% new action, 30% previous action
        self.previous_action = np.zeros(num_joints, dtype=np.float32)
        self.smoothed_action = np.zeros(num_joints, dtype=np.float32)
        self.gait_phase = 0.0
        self.gait_frequency = 1.5  # Hz
        self.num_joints = num_joints
        
        # Default standing pose and action scaling
        self.default_positions = np.array([0.0, 0.0, 0.5, -0.3, 0.0,
                                           0.0, 0.0, 0.5, -0.3, 0.0], dtype=np.float32)
        self.action_scale = np.array([0.4, 0.2, 0.6, 0.4, 0.2,
                                      0.4, 0.2, 0.6, 0.4, 0.2], dtype=np.float32)
    
    def build_observation(self, joint_positions: np.ndarray, joint_velocities: np.ndarray,
                          projected_gravity: np.ndarray, angular_velocity: np.ndarray,
                          command: np.ndarray) -> np.ndarray:
        """Construct the 41-dim observation vector."""
        phase_signal = np.array([np.sin(2 * np.pi * self.gait_phase),
                                 np.cos(2 * np.pi * self.gait_phase)], dtype=np.float32)
        obs = np.concatenate([
            projected_gravity,                                    # 3
            angular_velocity,                                     # 3
            joint_positions - self.default_positions,              # 10
            joint_velocities * 0.05,                              # 10
            self.smoothed_action,                                 # 10 (Filtered for smooth feedback)
            command,                                              # 3
            phase_signal,                                         # 2
        ]).astype(np.float32)
        return obs
    
    def predict(self, observation: np.ndarray, dt: float = 0.02) -> np.ndarray:
        """Run policy inference and apply Exponential Low-Pass Filtering for fluid walking."""
        obs_batch = observation.reshape(1, -1)
        raw_action = self.session.run([self.output_name], {self.input_name: obs_batch})[0][0]
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        
        # Low-pass filter to eliminate micro-chatter / high-frequency jitter
        self.smoothed_action = (self.filter_alpha * clipped_action + 
                                (1.0 - self.filter_alpha) * self.smoothed_action)
        
        # Convert to target positions
        target_positions = self.default_positions + self.smoothed_action * self.action_scale
        
        # Update state
        self.previous_action = clipped_action.copy()
        self.gait_phase = (self.gait_phase + self.gait_frequency * dt) % 1.0
        
        return target_positions
