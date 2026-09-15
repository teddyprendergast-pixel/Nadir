import jax.numpy as jnp

def compute_reference_motion(gait_phase: jnp.ndarray, stride_height: float = 0.15, stride_length: float = 0.2) -> jnp.ndarray:
    """Compute smooth cyclic reference joint positions q_ref(phi) for bipedal gait.
    
    gait_phase: scalar or batch float in [0, 1)
    Returns 10-dim joint angle offsets (radians) for:
      [left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee_pitch, left_ankle_pitch,
       right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee_pitch, right_ankle_pitch]
    """
    phi_l = gait_phase
    phi_r = (gait_phase + 0.5) % 1.0  # 180 degrees out of phase
    
    # Smooth sinusoidal leg trajectories
    # Left Leg
    l_hip_pitch = stride_length * jnp.sin(2 * jnp.pi * phi_l)
    l_knee_pitch = stride_height * jnp.maximum(0.0, jnp.sin(2 * jnp.pi * phi_l))
    l_ankle_pitch = -0.5 * l_knee_pitch
    
    # Right Leg
    r_hip_pitch = stride_length * jnp.sin(2 * jnp.pi * phi_r)
    r_knee_pitch = stride_height * jnp.maximum(0.0, jnp.sin(2 * jnp.pi * phi_r))
    r_ankle_pitch = -0.5 * r_knee_pitch
    
    return jnp.array([
        0.0, 0.0, l_hip_pitch, l_knee_pitch, l_ankle_pitch,
        0.0, 0.0, r_hip_pitch, r_knee_pitch, r_ankle_pitch
    ], dtype=jnp.float32)
