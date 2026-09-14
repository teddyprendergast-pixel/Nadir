from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class RewardConfig:
    tracking_lin_vel: float = 1.0
    tracking_yaw_vel: float = 0.5
    upright: float = 0.5
    base_height: float = 0.2
    action_rate: float = -0.01
    action_jerk: float = -0.005          # Penalize 2nd derivative of actions for smooth curves
    joint_torque: float = -0.0002
    joint_accel: float = -2.5e-7
    base_angular_accel: float = -0.001   # INERTIA PENALTY: Penalize torso rotational wobbling
    base_linear_accel: float = -0.001    # INERTIA PENALTY: Penalize torso linear jerk
    foot_air_time: float = 1.0
    foot_clearance: float = 0.2
    foot_impact: float = -0.01           # Penalize hard foot ground slams
    collision: float = -1.0
    feet_slip: float = -0.05

def velocity_tracking_reward(base_lin_vel, command, sigma=0.25):
    """Exponential kernel reward for tracking commanded velocity."""
    error = jnp.sum(jnp.square(command[:2] - base_lin_vel[:2]))
    return jnp.exp(-error / sigma**2)

def yaw_rate_tracking_reward(base_ang_vel, command, sigma=0.25):
    """Track commanded yaw rate."""
    error = jnp.square(command[2] - base_ang_vel[2])
    return jnp.exp(-error / sigma**2)

def upright_reward(projected_gravity):
    """Reward for keeping torso upright (gravity aligned with -Z)."""
    return jnp.square(projected_gravity[2] + 1.0)

def base_height_reward(base_height, target_height=0.32):
    """Penalize deviation from target standing height."""
    return jnp.exp(-40.0 * jnp.square(base_height - target_height))

def action_rate_penalty(action, previous_action):
    """Penalize jerky motor commands (1st derivative)."""
    return jnp.sum(jnp.square(action - previous_action))

def action_jerk_penalty(action, prev_action, prev_prev_action):
    """Penalize motor command acceleration/jerk (2nd derivative) for fluid curves."""
    jerk = action - 2.0 * prev_action + prev_prev_action
    return jnp.sum(jnp.square(jerk))

def base_angular_accel_penalty(base_ang_vel, previous_ang_vel, dt):
    """INERTIA PENALTY: Penalize torso rotational acceleration (prevents torso wobbling/pitching)."""
    ang_accel = (base_ang_vel - previous_ang_vel) / dt
    return jnp.sum(jnp.square(ang_accel))

def base_linear_accel_penalty(base_lin_vel, previous_lin_vel, dt):
    """INERTIA PENALTY: Penalize torso linear acceleration jerk."""
    lin_accel = (base_lin_vel - previous_lin_vel) / dt
    return jnp.sum(jnp.square(lin_accel))

def joint_torque_penalty(torque):
    """Penalize high joint torques."""
    return jnp.sum(jnp.square(torque))

def joint_acceleration_penalty(joint_vel, previous_joint_vel, dt):
    """Penalize joint acceleration."""
    joint_accel = (joint_vel - previous_joint_vel) / dt
    return jnp.sum(jnp.square(joint_accel))

def foot_air_time_reward(air_time, threshold=0.2):
    """Reward feet spending appropriate time in the air."""
    return jnp.sum(jnp.clip(air_time - threshold, 0.0, 0.5))

def foot_clearance_reward(foot_height, foot_vel_xy, target_clearance=0.04):
    """Reward foot clearance during swing phase."""
    clearance_error = jnp.square(foot_height - target_clearance)
    return jnp.sum(clearance_error * jnp.linalg.norm(foot_vel_xy, axis=-1))

def foot_impact_penalty(foot_vel_z, in_contact):
    """Penalize slamming feet onto the ground at high downward velocity."""
    return jnp.sum(jnp.square(jnp.clip(foot_vel_z, None, 0.0)) * in_contact)

def collision_penalty(contact_forces, forbidden_body_ids=None):
    """Penalize unwanted body contacts."""
    if forbidden_body_ids is None:
        return 0.0
    return jnp.sum(jnp.square(contact_forces))

def feet_slip_penalty(foot_vel, contact_force):
    """Penalize feet sliding while in contact with ground."""
    in_contact = contact_force > 1.0
    return jnp.sum(jnp.square(foot_vel[:2]) * in_contact)

def total_reward(components, weights):
    """Weighted sum of all reward components."""
    return sum(w * r for w, r in zip(weights, components))
