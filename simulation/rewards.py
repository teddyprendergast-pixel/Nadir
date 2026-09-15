"""
===============================================================================
Nadir Robot — Reinforcement Learning Reward & Penalty System
===============================================================================
What this file does:
  This file defines how the AI robot is "scored" during training.
  In Reinforcement Learning (RL), the robot learns by maximizing its score:
    - Positive rewards (+): "Treats" earned for doing good things (walking at the
      desired speed, standing tall, stepping with clean rhythm).
    - Negative rewards (-): "Penalties" deducted for doing bad things (falling over,
      shaking violently, stomping hard on the ground, or sliding like ice skates).

Why this matters for non-coders / beginners:
  Without carefully designed rewards, a robot might learn bizarre cheat strategies!
  For example, it might figure out that falling forward like a falling tree covers
  distance quickly, or it might shake its legs at 50 Hz to vibrate forward.
  These reward formulas teach the robot to walk smoothly, softly, and realistically
  just like a real living creature.
===============================================================================
"""

from dataclasses import dataclass
import jax.numpy as jnp


# ===========================================================================
# Reward Weights Configuration
# ===========================================================================
@dataclass
class RewardConfig:
    """
    Multiplier weights for each reward and penalty term.
    Positive values (+) encourage behaviors; negative values (-) discourage behaviors.
    """
    # Positive Rewards (Good behaviors we want to see):
    tracking_lin_vel: float = 1.0        # Reward for walking at the exact commanded forward/lateral speed
    tracking_yaw_vel: float = 0.5        # Reward for turning at the requested rotation rate
    upright: float = 0.5                 # Reward for keeping torso pointing straight up towards the sky
    base_height: float = 0.2             # Reward for keeping body at the ideal standing height (~32 cm)
    foot_air_time: float = 1.0           # Reward for swinging feet cleanly in the air during steps
    foot_clearance: float = 0.2          # Reward for lifting feet safely above the ground during swing phase

    # Penalties (Bad behaviors we want to eliminate):
    action_rate: float = -0.01           # Penalty for twitchy, sudden jumps in motor angles
    action_jerk: float = -0.005          # Penalty for rapid acceleration changes (keeps motion fluid and smooth)
    joint_torque: float = -0.0002        # Penalty for straining motors with excessive force
    joint_accel: float = -2.5e-7         # Penalty for violent joint acceleration
    base_angular_accel: float = -0.001   # INERTIA PENALTY: Stops the torso from pitching and wobbling
    base_linear_accel: float = -0.001    # INERTIA PENALTY: Stops the torso from jerking up and down
    foot_impact: float = -0.01           # Penalty for stomping feet hard into the floor (saves 3D printed parts!)
    collision: float = -1.0              # Huge penalty for knees or body slamming into the ground
    feet_slip: float = -0.05             # Penalty for feet sliding like ice skates while touching the floor


# ===========================================================================
# Individual Reward & Penalty Formulas
# ===========================================================================

def velocity_tracking_reward(base_lin_vel, command, sigma=0.25):
    """
    Reward for matching the commanded forward and sideways walking speed.
    
    Math explanation:
      Uses an exponential bell-curve (Gaussian):
      - If speed error is 0.0, score = 1.0 (100% perfect score).
      - As the robot walks too slow or too fast, score drops smoothly towards 0.0.
    """
    error = jnp.sum(jnp.square(command[:2] - base_lin_vel[:2]))
    return jnp.exp(-error / sigma**2)


def yaw_rate_tracking_reward(base_ang_vel, command, sigma=0.25):
    """
    Reward for matching the commanded turning rate (yaw rotation).
    Gives a perfect 1.0 if the robot rotates at the exact speed requested by the user.
    """
    error = jnp.square(command[2] - base_ang_vel[2])
    return jnp.exp(-error / sigma**2)


def upright_reward(projected_gravity):
    """
    Reward for keeping the torso upright.
    
    Explanation:
      Gravity pulls along [0, 0, -1] in 3D world coordinates.
      If the robot is standing completely upright, the projected gravity Z component
      is -1.0. Adding 1.0 gives 0.0 deviation.
    """
    return jnp.square(projected_gravity[2] + 1.0)


def base_height_reward(base_height, target_height=0.32):
    """
    Reward for keeping the torso at the ideal standing height (default: 32 cm).
    Prevents the robot from crouching too low like a crab or jumping too high.
    """
    return jnp.exp(-40.0 * jnp.square(base_height - target_height))


def action_rate_penalty(action, previous_action):
    """
    Penalizes sudden jumps in motor commands between consecutive time steps (1st derivative).
    Encourages continuous, steady joint trajectories.
    """
    return jnp.sum(jnp.square(action - previous_action))


def action_jerk_penalty(action, prev_action, prev_prev_action):
    """
    Penalizes sudden acceleration changes in motor commands (2nd derivative).
    
    Why this matters for physical hardware:
      Think of a car driver who constantly taps and releases the gas pedal.
      Even if the speed is okay, the ride feels jerky!
      Penalizing jerk produces silky-smooth S-curve motions that protect the
      plastic gears inside the Feetech servo motors from stripping.
    """
    jerk = action - 2.0 * prev_action + prev_prev_action
    return jnp.sum(jnp.square(jerk))


def base_angular_accel_penalty(base_ang_vel, previous_ang_vel, dt):
    """
    INERTIA PENALTY: Penalizes rapid angular acceleration of the torso.
    Keeps the camera view stable and stops the robot body from nodding or rocking.
    """
    ang_accel = (base_ang_vel - previous_ang_vel) / dt
    return jnp.sum(jnp.square(ang_accel))


def base_linear_accel_penalty(base_lin_vel, previous_lin_vel, dt):
    """
    INERTIA PENALTY: Penalizes sudden linear jerks of the torso.
    Encourages a steady, floating forward glide instead of bouncy stutter-stepping.
    """
    lin_accel = (base_lin_vel - previous_lin_vel) / dt
    return jnp.sum(jnp.square(lin_accel))


def joint_torque_penalty(torque):
    """
    Penalizes excessive motor torque (force).
    Saves battery power and prevents the motors from overheating.
    """
    return jnp.sum(jnp.square(torque))


def joint_acceleration_penalty(joint_vel, previous_joint_vel, dt):
    """
    Penalizes high acceleration of the leg joints.
    Helps avoid violent leg whipping motions.
    """
    joint_accel = (joint_vel - previous_joint_vel) / dt
    return jnp.sum(jnp.square(joint_accel))


def foot_air_time_reward(air_time, threshold=0.2):
    """
    Rewards lifting feet and keeping them in the air long enough to take a real step.
    Prevents the robot from shuffling its feet across the floor like a penguin.
    """
    return jnp.sum(jnp.clip(air_time - threshold, 0.0, 0.5))


def foot_clearance_reward(foot_height, foot_vel_xy, target_clearance=0.04):
    """
    Rewards lifting feet to a safe height (~4 cm) while swinging forward.
    Essential for forest floors to avoid tripping on roots, rocks, and moss.
    """
    clearance_error = jnp.square(foot_height - target_clearance)
    return jnp.sum(clearance_error * jnp.linalg.norm(foot_vel_xy, axis=-1))


def foot_impact_penalty(foot_vel_z, in_contact):
    """
    Penalizes slamming feet down hard onto the floor at high downward speed.
    Teaches the robot to step gently on its feet like a human, preventing
    chassis vibration and protecting the 3D-printed foot brackets.
    """
    return jnp.sum(jnp.square(jnp.clip(foot_vel_z, None, 0.0)) * in_contact)


def collision_penalty(contact_forces, forbidden_body_ids=None):
    """
    Heavy penalty for parts of the robot hitting each other or hitting the ground
    (e.g., knees banging together or torso crashing into the floor).
    """
    if forbidden_body_ids is None:
        return 0.0
    return jnp.sum(jnp.square(contact_forces))


def feet_slip_penalty(foot_vel, contact_force):
    """
    Penalizes feet sliding along the floor while bearing weight.
    Ensures that when a foot touches the ground, it plants firmly for traction
    instead of slipping like it's on wet ice.
    """
    in_contact = contact_force > 1.0
    return jnp.sum(jnp.square(foot_vel[:2]) * in_contact)


# ===========================================================================
# Total Combined Reward
# ===========================================================================
def total_reward(components, weights):
    """
    Calculates the final overall score by multiplying each reward/penalty component
    by its corresponding weight in RewardConfig and summing them together.
    """
    return sum(w * r for w, r in zip(weights, components))
