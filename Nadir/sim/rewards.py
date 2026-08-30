"""Reward terms for the Nadir walking task.

Sign convention: every function here returns a quantity that is *larger when
the behaviour is better*, except those named `*_penalty`, which return a
non-negative cost. Penalties are given negative weights in `RewardConfig`.
Getting this backwards is not a tuning problem, it is a correctness bug — see
docs/audit-2026-08-30.md.
"""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class RewardConfig:
    # Task terms (positive weight, reward is in [0, 1])
    tracking_lin_vel: float = 1.5
    tracking_yaw_vel: float = 1.0   # raised: with hip yaw this is now achievable
    upright: float = 0.5
    base_height: float = 0.5
    gait_contact: float = 0.5
    foot_clearance: float = 0.5
    feet_air_time: float = 0.3
    alive: float = 0.15

    # Regularisation terms (negative weight, function returns a cost >= 0)
    lin_vel_z: float = -1.0
    ang_vel_xy: float = -0.05
    action_rate: float = -0.02
    joint_vel: float = -1e-3
    joint_limit: float = -1.0
    feet_slip: float = -0.1

    # Gait naturalness. Without these the policy satisfies the contact
    # schedule with a splayed, twitchy, forward-leaning motion that is
    # technically walking and does not look like it.
    orientation: float = -2.0        # torso tilt
    joint_deviation: float = -0.5    # hip yaw/roll and ankle roll drift
    action_smoothness: float = -0.01 # second difference of the command
    stance_width: float = -0.5       # feet under the hips, and not crossing

    # Kernel widths / targets
    tracking_sigma: float = 0.25
    gait_period_s: float = 0.6      # one full left-right cycle
    swing_height_m: float = 0.04    # target sole clearance mid-swing
    move_cmd_threshold: float = 0.05  # |command| below this means 'stand still'
    stance_half_width_m: float = 0.05 # nominal lateral foot offset from centre
    # Measured from the MJCF by forward kinematics: the torso origin height
    # with DEFAULT_POSE and the soles on the floor. Must stay <= the model's
    # straight-legged maximum of 0.2700 m, or this term becomes a permanent
    # incentive to jump. The shipped value was 0.32, which is unreachable.
    target_height: float = 0.2577
    air_time_target: float = 0.25


# --- task terms -------------------------------------------------------------


def velocity_tracking_reward(base_lin_vel_b, command, sigma=0.25):
    """Exponential kernel on planar velocity error. 1.0 = perfect tracking.

    `base_lin_vel_b` must be in the *body* frame: the command is a body-frame
    velocity, so comparing it against a world-frame velocity silently asks the
    robot to walk north regardless of which way it is facing.
    """
    error = jnp.sum(jnp.square(command[:2] - base_lin_vel_b[:2]))
    return jnp.exp(-error / (sigma**2))


def yaw_rate_tracking_reward(base_ang_vel_b, command, sigma=0.25):
    """Exponential kernel on yaw-rate error. 1.0 = perfect tracking."""
    error = jnp.square(command[2] - base_ang_vel_b[2])
    return jnp.exp(-error / (sigma**2))


def upright_reward(projected_gravity):
    """cos(tilt). 1.0 when upright, 0.0 when horizontal, clipped below at 0.

    `projected_gravity` is the world down-vector [0, 0, -1] expressed in the
    body frame, so its z component is -1 when the torso is upright. The
    previous implementation returned (pg_z + 1)**2, which is 0 when upright
    and 4 when inverted, with a positive weight — it paid the robot to fall.
    """
    return jnp.clip(-projected_gravity[2], 0.0, 1.0)


def base_height_reward(base_height, target_height=0.2577):
    """Gaussian around the target standing height. 1.0 at the target."""
    return jnp.exp(-40.0 * jnp.square(base_height - target_height))


def gait_contact_reward(contact, phase, moving=1.0):
    """Reward matching an alternating left/right contact schedule.

    The environment already feeds the policy a gait-phase clock in its
    observation, but nothing rewarded following it — so the policy had a
    clock and no reason to use it, and settled into a flat-footed shuffle.
    This closes that loop: the right foot should be in stance for the first
    half of the cycle and the left foot for the second.

    `moving` gates the term on command magnitude. Applied unconditionally it
    pays the robot to march on the spot when told to stand still — measured
    at 49M steps as standing path length rising from 0.52 m to 1.19 m. A
    stationary command should reward *both* feet planted instead.
    """
    desired_stance_r = phase < 0.5
    desired_stance_l = jnp.logical_not(desired_stance_r)
    stepping = jnp.stack([desired_stance_r, desired_stance_l]).reshape(2)
    planted = jnp.ones(2, dtype=bool)
    desired = jnp.where(moving > 0.5, stepping, planted)
    return jnp.mean((contact == desired).astype(jnp.float32))


def foot_clearance_reward(sole_height, phase, target=0.03, sigma=0.02, moving=1.0):
    """Reward the *swing* foot lifting to a target height.

    Without this the cheapest way to satisfy the contact schedule is to barely
    unweight a foot while sliding it, which reads as a shuffle and transfers
    badly to a real robot with backlash and finite servo bandwidth.
    """
    swing_r = phase >= 0.5
    swing = jnp.stack([swing_r, jnp.logical_not(swing_r)]).reshape(2)
    close = jnp.exp(-jnp.square((sole_height - target) / sigma))
    # Gated like gait_contact_reward: no reason to lift a foot when the
    # command is to stand still.
    return jnp.mean(close * swing.astype(jnp.float32)) * jnp.clip(moving, 0.0, 1.0)


def feet_air_time_reward(air_time, first_contact, target=0.25):
    """Pay out accumulated swing time at the moment each foot lands.

    Rewarding air time continuously encourages hopping and standing on one
    leg; paying it only on touchdown, relative to a target swing duration, is
    what produces alternating steps rather than a shuffle.
    """
    return jnp.sum((air_time - target) * first_contact)


def alive_reward():
    """Constant bonus per surviving step, to offset early-termination bias."""
    return 1.0


# --- penalties (return a non-negative cost; weights are negative) -----------


def lin_vel_z_penalty(base_lin_vel_b):
    """Discourage bouncing."""
    return jnp.square(base_lin_vel_b[2])


def ang_vel_xy_penalty(base_ang_vel_b):
    """Discourage roll/pitch rate."""
    return jnp.sum(jnp.square(base_ang_vel_b[:2]))


def action_rate_penalty(action, previous_action):
    """Penalise jerky command changes.

    On position-controlled servos this is the term that keeps goal positions
    trackable by the internal loop; without it the policy learns to chatter at
    the control rate, which the real STS3215 cannot follow.
    """
    return jnp.sum(jnp.square(action - previous_action))


def joint_velocity_penalty(joint_vel):
    return jnp.sum(jnp.square(joint_vel))


def joint_limit_penalty(joint_pos, lower, upper, margin=0.05):
    """Cost for driving a joint into the last `margin` radians of its range."""
    below = jnp.clip((lower + margin) - joint_pos, 0.0, None)
    above = jnp.clip(joint_pos - (upper - margin), 0.0, None)
    return jnp.sum(jnp.square(below) + jnp.square(above))


def feet_slip_penalty(foot_vel_xy, in_contact):
    """Penalise horizontal foot motion while loaded."""
    return jnp.sum(jnp.sum(jnp.square(foot_vel_xy), axis=-1) * in_contact)


def orientation_penalty(projected_gravity):
    """Penalise torso tilt directly.

    `upright_reward` is cos(tilt), which is flat near upright — at 20 degrees
    it still pays 0.94, so it barely discourages a persistent lean. The
    squared horizontal components of projected gravity have their steepest
    gradient exactly where we care, at small tilt.
    """
    return jnp.sum(jnp.square(projected_gravity[:2]))


def joint_deviation_penalty(joint_pos, default_pose, mask):
    """Penalise drift from the nominal pose on selected joints.

    Applied to hip yaw, hip roll and ankle roll. Left unconstrained these
    splay the legs outward and roll the feet, which satisfies every other
    reward term while looking nothing like walking. Hip pitch, knee and ankle
    pitch are deliberately excluded — those are the joints that must move
    freely to produce a stride.
    """
    return jnp.sum(jnp.square((joint_pos - default_pose) * mask))


def action_smoothness_penalty(action, prev_action, prev_action_2):
    """Second difference of the commanded position.

    `action_rate_penalty` penalises the velocity of the goal-position signal;
    this penalises its acceleration. That is what removes the high-frequency
    twitch a position-controlled servo cannot reproduce anyway.
    """
    return jnp.sum(jnp.square(action - 2.0 * prev_action + prev_action_2))


def stance_width_penalty(foot_y_body, half_width=0.05):
    """Keep the feet under the hips, and stop them crossing.

    `foot_y_body` is (right_y, left_y) in the torso frame, so the right foot
    should sit near -half_width and the left near +half_width.
    """
    target = jnp.array([-half_width, half_width])
    offset = jnp.sum(jnp.square(foot_y_body - target))
    # Explicit crossing term: right foot to the left of the left foot.
    crossed = jnp.clip(foot_y_body[0] - foot_y_body[1], 0.0, None)
    return offset + jnp.square(crossed)


def total_reward(components, weights):
    """Weighted sum. `components` and `weights` are matched sequences."""
    return sum(w * r for w, r in zip(weights, components))
