"""Reward-term tests.

These exist because the reward that shipped in f751b86 was sign-inverted: it
returned 0 when the robot was upright and 4 when it was inverted, with a
positive weight. Nothing caught it, because every test in this file used to
import a function name that did not exist and skip on ImportError.
"""

import numpy as np
import pytest

from Nadir.sim import rewards as R

UPRIGHT = np.array([0.0, 0.0, -1.0])      # world down-vector in a level body frame
HORIZONTAL = np.array([1.0, 0.0, 0.0])    # torso on its side
INVERTED = np.array([0.0, 0.0, 1.0])      # torso upside down


def test_upright_reward_is_maximal_when_upright():
    assert R.upright_reward(UPRIGHT) == pytest.approx(1.0)


def test_upright_reward_is_zero_when_horizontal_or_worse():
    assert R.upright_reward(HORIZONTAL) == pytest.approx(0.0)
    assert R.upright_reward(INVERTED) == pytest.approx(0.0)


def test_upright_reward_is_monotone_in_tilt():
    """Regression guard for the inverted sign: tilting must never pay more."""
    tilts = np.linspace(0.0, np.pi, 25)
    vals = [float(R.upright_reward(np.array([np.sin(t), 0.0, -np.cos(t)]))) for t in tilts]
    assert all(a >= b - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_velocity_tracking_peaks_at_the_commanded_velocity():
    cmd = np.array([0.4, 0.0, 0.0])
    perfect = R.velocity_tracking_reward(np.array([0.4, 0.0, 0.0]), cmd)
    off = R.velocity_tracking_reward(np.array([0.0, 0.0, 0.0]), cmd)
    wrong_way = R.velocity_tracking_reward(np.array([-0.4, 0.0, 0.0]), cmd)
    assert perfect == pytest.approx(1.0)
    assert perfect > off > wrong_way


def test_yaw_tracking_peaks_at_the_commanded_rate():
    cmd = np.array([0.0, 0.0, 0.8])
    assert R.yaw_rate_tracking_reward(np.array([0.0, 0.0, 0.8]), cmd) == pytest.approx(1.0)
    assert (R.yaw_rate_tracking_reward(np.array([0.0, 0.0, 0.8]), cmd)
            > R.yaw_rate_tracking_reward(np.array([0.0, 0.0, -0.8]), cmd))


def test_base_height_reward_peaks_at_target_and_is_reachable():
    """The target must be a height the robot can actually stand at.

    The shipped config targeted 0.32 m; the model's straight-legged maximum is
    0.2700 m, so the term was an unbounded incentive to jump.
    """
    from Nadir.sim.env_mjx import NadirEnv

    target = R.RewardConfig().target_height
    assert R.base_height_reward(target, target) == pytest.approx(1.0)
    assert target <= NadirEnv.STANDING_HEIGHT + 1e-6
    assert target < 0.2700  # straight-legged maximum, measured from the MJCF


def test_penalties_are_non_negative():
    """Penalty helpers must return a cost; the weights carry the sign."""
    assert R.action_rate_penalty(np.ones(10), np.zeros(10)) >= 0
    assert R.joint_velocity_penalty(np.full(10, -3.0)) >= 0
    assert R.lin_vel_z_penalty(np.array([0.0, 0.0, -2.0])) >= 0
    assert R.ang_vel_xy_penalty(np.array([1.0, -1.0, 0.0])) >= 0


def test_action_rate_penalty_is_zero_for_a_held_action():
    a = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.1, -0.2, 0.3, 0.0, 0.0])
    assert R.action_rate_penalty(a, a) == pytest.approx(0.0)


def test_joint_limit_penalty_only_bites_near_the_limits():
    lower, upper = np.full(3, -1.0), np.full(3, 1.0)
    assert R.joint_limit_penalty(np.zeros(3), lower, upper) == pytest.approx(0.0)
    assert R.joint_limit_penalty(np.full(3, 0.999), lower, upper) > 0.0


def test_feet_air_time_pays_only_on_touchdown():
    air = np.array([0.4, 0.4])
    assert R.feet_air_time_reward(air, np.array([0.0, 0.0])) == pytest.approx(0.0)
    assert R.feet_air_time_reward(air, np.array([1.0, 0.0])) > 0.0


def test_gait_contact_rewards_alternating_stance():
    """The env feeds the policy a gait clock; this is what makes it mean something."""
    both_down = np.array([True, True])
    r_down_l_up = np.array([True, False])
    l_down_r_up = np.array([False, True])

    # first half of the cycle: right foot should be down, left up
    assert R.gait_contact_reward(r_down_l_up, 0.25) == pytest.approx(1.0)
    assert R.gait_contact_reward(l_down_r_up, 0.25) == pytest.approx(0.0)
    # second half: the other way round
    assert R.gait_contact_reward(l_down_r_up, 0.75) == pytest.approx(1.0)
    assert R.gait_contact_reward(r_down_l_up, 0.75) == pytest.approx(0.0)
    # standing on both feet only ever half-satisfies the schedule
    assert R.gait_contact_reward(both_down, 0.25) == pytest.approx(0.5)


def test_gait_rewards_are_gated_on_a_movement_command():
    """A robot told to stand still must not be paid to march on the spot.

    Applied unconditionally, the gait terms raised standing path length from
    0.52 m to 1.19 m at 49M steps — the policy stepped in place because that
    is what the contact schedule rewarded.
    """
    both_down = np.array([True, True])
    r_down_l_up = np.array([True, False])

    # moving: alternating stance is what scores
    assert R.gait_contact_reward(r_down_l_up, 0.25, moving=1.0) == pytest.approx(1.0)
    assert R.gait_contact_reward(both_down, 0.25, moving=1.0) == pytest.approx(0.5)

    # standing: both feet planted is what scores, stepping is penalised
    assert R.gait_contact_reward(both_down, 0.25, moving=0.0) == pytest.approx(1.0)
    assert R.gait_contact_reward(r_down_l_up, 0.25, moving=0.0) == pytest.approx(0.5)

    # and there is no reward for lifting a foot while standing
    assert R.foot_clearance_reward(
        np.array([0.0, 0.03]), 0.25, 0.03, moving=0.0
    ) == pytest.approx(0.0)


def test_foot_clearance_rewards_lifting_the_swing_foot():
    target = 0.03
    # phase < 0.5 -> left foot is swinging
    lifted = np.array([0.0, target])
    dragging = np.array([0.0, 0.0])
    assert R.foot_clearance_reward(lifted, 0.25, target) > \
           R.foot_clearance_reward(dragging, 0.25, target)
    # the stance foot's height is irrelevant — only the swing foot is scored,
    # so raising the stance foot must not change the reward at all
    swing_only = R.foot_clearance_reward(np.array([0.0, 0.0]), 0.25, target)
    stance_raised = R.foot_clearance_reward(np.array([target, 0.0]), 0.25, target)
    assert stance_raised == pytest.approx(swing_only)


def test_total_reward_is_a_weighted_sum():
    assert R.total_reward([1.0, 2.0, 3.0], [1.0, -1.0, 0.5]) == pytest.approx(0.5)


# --- gait naturalness -------------------------------------------------------


def test_orientation_penalty_is_steeper_than_upright_reward_near_vertical():
    """The reason both terms exist.

    cos(tilt) is flat near upright — at 20 degrees it still pays 0.94 — so it
    barely discourages a persistent lean. The squared horizontal projected
    gravity is steepest exactly there.
    """
    def pg(tilt):
        return np.array([np.sin(tilt), 0.0, -np.cos(tilt)])

    small = np.deg2rad(10.0)
    upright_drop = float(R.upright_reward(pg(0.0)) - R.upright_reward(pg(small)))
    orient_rise = float(R.orientation_penalty(pg(small)) - R.orientation_penalty(pg(0.0)))
    assert orient_rise > upright_drop
    assert R.orientation_penalty(pg(0.0)) == pytest.approx(0.0)


def test_joint_deviation_penalty_only_scores_masked_joints():
    default = np.zeros(12)
    mask = np.zeros(12); mask[[0, 2, 5, 6, 8, 11]] = 1.0

    swing_joints = np.zeros(12); swing_joints[[1, 3, 4]] = 1.0   # pitch/knee
    splayed = np.zeros(12); splayed[[0, 2]] = 1.0                # yaw/roll

    # the joints that must move to make a stride are not penalised
    assert R.joint_deviation_penalty(swing_joints, default, mask) == pytest.approx(0.0)
    # the ones that make the legs splay are
    assert R.joint_deviation_penalty(splayed, default, mask) > 0.0


def test_action_smoothness_penalises_acceleration_not_velocity():
    a = np.ones(12)
    # constant velocity ramp: rate penalty bites, smoothness does not
    assert R.action_smoothness_penalty(2 * a, a, 0 * a) == pytest.approx(0.0)
    assert R.action_rate_penalty(2 * a, a) > 0.0
    # a reversal is pure acceleration
    assert R.action_smoothness_penalty(0 * a, a, 0 * a) > 0.0


def test_stance_width_penalty_rewards_feet_under_the_hips():
    hw = 0.05
    nominal = np.array([-hw, hw])          # right foot -y, left foot +y
    assert R.stance_width_penalty(nominal, hw) == pytest.approx(0.0)
    # too wide, and too narrow, both cost
    assert R.stance_width_penalty(np.array([-0.12, 0.12]), hw) > 0.0
    assert R.stance_width_penalty(np.array([0.0, 0.0]), hw) > 0.0


def test_stance_width_penalty_punishes_crossed_feet():
    hw = 0.05
    crossed = np.array([hw, -hw])          # right foot now on the left side
    normal = np.array([-hw, hw])
    assert R.stance_width_penalty(crossed, hw) > R.stance_width_penalty(normal, hw)
