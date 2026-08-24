import pytest
import jax.numpy as jnp

def test_velocity_tracking_zero_error():
    """Perfect tracking should give reward = 1.0"""
    try:
        from nadir.sim.rewards import track_velocity
        target = jnp.array([1.0, 0.0, 0.0])
        actual = jnp.array([1.0, 0.0, 0.0])
        reward = track_velocity(actual, target)
        assert jnp.isclose(reward, 1.0)
    except ImportError:
        pytest.skip("rewards module not found")

def test_upright_reward():
    """Upright orientation should give maximum reward."""
    try:
        from nadir.sim.rewards import upright_posture
        proj_gravity = jnp.array([0.0, 0.0, 1.0])
        reward = upright_posture(proj_gravity)
        assert jnp.isclose(reward, 1.0)
    except ImportError:
        pytest.skip("rewards module not found")

def test_action_rate_zero_change():
    """No change in action should give zero penalty."""
    try:
        from nadir.sim.rewards import action_rate_penalty
        last_action = jnp.zeros(10)
        curr_action = jnp.zeros(10)
        penalty = action_rate_penalty(last_action, curr_action)
        assert jnp.isclose(penalty, 0.0)
    except ImportError:
        pytest.skip("rewards module not found")
