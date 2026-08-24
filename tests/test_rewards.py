import pytest
import jax.numpy as jnp

def test_velocity_tracking_zero_error():
    """Perfect tracking should give reward = 1.0"""
    try:
        from Nadir.sim.rewards import velocity_tracking_reward
        target = jnp.array([1.0, 0.0, 0.0])
        actual = jnp.array([1.0, 0.0, 0.0])
        reward = velocity_tracking_reward(actual, target)
        assert jnp.isclose(reward, 1.0)
    except ImportError:
        pytest.skip("rewards module not found")

def test_upright_reward():
    """Upright orientation should give maximum reward when gravity aligned with -Z."""
    try:
        from Nadir.sim.rewards import upright_reward
        proj_gravity = jnp.array([0.0, 0.0, -1.0])
        reward = upright_reward(proj_gravity)
        assert jnp.isclose(reward, 0.0)  # (proj_gravity[2] + 1.0)^2 -> (-1.0 + 1.0)^2 = 0
    except ImportError:
        pytest.skip("rewards module not found")

def test_action_rate_zero_change():
    """No change in action should give zero penalty."""
    try:
        from Nadir.sim.rewards import action_rate_penalty
        last_action = jnp.zeros(10)
        curr_action = jnp.zeros(10)
        penalty = action_rate_penalty(last_action, curr_action)
        assert jnp.isclose(penalty, 0.0)
    except ImportError:
        pytest.skip("rewards module not found")
