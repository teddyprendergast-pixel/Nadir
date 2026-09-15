import pytest
import numpy as np
import jax
import jax.numpy as jnp

def test_mjcf_loads():
    """Verify the MJCF model loads without error."""
    import mujoco
    import os
    
    xml_path = 'hardware/nadir.xml'
    if not os.path.exists(xml_path):
        pytest.skip(f"{xml_path} not found")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    assert model.nq > 0
    assert model.nu == 10  # 10 actuators

def test_env_reset():
    """Verify reset returns valid observation shape."""
    from simulation.env_mjx import NadirEnv
    env = NadirEnv(num_envs=1)
    rng = jax.random.PRNGKey(0)
    state, obs, priv_obs = env.reset(jax.random.split(rng, 1))
    assert obs.shape == (1, 41)
    assert priv_obs.shape == (1, 98)

def test_env_step_zero_action():
    """Verify stepping with zero action doesn't crash."""
    from simulation.env_mjx import NadirEnv
    env = NadirEnv(num_envs=1)
    rng = jax.random.PRNGKey(0)
    rng, rng_step = jax.random.split(rng)
    state, obs, priv_obs = env.reset(jax.random.split(rng, 1))
    action = jnp.zeros((1, 10))
    next_state, next_obs, next_priv_obs, reward, done = env.step(jax.random.split(rng_step, 1), state, action)
    assert next_obs.shape == (1, 41)
    assert reward.shape == (1,)

def test_position_actuators():
    """Verify actuators are position-controlled, not torque."""
    import mujoco
    import os
    
    xml_path = 'hardware/nadir.xml'
    if not os.path.exists(xml_path):
        pytest.skip(f"{xml_path} not found")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    for i in range(model.nu):
        assert model.actuator_biastype[i] != 0, f"Actuator {i} is not position-controlled"
