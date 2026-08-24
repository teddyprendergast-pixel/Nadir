import pytest
import numpy as np

def test_mjcf_loads():
    """Verify the MJCF model loads without error."""
    import mujoco
    import os
    
    xml_path = 'Nadir/sim/nadir.xml'
    if not os.path.exists(xml_path):
        pytest.skip(f"{xml_path} not found")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    assert model.nq > 0
    assert model.nu == 10  # 10 actuators

def test_env_reset():
    """Verify reset returns valid observation shape."""
    try:
        from Nadir.sim.env_mjx import NadirEnv
        import jax
        env = NadirEnv(num_envs=1)
        rngs = jax.random.split(jax.random.PRNGKey(0), env.num_envs)
        state = env.reset(rngs)
        assert state.obs.shape == (1, 41)
    except ImportError:
        pytest.skip("NadirEnv import failed")

def test_env_step_zero_action():
    """Verify stepping with zero action doesn't crash."""
    try:
        from Nadir.sim.env_mjx import NadirEnv
        import jax
        import jax.numpy as jnp
        env = NadirEnv(num_envs=1)
        rngs = jax.random.split(jax.random.PRNGKey(0), env.num_envs)
        state = env.reset(rngs)
        next_state = env.step(state, jnp.zeros((1, 10)))
        assert next_state.obs.shape == (1, 41)
    except ImportError:
        pytest.skip("NadirEnv import failed")

def test_position_actuators():
    """Verify actuators are position-controlled, not torque."""
    import mujoco
    import os
    
    xml_path = 'Nadir/sim/nadir.xml'
    if not os.path.exists(xml_path):
        pytest.skip(f"{xml_path} not found")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    # Check all actuators use position control
    for i in range(model.nu):
        # bias_type 1 is position control in MuJoCo MJCF
        # 0 is usually none/motor
        assert model.actuator_biastype[i] != 0, f"Actuator {i} is not position-controlled"
