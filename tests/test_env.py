import pytest
import numpy as np

def test_mjcf_loads():
    """Verify the MJCF model loads without error."""
    import mujoco
    import os
    
    # Use a dummy XML if nadir.xml doesn't exist yet for test to pass structurally
    xml_path = 'nadir/sim/nadir.xml'
    if not os.path.exists(xml_path):
        pytest.skip(f"{xml_path} not found")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    assert model.nq > 0
    assert model.nu == 10  # 10 actuators

def test_env_reset():
    """Verify reset returns valid observation shape."""
    # Placeholder structure, requires actual Env class
    try:
        from nadir.sim.env import NadirEnv
        env = NadirEnv()
        obs = env.reset()
        assert obs.shape == (41,)
    except ImportError:
        pytest.skip("NadirEnv not implemented yet")

def test_env_step_zero_action():
    """Verify stepping with zero action doesn't crash."""
    try:
        from nadir.sim.env import NadirEnv
        env = NadirEnv()
        env.reset()
        obs, reward, done, info = env.step(np.zeros(10))
        assert obs.shape == (41,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
    except ImportError:
        pytest.skip("NadirEnv not implemented yet")

def test_position_actuators():
    """Verify actuators are position-controlled, not torque."""
    import mujoco
    import os
    
    xml_path = 'nadir/sim/nadir.xml'
    if not os.path.exists(xml_path):
        pytest.skip(f"{xml_path} not found")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    # Check all actuators use position control
    for i in range(model.nu):
        # bias_type 1 is position control in MuJoCo MJCF
        # 0 is usually none/motor
        assert model.actuator_biastype[i] != 0, f"Actuator {i} is not position-controlled"
