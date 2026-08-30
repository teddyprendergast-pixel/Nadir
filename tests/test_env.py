"""Environment and model tests.

The previous version of this file imported `nadir.sim.env` (lowercase package,
module that does not exist) inside try/except ImportError and skipped. All of
it passed, and none of it ran.
"""

import os

import mujoco
import numpy as np
import pytest

XML = os.path.join(os.path.dirname(__file__), "..", "Nadir", "sim", "nadir.xml")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(os.path.abspath(XML))


# --- model ----------------------------------------------------------------


def test_mjcf_loads_with_expected_dimensions(model):
    assert model.nu == 10                    # 10 actuated joints
    assert model.nq == 17                    # free joint (7) + 10 hinges
    assert model.nv == 16                    # free joint (6) + 10 hinges


def test_all_actuators_are_position_controlled(model):
    """Non-negotiable #1: the STS3215 closes its own loop. Never <motor>."""
    for i in range(model.nu):
        assert model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE, (
            f"actuator {i} is not a position actuator"
        )
        # A <position> actuator has bias = [0, -kp, -kv].
        assert model.actuator_biasprm[i][1] < 0


def test_ctrlrange_covers_the_full_joint_range(model):
    """ctrl is a goal angle in radians, so ctrlrange must equal jnt_range.

    The shipped MJCF used ctrlrange="-1 1" on every actuator while the knees
    range to 2.0 rad and the hip pitches to 1.2 rad, so MuJoCo silently
    clamped a large part of the commanded action space.
    """
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        lo_j, hi_j = model.jnt_range[jid]
        lo_c, hi_c = model.actuator_ctrlrange[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        assert lo_c <= lo_j + 1e-9 and hi_c >= hi_j - 1e-9, (
            f"{name}: ctrlrange [{lo_c}, {hi_c}] clips joint range [{lo_j}, {hi_j}]"
        )


def test_total_mass_is_in_the_design_envelope(model):
    """~1.5 kg desktop-scale biped."""
    assert 0.8 < sum(model.body_mass) < 2.0


def test_standing_height_constant_matches_the_model(model):
    """STANDING_HEIGHT must be derived from the MJCF, not typed in by hand."""
    from Nadir.sim.env_mjx import NadirEnv

    d = mujoco.MjData(model)
    d.qpos[0:3] = [0.0, 0.0, 1.0]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:] = NadirEnv.DEFAULT_POSE
    mujoco.mj_forward(model, d)
    gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("foot_geom_r", "foot_geom_l")]
    sole = min(d.geom_xpos[g][2] - 0.005 for g in gids)
    assert NadirEnv.STANDING_HEIGHT == pytest.approx(1.0 - sole, abs=1e-3)


# --- environment ----------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    from Nadir.sim.env_mjx import NadirEnv
    return NadirEnv(num_envs=2)


def test_observation_dimensions_match_the_config(env):
    import jax
    from Nadir.training.config import EnvConfig

    cfg = EnvConfig()
    _, obs, priv = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    assert obs.shape == (2, cfg.obs_dim)
    assert priv.shape == (2, cfg.privileged_obs_dim)


def test_actor_observation_excludes_base_linear_velocity(env):
    """Base linear velocity is not observable on the real robot.

    If it leaks into the actor observation the policy learns to depend on
    something the Pi cannot supply, and sim2real fails in a way that is very
    hard to diagnose. It belongs to the critic only.
    """
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace

    cmd, prev, phase = jnp.zeros(3), jnp.zeros(10), jnp.zeros(1)
    qpos = jnp.array(np.concatenate([[0, 0, 0.26], [1, 0, 0, 0], env.DEFAULT_POSE]))

    base = env._get_obs(SimpleNamespace(qpos=qpos, qvel=jnp.zeros(16)), cmd, prev, phase)
    moving = env._get_obs(
        SimpleNamespace(qpos=qpos, qvel=jnp.zeros(16).at[0:3].set(jnp.array([1.0, 0.5, 0.2]))),
        cmd, prev, phase,
    )
    assert np.allclose(np.asarray(base), np.asarray(moving)), (
        "actor observation changed with base linear velocity"
    )


def test_upright_state_is_not_terminal(env):
    """The shipped env used `done = projected_gravity[2] < 0`, and upright is
    -1, so every environment terminated on step 1, forever."""
    import jax

    state, _, _ = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    assert not bool(np.any(np.asarray(state.done)))
    assert not bool(np.any(np.asarray(state.terminated)))


def test_step_runs_and_environments_auto_reset_on_termination(env):
    """A fallen robot must be replaced by a fresh episode, not simulated on.

    Without auto-reset the rollout reports done=True forever while the physics
    keeps running, and GAE collapses to a one-step bandit.
    """
    import jax
    import jax.numpy as jnp

    state, obs, priv = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))

    # Force a terminal state: put both torsos upside down.
    flipped = state.mjx_data.qpos.at[:, 3:7].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
    state = state.replace(mjx_data=state.mjx_data.replace(qpos=flipped))

    (state, obs, priv, reward, done, terminated,
     ep_return, ep_length) = env.step(state, jnp.zeros((2, 10)))

    assert bool(np.all(np.asarray(terminated))), "inverted torso must terminate"
    assert bool(np.all(np.asarray(done)))
    # Auto-reset: the surviving state is a fresh episode.
    assert bool(np.all(np.asarray(state.step_count) == 0)), (
        f"step_count should be 0 after auto-reset, got {np.asarray(state.step_count)}"
    )
    assert np.all(np.isfinite(np.asarray(obs)))
    assert np.all(np.isfinite(np.asarray(reward)))


def test_episode_statistics_reset_with_the_episode(env):
    """episode_return and step_count are per-episode accumulators.

    Carrying them across the auto-reset turns them into a running total since
    the environment was created, which silently makes the reported "episode
    return" a monotonically increasing number that tracks wall-clock progress
    rather than policy quality — and makes best-checkpoint selection pick the
    latest checkpoint every time.
    """
    import jax
    import jax.numpy as jnp

    state, _, _ = env.reset(jax.random.split(jax.random.PRNGKey(3), 2))

    # One ordinary step: the accumulators advance.
    state, _, _, reward, done, _, ep_ret, ep_len = env.step(state, jnp.zeros((2, 10)))
    assert not bool(np.any(np.asarray(done)))
    assert np.allclose(np.asarray(state.episode_return), np.asarray(reward))
    assert bool(np.all(np.asarray(state.step_count) == 1))

    # Force termination: the *returned* statistics describe the finished
    # episode, while the state that comes back has already reset to zero.
    flipped = state.mjx_data.qpos.at[:, 3:7].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
    state = state.replace(mjx_data=state.mjx_data.replace(qpos=flipped))
    state, _, _, _, done, _, ep_ret, ep_len = env.step(state, jnp.zeros((2, 10)))

    assert bool(np.all(np.asarray(done)))
    assert bool(np.all(np.asarray(ep_len) == 2)), np.asarray(ep_len)
    assert bool(np.all(np.asarray(state.episode_return) == 0.0)), (
        f"episode_return leaked across the reset: {np.asarray(state.episode_return)}"
    )
    assert bool(np.all(np.asarray(state.step_count) == 0))


def test_commands_are_resampled_during_an_episode(env):
    """The shipped env hardcoded [0.5, 0, 0] at reset and never changed it, so
    the policy was never actually velocity-conditioned."""
    assert env.command_resample_steps > 0
    assert env.command_resample_steps < env.episode_length
