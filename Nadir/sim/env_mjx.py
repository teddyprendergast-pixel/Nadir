"""MJX training environment for Nadir.

Design notes that matter (see docs/audit-2026-08-30.md for the history):

* The policy emits joint **position targets**, never torques. `ctrl` is a goal
  angle in radians, matching how the STS3215 is actually driven.
* Environments **auto-reset on termination**. Without this the rollout keeps
  simulating a fallen robot while reporting done=True forever, which collapses
  GAE to a one-step bandit.
* Termination and truncation are reported separately. Only *termination* zeroes
  the value bootstrap; hitting the time limit must not teach the critic that
  the world ends at 20 s.
* Velocity commands are body-frame, so the tracking reward compares against a
  body-frame velocity.
"""

import os
from typing import Any, Dict, Tuple

import flax.struct
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from . import rewards as R


@flax.struct.dataclass
class EnvState:
    mjx_data: mjx.Data
    obs: jnp.ndarray
    privileged_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray          # episode boundary (termination OR truncation)
    terminated: jnp.ndarray    # true failure only; zeroes the value bootstrap
    step_count: jnp.ndarray
    command: jnp.ndarray
    previous_action: jnp.ndarray
    action_buffer: jnp.ndarray  # (latency_steps + 1, nu) FIFO of goal positions
    gait_phase: jnp.ndarray
    feet_air_time: jnp.ndarray
    last_contact: jnp.ndarray
    last_foot_pos: jnp.ndarray  # (2, 3), for finite-difference foot velocity
    episode_return: jnp.ndarray
    rng: jnp.ndarray


class NadirEnv:
    # Nominal standing pose, radians, in MJCF joint order.
    DEFAULT_POSE = (0.0, 0.0, 0.5, -0.3, 0.0, 0.0, 0.0, 0.5, -0.3, 0.0)

    # Per-joint action scale: the policy outputs a in [-1, 1] and the goal
    # position is DEFAULT_POSE + a * ACTION_SCALE.
    ACTION_SCALE = (0.4, 0.2, 0.6, 0.4, 0.2, 0.4, 0.2, 0.6, 0.4, 0.2)

    # Measured from the MJCF by forward kinematics, not guessed: with
    # DEFAULT_POSE and the soles on the floor the torso origin sits here.
    # Straight-legged maximum is 0.2700 m.
    STANDING_HEIGHT = 0.2577

    def __init__(self, num_envs: int, config: Dict[str, Any] = None):
        self.num_envs = num_envs
        cfg = config or {}

        xml_path = os.path.join(os.path.dirname(__file__), "nadir.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.nu = self.mj_model.nu
        self.decimation = int(cfg.get("decimation", 10))       # 500 Hz -> 50 Hz
        self.dt = self.mj_model.opt.timestep * self.decimation  # 0.02 s
        self.episode_length = int(cfg.get("episode_length", 1000))
        self.command_resample_steps = int(cfg.get("command_resample_steps", 200))

        # Control latency, in whole 50 Hz steps. PINNED AT 0 until Phase 1
        # measures it. hardware/measured/actuators.yaml carries
        # bus.write_all_latency_ms and actuator_model.fitted.response_lag_ms;
        # both are null today, and non-negotiable #3 says the range comes from
        # measurement, not from a number someone picked. The FIFO below is the
        # machinery, ready to be switched on once those fields are populated.
        self.latency_steps = int(cfg.get("latency_steps", 0))

        self.default_pose = jnp.array(self.DEFAULT_POSE)
        self.action_scale = jnp.array(self.ACTION_SCALE)
        self.reward_config = R.RewardConfig()

        # Joint limits, read from the model rather than restated in Python.
        jnt_range = self.mj_model.jnt_range[1:]  # skip the free joint
        self.joint_lower = jnp.array(jnt_range[:, 0])
        self.joint_upper = jnp.array(jnt_range[:, 1])

        # Actuator ctrlrange, for clipping goal positions to what the servo
        # can actually be commanded to.
        self.ctrl_lower = jnp.array(self.mj_model.actuator_ctrlrange[:, 0])
        self.ctrl_upper = jnp.array(self.mj_model.actuator_ctrlrange[:, 1])

        self.foot_geom_ids = jnp.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("foot_geom_r", "foot_geom_l")
        ])
        self.foot_half_height = 0.005
        self.contact_eps = 0.003  # sole within 3 mm of the floor counts as down

        # Command sampling ranges (body frame).
        self.cmd_vx = (-0.3, 0.6)
        self.cmd_vy = (-0.2, 0.2)
        self.cmd_wz = (-1.0, 1.0)
        self.zero_command_prob = 0.1

        # Seeded once on CPU so reset needs no forward-kinematics call.
        self._default_foot_pos = self._compute_default_foot_pos()

        # A pristine mjx.Data, built once. _reset_state runs inside the step's
        # auto-reset branch, so building it there would re-trace make_data on
        # every environment step and bloat the XLA graph. Closing over a
        # concrete value makes it a graph constant instead.
        self._empty_data = mjx.make_data(self.mjx_model)

    # --- setup helpers ------------------------------------------------------

    def _compute_default_foot_pos(self) -> jnp.ndarray:
        d = mujoco.MjData(self.mj_model)
        d.qpos[0:3] = [0.0, 0.0, self.STANDING_HEIGHT]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[7:] = self.DEFAULT_POSE
        mujoco.mj_forward(self.mj_model, d)
        return jnp.array(d.geom_xpos[[int(i) for i in self.foot_geom_ids]])

    def _sample_command(self, rng: jnp.ndarray) -> jnp.ndarray:
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        cmd = jnp.array([
            jax.random.uniform(k1, minval=self.cmd_vx[0], maxval=self.cmd_vx[1]),
            jax.random.uniform(k2, minval=self.cmd_vy[0], maxval=self.cmd_vy[1]),
            jax.random.uniform(k3, minval=self.cmd_wz[0], maxval=self.cmd_wz[1]),
        ])
        # Some episodes command a full stop, so standing still stays in-distribution.
        stand = jax.random.uniform(k4) < self.zero_command_prob
        return jnp.where(stand, jnp.zeros(3), cmd)

    # --- core -------------------------------------------------------------

    def _reset_state(self, rng: jnp.ndarray) -> EnvState:
        """Build a fresh EnvState. Cheap by construction: no mjx.step, no
        mjx.forward, because the observation depends only on qpos/qvel."""
        rng, k_pose, k_vel, k_cmd = jax.random.split(rng, 4)

        qpos = jnp.zeros(self.mjx_model.nq)
        qpos = qpos.at[0:3].set(jnp.array([0.0, 0.0, self.STANDING_HEIGHT + 0.01]))
        qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))  # valid unit quat
        joint_noise = jax.random.uniform(
            k_pose, shape=(self.nu,), minval=-0.05, maxval=0.05
        )
        qpos = qpos.at[7:].set(self.default_pose + joint_noise)

        qvel = jax.random.uniform(
            k_vel, shape=(self.mjx_model.nv,), minval=-0.05, maxval=0.05
        )

        data = self._empty_data.replace(qpos=qpos, qvel=qvel, ctrl=self.default_pose)

        command = self._sample_command(k_cmd)
        prev_action = jnp.zeros(self.nu)
        action_buffer = jnp.tile(self.default_pose, (self.latency_steps + 1, 1))
        gait_phase = jnp.zeros(1)

        obs = self._get_obs(data, command, prev_action, gait_phase)
        priv = self._get_privileged_obs(data, obs)

        return EnvState(
            mjx_data=data,
            obs=obs,
            privileged_obs=priv,
            reward=jnp.array(0.0),
            done=jnp.array(False),
            terminated=jnp.array(False),
            step_count=jnp.array(0, dtype=jnp.int32),
            command=command,
            previous_action=prev_action,
            action_buffer=action_buffer,
            gait_phase=gait_phase,
            feet_air_time=jnp.zeros(2),
            last_contact=jnp.ones(2, dtype=bool),
            last_foot_pos=self._default_foot_pos,
            episode_return=jnp.array(0.0),
            rng=rng,
        )

    def reset(self, rng: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray]:
        state = jax.vmap(self._reset_state)(rng)
        return state, state.obs, state.privileged_obs

    def _step_single(self, s: EnvState, action: jnp.ndarray):
        rng, k_cmd, k_reset = jax.random.split(s.rng, 3)

        action = jnp.clip(action, -1.0, 1.0)
        goal = self.default_pose + action * self.action_scale
        # Clip to the actuator's ctrlrange explicitly. MuJoCo would clamp this
        # silently; doing it here means the penalty terms see the goal the
        # servo will really receive.
        goal = jnp.clip(goal, self.ctrl_lower, self.ctrl_upper)

        # Latency FIFO: push the new goal, apply the oldest.
        buf = jnp.roll(s.action_buffer, shift=1, axis=0).at[0].set(goal)
        applied = buf[self.latency_steps]

        def body_fn(_, data):
            return mjx.step(self.mjx_model, data.replace(ctrl=applied))

        data = jax.lax.fori_loop(0, self.decimation, body_fn, s.mjx_data)

        step_count = s.step_count + 1
        gait_phase = (s.gait_phase + self.dt) % 1.0

        # Resample the velocity command periodically. The previous version
        # hardcoded [0.5, 0, 0] at reset and never changed it, so the policy
        # was not velocity-conditioned at all despite taking a command input.
        resample = (step_count % self.command_resample_steps) == 0
        command = jnp.where(resample, self._sample_command(k_cmd), s.command)

        obs = self._get_obs(data, command, action, gait_phase)
        priv = self._get_privileged_obs(data, obs)

        # --- kinematics used by several reward terms ---
        quat = data.qpos[3:7]
        projected_gravity = self._quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        base_lin_vel_w = data.qvel[0:3]
        base_lin_vel_b = self._quat_rotate_inv(quat, base_lin_vel_w)
        base_ang_vel_b = data.qvel[3:6]  # free joint angular vel is body-frame
        joint_pos = data.qpos[7:]
        joint_vel = data.qvel[6:]
        torso_height = data.qpos[2]

        foot_pos = data.geom_xpos[self.foot_geom_ids]
        sole_z = foot_pos[:, 2] - self.foot_half_height
        contact = sole_z < self.contact_eps
        first_contact = jnp.logical_and(contact, jnp.logical_not(s.last_contact))
        air_time = s.feet_air_time + self.dt
        foot_vel_xy = (foot_pos[:, :2] - s.last_foot_pos[:, :2]) / self.dt

        c = self.reward_config
        components = [
            R.velocity_tracking_reward(base_lin_vel_b, command, c.tracking_sigma),
            R.yaw_rate_tracking_reward(base_ang_vel_b, command, c.tracking_sigma),
            R.upright_reward(projected_gravity),
            R.base_height_reward(torso_height, c.target_height),
            R.feet_air_time_reward(air_time, first_contact, c.air_time_target),
            R.alive_reward(),
            R.lin_vel_z_penalty(base_lin_vel_b),
            R.ang_vel_xy_penalty(base_ang_vel_b),
            R.action_rate_penalty(action, s.previous_action),
            R.joint_velocity_penalty(joint_vel),
            R.joint_limit_penalty(joint_pos, self.joint_lower, self.joint_upper),
            R.feet_slip_penalty(foot_vel_xy, contact),
        ]
        weights = [
            c.tracking_lin_vel, c.tracking_yaw_vel, c.upright, c.base_height,
            c.feet_air_time, c.alive, c.lin_vel_z, c.ang_vel_xy, c.action_rate,
            c.joint_vel, c.joint_limit, c.feet_slip,
        ]
        reward = R.total_reward(components, weights)

        # Air time resets on touchdown, and is zero while the foot is loaded.
        feet_air_time = jnp.where(contact, 0.0, air_time)

        # Termination: torso past horizontal, or collapsed. Truncation: time
        # limit. Only the former is a real failure.
        fell = projected_gravity[2] > 0.0
        collapsed = torso_height < 0.15
        terminated = jnp.logical_or(fell, collapsed)
        truncated = step_count >= self.episode_length
        done = jnp.logical_or(terminated, truncated)

        episode_return = s.episode_return + reward

        stepped = EnvState(
            mjx_data=data,
            obs=obs,
            privileged_obs=priv,
            reward=reward,
            done=done,
            terminated=terminated,
            step_count=step_count,
            command=command,
            previous_action=action,
            action_buffer=buf,
            gait_phase=gait_phase,
            feet_air_time=feet_air_time,
            last_contact=contact,
            last_foot_pos=foot_pos,
            episode_return=episode_return,
            rng=rng,
        )

        # Auto-reset. Both branches are evaluated under vmap, so this has to
        # stay cheap. Only qpos/qvel/ctrl are swapped inside mjx.Data: every
        # other field (contacts, constraint Jacobians, derived kinematics) is
        # recomputed from those by the next mjx.step, so selecting over the
        # whole Data structure would burn a large `where` over the contact and
        # constraint buffers on every single step for no effect.
        fresh = self._reset_state(k_reset)

        def _sel(a, b):
            return jnp.where(done, a, b)

        reset_data = stepped.mjx_data.replace(
            qpos=_sel(fresh.mjx_data.qpos, stepped.mjx_data.qpos),
            qvel=_sel(fresh.mjx_data.qvel, stepped.mjx_data.qvel),
            ctrl=_sel(fresh.mjx_data.ctrl, stepped.mjx_data.ctrl),
            time=_sel(fresh.mjx_data.time, stepped.mjx_data.time),
        )
        next_state = jax.tree_util.tree_map(_sel, fresh, stepped)
        next_state = next_state.replace(mjx_data=reset_data)
        # reward/done/terminated describe the transition just taken, so they
        # survive the reset. episode_return and step_count must NOT: they are
        # per-episode accumulators, and carrying them across the reset turns
        # them into a running total since the env was created.
        next_state = next_state.replace(
            reward=reward, done=done, terminated=terminated
        )

        # The completed episode's statistics are returned separately, because
        # by construction they are gone from the state once it has reset.
        return (next_state, next_state.obs, next_state.privileged_obs,
                reward, done, terminated, episode_return, step_count)

    def step(self, state: EnvState, action: jnp.ndarray):
        """Vectorised step. RNG lives inside the state, one stream per env.

        Returns (state, obs, privileged_obs, reward, done, terminated,
        episode_return, episode_length). The last two are the *completed*
        episode's totals on a step where `done` is set, and are meaningless
        otherwise — mask them with `done` before averaging.
        """
        return jax.vmap(self._step_single)(state, action)

    # --- observations -------------------------------------------------------

    def _quat_rotate_inv(self, quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
        """Rotate `vec` by the inverse of unit quaternion `quat` (w, x, y, z)."""
        w, u = quat[0], quat[1:4]
        t = 2.0 * jnp.cross(u, vec)
        return vec - w * t + jnp.cross(u, t)

    def _get_obs(self, data, command, prev_action, gait_phase) -> jnp.ndarray:
        """41-dim actor observation.

        Deliberately restricted to what the robot can actually sense: projected
        gravity and angular velocity from the BNO085, joint positions from the
        servo encoders, and its own last command. No base linear velocity —
        that is not observable on the real robot, so it lives in the critic's
        privileged observation only.
        """
        quat = data.qpos[3:7]
        projected_gravity = self._quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        base_ang_vel = data.qvel[3:6]
        joint_positions = data.qpos[7:] - self.default_pose
        joint_velocities = data.qvel[6:] * 0.05
        gait_obs = jnp.array([
            jnp.sin(2 * jnp.pi * gait_phase[0]),
            jnp.cos(2 * jnp.pi * gait_phase[0]),
        ])
        return jnp.concatenate([
            projected_gravity,    # 3
            base_ang_vel,         # 3
            joint_positions,      # 10
            joint_velocities,     # 10
            prev_action,          # 10
            command,              # 3
            gait_obs,             # 2
        ])                        # = 41

    def _get_privileged_obs(self, data, standard_obs) -> jnp.ndarray:
        """45-dim critic observation: the actor's 41 plus what only the sim knows.

        The previous version padded this with 50 constant zeros labelled
        "terrain_heights" over flat ground, plus a constant friction and a
        constant zero external force — 54 of its 57 privileged dimensions
        carried no information at all.
        """
        quat = data.qpos[3:7]
        true_base_lin_vel_b = self._quat_rotate_inv(quat, data.qvel[0:3])  # 3
        torso_height = jnp.array([data.qpos[2]])                            # 1
        # Everything here must be derivable from qpos/qvel alone. A freshly
        # reset state has not been stepped, so its geom_xpos is still zeros —
        # anything read from forward kinematics would be silently wrong on the
        # first observation of every episode.
        return jnp.concatenate([
            standard_obs, true_base_lin_vel_b, torso_height
        ])                                                                  # = 45
