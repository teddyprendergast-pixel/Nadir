import os
from functools import partial
from typing import Tuple, Dict, Any
from flax import struct
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from .rewards import RewardConfig, total_reward, velocity_tracking_reward, upright_reward, action_rate_penalty

@struct.dataclass
class EnvState:
    mjx_data: mjx.Data
    obs: jnp.ndarray
    privileged_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: Dict[str, jnp.ndarray]
    step_count: jnp.ndarray
    command: jnp.ndarray
    previous_action: jnp.ndarray
    gait_phase: jnp.ndarray
    rng: jnp.ndarray

class NadirEnv:
    def __init__(self, num_envs: int, config: Dict[str, Any] = None):
        self.num_envs = num_envs
        self.config = config or {}
        
        xml_path = os.path.join(os.path.dirname(__file__), "nadir.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # Ensure correct timing
        self.mj_model.opt.timestep = 0.002
        self.decimation = 10  # 500 Hz physics -> 50 Hz policy
        
        self.mjx_model = mjx.put_model(self.mj_model)
        
        # Default standing pose (radians)
        self.default_positions = jnp.array([
            0.0, 0.0, 0.5, -0.3, 0.0,
            0.0, 0.0, 0.5, -0.3, 0.0
        ])
        self.action_scale = jnp.array([
            0.4, 0.2, 0.6, 0.4, 0.2,
            0.4, 0.2, 0.6, 0.4, 0.2
        ])
        
        self.reward_config = RewardConfig()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jnp.ndarray) -> EnvState:
        def _reset_single(rng):
            rng, rng_noise = jax.random.split(rng)
            mjx_data = mjx.make_data(self.mjx_model)
            
            # Reset positions and velocities with slight noise
            noise = jax.random.uniform(rng_noise, shape=(self.mjx_model.nq,), minval=-0.01, maxval=0.01)
            qpos = self.mjx_model.qpos0 + noise
            # Set joint positions to default pose
            qpos = qpos.at[7:17].set(self.default_positions + noise[7:17])
            qvel = jnp.zeros(self.mjx_model.nv)
            
            mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
            mjx_data = mjx.step(self.mjx_model, mjx_data)
            
            command = jnp.array([0.5, 0.0, 0.0]) # Example command: vx, vy, yaw_rate
            prev_action = jnp.zeros(10)
            gait_phase = jnp.zeros(1)
            
            obs = self._get_obs(mjx_data, command, prev_action, gait_phase)
            priv_obs = self._get_privileged_obs(mjx_data)
            
            return EnvState(
                mjx_data=mjx_data,
                obs=obs,
                privileged_obs=priv_obs,
                reward=jnp.array(0.0),
                done=jnp.array(False),
                info={},
                step_count=jnp.array(0),
                command=command,
                previous_action=prev_action,
                gait_phase=gait_phase,
                rng=rng
            )
            
        return jax.vmap(_reset_single)(rng)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        def _step_single(state, action):
            action = jnp.clip(action, -1.0, 1.0)
            target_pos = self.default_positions + action * self.action_scale
            
            # Control limit enforcement is handled by the mjcf, but we set the control array directly
            ctrl = target_pos
            
            def body_fn(i, data):
                data = data.replace(ctrl=ctrl)
                return mjx.step(self.mjx_model, data)
                
            new_data = jax.lax.fori_loop(0, self.decimation, body_fn, state.mjx_data)
            
            # Update gait phase (50Hz increments)
            new_gait_phase = (state.gait_phase + 0.02) % 1.0
            
            # Recompute observations
            obs = self._get_obs(new_data, state.command, action, new_gait_phase)
            priv_obs = self._get_privileged_obs(new_data)
            
            # Compute rewards
            base_lin_vel = new_data.qvel[:3]
            base_ang_vel = new_data.qvel[3:6]
            projected_gravity = obs[:3]
            
            components = [
                velocity_tracking_reward(base_lin_vel, state.command),
                upright_reward(projected_gravity),
                action_rate_penalty(action, state.previous_action)
            ]
            weights = [self.reward_config.tracking_lin_vel, self.reward_config.upright, self.reward_config.action_rate]
            
            reward = total_reward(components, weights)
            
            # Determine done condition
            done = jnp.where(projected_gravity[2] < 0.0, True, False) # Torso tipped over
            
            # Randomize command periodically (simplified)
            rng, cmd_rng = jax.random.split(state.rng)
            
            return EnvState(
                mjx_data=new_data,
                obs=obs,
                privileged_obs=priv_obs,
                reward=reward,
                done=done,
                info={},
                step_count=state.step_count + 1,
                command=state.command,
                previous_action=action,
                gait_phase=new_gait_phase,
                rng=rng
            )

        return jax.vmap(_step_single)(state, action)

    def _quat_rotate_inv(self, quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
        """Rotate a vector by the inverse of a quaternion (wxyz format)."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Inverse rotation: q* v q^-1, for unit quaternion q^-1 = conjugate
        t = 2.0 * jnp.cross(jnp.array([x, y, z]), vec)
        return vec - w * t + jnp.cross(jnp.array([x, y, z]), t)

    def _get_obs(self, data: mjx.Data, command: jnp.ndarray, prev_action: jnp.ndarray, gait_phase: jnp.ndarray) -> jnp.ndarray:
        """Build 41-dim actor observation from MJX data."""
        # Project world gravity [0, 0, -9.81] into the body frame
        quat = data.qpos[3:7]  # torso quaternion (w, x, y, z)
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        projected_gravity = self._quat_rotate_inv(quat, gravity_world)
        
        base_ang_vel = data.qvel[3:6]
        joint_positions = data.qpos[7:17] - self.default_positions
        joint_velocities = data.qvel[6:16] * 0.05
        
        gait_obs = jnp.array([jnp.sin(2 * jnp.pi * gait_phase[0]), jnp.cos(2 * jnp.pi * gait_phase[0])])
        
        return jnp.concatenate([
            projected_gravity,
            base_ang_vel,
            joint_positions,
            joint_velocities,
            prev_action,
            command,
            gait_obs
        ])

    def _get_privileged_obs(self, data: mjx.Data) -> jnp.ndarray:
        true_base_vel = data.qvel[:3]
        terrain_heights = jnp.zeros(50) # Placeholder for terrain height sample
        ground_friction = jnp.array([1.0])
        ext_force = jnp.zeros(3)
        
        return jnp.concatenate([
            true_base_vel,
            terrain_heights,
            ground_friction,
            ext_force
        ])
