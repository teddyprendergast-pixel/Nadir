"""
Nadir Bipedal Walker Gymnasium Environment
"""

import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class NadirBipedalWalkerEnv(gym.Env):
    """
    MuJoCo-based simulation environment for the Nadir 6-DoF Bipedal Walker.

    Observation Space:
        - Torso vertical offset (qpos[1])
        - Torso pitch angle (qpos[2])
        - Joint positions for 6 actuators (qpos[3:9])
        - Torso horizontal velocity (qvel[0])
        - Torso vertical velocity (qvel[1])
        - Torso angular velocity (qvel[2])
        - Joint velocities for 6 actuators (qvel[3:9])
        Total dimensions = 17 (excluding root horizontal translation x)

    Action Space:
        - Continuous torques / targets for 6 actuators in [-1.0, 1.0]:
          [thigh_right, leg_right, foot_right, thigh_left, leg_left, foot_left]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        frame_skip: int = 4,
        forward_reward_weight: float = 1.2,
        ctrl_cost_weight: float = 0.05,
        healthy_reward: float = 1.0,
        fall_penalty: float = 10.0,
        min_torso_height: float = 0.60,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.fall_penalty = fall_penalty
        self.min_torso_height = min_torso_height

        # Resolve XML path
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "models", "nadir.xml")
            if not os.path.exists(xml_path):
                # Fallback to local directory
                xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "nadir.xml")

        self.xml_path = os.path.abspath(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # 6 actuators: 3 right leg joints + 3 left leg joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation dimensions: (nq - 1 for rootx) + nv
        obs_dim = (self.model.nq - 1) + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.viewer = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Compute observations
        obs = self._get_obs()

        # Compute forward velocity (x-axis in world frame)
        forward_velocity = self.data.qvel[0]
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))

        # Torso height = baseline (1.25m) + rootz translation (qpos[1])
        torso_z = 1.25 + self.data.qpos[1]

        # Check termination (fall condition or severe pitch tilt)
        pitch_angle = self.data.qpos[2]  # rooty hinge angle
        is_fallen = torso_z < self.min_torso_height or abs(pitch_angle) > 1.2

        if is_fallen:
            terminated = True
            reward = -self.fall_penalty
        else:
            terminated = False
            reward = (
                self.forward_reward_weight * forward_velocity
                - ctrl_cost
                + self.healthy_reward
            )

        info = {
            "forward_velocity": float(forward_velocity),
            "torso_z": float(torso_z),
            "ctrl_cost": float(ctrl_cost),
            "pitch_angle": float(pitch_angle),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Add slight stochastic noise to joint positions and velocities
        pos_noise = self.np_random.uniform(low=-0.015, high=0.015, size=self.model.nq)
        vel_noise = self.np_random.uniform(low=-0.015, high=0.015, size=self.model.nv)

        self.data.qpos[:] = pos_noise
        self.data.qvel[:] = vel_noise

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        # Exclude qpos[0] (rootx absolute horizontal translation) for translation invariance
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ]).astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # Offscreen rendering if needed
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data, camera="track")
            return renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Alias for backwards compatibility
BipedalWalkerEnv = NadirBipedalWalkerEnv
