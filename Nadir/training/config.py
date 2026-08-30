"""Training configuration.

Reward weights are NOT duplicated here. They live in
`Nadir.sim.rewards.RewardConfig`, which is the single source of truth — the
previous `RewardWeights` dataclass in this file was never read by anything and
had drifted out of agreement with the values the env actually used.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PPOConfig:
    # PPO hyperparameters
    learning_rate: float = 3e-4
    num_steps: int = 24          # steps per env per rollout
    num_envs: int = 4096         # parallel environments
    num_minibatches: int = 32
    update_epochs: int = 5
    gamma: float = 0.99          # 50 Hz control -> ~2 s effective horizon
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    normalize_advantage: bool = True

    # Training
    total_timesteps: int = 300_000_000
    seed: int = 42
    log_interval: int = 10       # log every N updates
    save_interval: int = 100     # save every N updates
    checkpoint_dir: str = "checkpoints"

    # Network
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0


@dataclass
class EnvConfig:
    num_envs: int = 4096
    episode_length: int = 1000   # 20 s at 50 Hz

    # Robot
    num_joints: int = 10
    num_actions: int = 10
    obs_dim: int = 41            # actor: what the BNO085 + encoders can see
    privileged_obs_dim: int = 45 # critic: + body-frame base velocity, torso height

    # Control
    decimation: int = 10         # 500 Hz physics per 50 Hz policy step
    dt: float = 0.02

    command_resample_steps: int = 200  # 4 s at 50 Hz

    # Modelled actuator latency, in whole 50 Hz control steps.
    # PINNED AT 0. Non-negotiable #3 requires this to come from measurement;
    # hardware/measured/actuators.yaml is status: unmeasured, so there is no
    # honest value to put here yet. The env implements the delay FIFO so this
    # becomes a one-line change once Phase 1 system-ID lands.
    latency_steps: int = 0

    def env_kwargs(self) -> dict:
        return {
            "episode_length": self.episode_length,
            "decimation": self.decimation,
            "command_resample_steps": self.command_resample_steps,
            "latency_steps": self.latency_steps,
        }
