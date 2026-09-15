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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    normalize_advantage: bool = True
    
    # Training
    total_timesteps: int = 100_000_000
    seed: int = 42
    log_interval: int = 10       # log every N updates
    save_interval: int = 100     # save every N updates
    checkpoint_dir: str = 'checkpoints'
    
    # Network
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    activation: str = 'elu'      # 'elu' or 'relu'
    init_noise_std: float = 1.0  # initial action log_std

@dataclass
class EnvConfig:
    num_envs: int = 4096
    episode_length: int = 1000   # max steps per episode (20 seconds at 50Hz)
    
    # Robot
    num_joints: int = 10
    num_actions: int = 10
    obs_dim: int = 41            # actor observation dimension
    privileged_obs_dim: int = 98 # critic observation dimension (41 + 57 privileged)
    
    # Control
    decimation: int = 10         # physics steps per policy step (500Hz/50Hz)
    dt: float = 0.02             # policy timestep (seconds)
    
    # Action scaling per joint [hip_p_r, hip_r_r, knee_r, ank_p_r, ank_r_r, hip_p_l, hip_r_l, knee_l, ank_p_l, ank_r_l]
    action_scale: Tuple[float, ...] = (0.4, 0.2, 0.6, 0.4, 0.2, 0.4, 0.2, 0.6, 0.4, 0.2)
    default_joint_positions: Tuple[float, ...] = (0.0, 0.0, 0.5, -0.3, 0.0, 0.0, 0.0, 0.5, -0.3, 0.0)
    
    # Commands
    command_ranges: dict = field(default_factory=lambda: {
        'vx': (-0.5, 1.0),     # m/s
        'vy': (-0.3, 0.3),     # m/s  
        'yaw_rate': (-1.0, 1.0) # rad/s
    })
    command_resample_steps: int = 200  # resample commands every N steps

@dataclass  
class RewardWeights:
    velocity_tracking: float = 1.5
    yaw_rate_tracking: float = 0.8
    upright: float = 0.5
    base_height: float = 0.3
    action_rate: float = -0.01
    joint_torque: float = -0.0002
    joint_acceleration: float = -2.5e-7
    foot_air_time: float = 1.0
    foot_clearance: float = -0.5
    feet_slip: float = -0.04
    collision: float = -1.0
    alive: float = 0.15
