"""
Multi-core Vectorized PPO Training Script for Nadir Bipedal Walker
"""

import argparse
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sim.env import NadirBipedalWalkerEnv


def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = NadirBipedalWalkerEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Nadir Bipedal Walker with PPO")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments (default: cpu count)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log-dir", type=str, default="tensorboard_logs", help="TensorBoard log directory")
    args = parser.parse_args()

    # Determine CPU threads / parallel workers
    n_envs = args.num_envs if args.num_envs is not None else (os.cpu_count() or 4)
    print(f"[Nadir Trainer] Initializing {n_envs} parallel simulation workers...")

    # Check for tensorboard availability
    has_tensorboard = False
    try:
        import tensorboard  # noqa: F401
        has_tensorboard = True
    except ImportError:
        print("[Nadir Trainer] Tensorboard is not installed. TensorBoard logging disabled.")

    # Check for progress_bar dependencies (tqdm, rich)
    has_progress_bar = False
    try:
        import rich, tqdm  # noqa: F401
        has_progress_bar = True
    except ImportError:
        print("[Nadir Trainer] tqdm/rich not installed. Progress bar disabled.")

    tensorboard_log_path = args.log_dir if has_tensorboard else None

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    if has_tensorboard and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    # Initialize vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecMonitor(env)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=args.save_dir,
        name_prefix="nadir_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Policy architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # Instantiate PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log_path,
        verbose=1,
    )

    print(f"[Nadir Trainer] Starting PPO learning for {args.timesteps:,} steps across {n_envs} environments...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback],
            tb_log_name="PPO_Nadir",
            progress_bar=has_progress_bar,
        )
    except KeyboardInterrupt:
        print("\n[Nadir Trainer] Training interrupted by user. Saving current model...")
    finally:
        final_path = os.path.join(args.save_dir, "nadir_ppo_final")
        model.save(final_path)
        print(f"[Nadir Trainer] Final model saved successfully to: {final_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
