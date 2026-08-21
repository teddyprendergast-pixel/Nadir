"""
Evaluation and Metrics Benchmark for Nadir Bipedal Walker
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from sim.env import NadirBipedalWalkerEnv


def evaluate_policy(model_path: str, num_episodes: int = 10, render: bool = False):
    render_mode = "human" if render else None
    env = NadirBipedalWalkerEnv(render_mode=render_mode)

    print(f"[Nadir Eval] Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)

    episode_rewards = []
    episode_lengths = []
    forward_velocities = []
    control_efforts = []

    print(f"[Nadir Eval] Running {num_episodes} evaluation episodes...")
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        velocities = []
        efforts = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            velocities.append(info.get("forward_velocity", 0.0))
            efforts.append(info.get("ctrl_cost", 0.0))
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        avg_vel = np.mean(velocities) if velocities else 0.0
        avg_eff = np.mean(efforts) if efforts else 0.0
        forward_velocities.append(avg_vel)
        control_efforts.append(avg_eff)

        print(f"  Episode {ep:02d}: Reward = {total_reward:8.2f} | Steps = {steps:4d} | Avg Vel = {avg_vel:5.2f} m/s")

    env.close()

    print("\n" + "=" * 50)
    print(">> EVALUATION BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Mean Episode Reward:    {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length:    {np.mean(episode_lengths):8.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"Mean Forward Velocity:  {np.mean(forward_velocities):8.2f} ± {np.std(forward_velocities):.2f} m/s")
    print(f"Mean Control Cost:      {np.mean(control_efforts):8.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Nadir Bipedal Walker Policy")
    parser.add_argument("--model", type=str, default="checkpoints/pretrained_ppo_walker.zip", help="Path to model .zip")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render simulation live")
    args = parser.parse_args()

    # Fallback search if model path not directly found
    if not os.path.exists(args.model):
        alt_paths = [
            "checkpoints/nadir_ppo_final.zip",
            "../walker/ppo_bipedal_walker.zip",
            "checkpoints/pretrained_ppo_walker.zip",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                args.model = p
                break

    evaluate_policy(args.model, args.episodes, args.render)
