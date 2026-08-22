"""
Interactive Real-Time MuJoCo Visualizer for Nadir Bipedal Walker in Training Pipeline
"""

import argparse
import os
import sys
import time

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from sim.env import NadirBipedalWalkerEnv


def visualize(model_path: str = None, fps: int = 50, num_episodes: int = 0):
    """
    Launches a window to visualize the Nadir Bipedal Walker using MuJoCo passive viewer.
    """
    candidate_paths = []
    if model_path:
        candidate_paths.append(model_path)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_paths.extend([
        os.path.join(base_dir, "checkpoints", "pretrained_ppo_walker.zip"),
        os.path.join(base_dir, "checkpoints", "nadir_ppo_final.zip"),
        os.path.join(base_dir, "..", "walker", "ppo_bipedal_walker.zip"),
    ])

    resolved_model_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_model_path = p
            break

    print("[Nadir Visualizer] Opening visualizer window for Nadir Bipedal Walker...")
    env = NadirBipedalWalkerEnv(render_mode="human")

    model = None
    if resolved_model_path:
        print(f"[Nadir Visualizer] Loading policy checkpoint from: {resolved_model_path}")
        try:
            model = PPO.load(resolved_model_path, env=env)
            print("[Nadir Visualizer] Model policy successfully loaded.")
        except Exception as e:
            print(f"[Nadir Visualizer] Warning: Could not load checkpoint ({e}). Using random actions.")
    else:
        print("[Nadir Visualizer] No model checkpoint found. Running demo with random policy.")

    frame_delay = 1.0 / max(fps, 1)
    episode_count = 0

    print("[Nadir Visualizer] Visualizer running. Press Ctrl+C in terminal to exit.")
    obs, info = env.reset()

    try:
        while True:
            t0 = time.time()
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                episode_count += 1
                if num_episodes > 0 and episode_count >= num_episodes:
                    print(f"[Nadir Visualizer] Completed {episode_count} episode(s).")
                    break
                obs, info = env.reset()

            elapsed = time.time() - t0
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Nadir Visualizer] Visualizer closed by user.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Nadir Bipedal Walker in MuJoCo GUI")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .zip checkpoint")
    parser.add_argument("--fps", type=int, default=50, help="Target frame rate (FPS)")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to run (0 for infinite loop)")
    args = parser.parse_args()

    visualize(model_path=args.model, fps=args.fps, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
