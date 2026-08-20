"""
Interactive Real-Time MuJoCo Visualizer for Nadir Bipedal Walker
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from sim.env import NadirBipedalWalkerEnv


def main():
    parser = argparse.ArgumentParser(description="Visualize Nadir Bipedal Walker Policy in MuJoCo")
    parser.add_argument("--model", type=str, default="checkpoints/pretrained_ppo_walker.zip", help="Path to trained model .zip")
    parser.add_argument("--fps", type=int, default=50, help="Target display FPS")
    args = parser.parse_args()

    # Search paths for model
    candidate_paths = [
        args.model,
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pretrained_ppo_walker.zip"),
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "nadir_ppo_final.zip"),
        os.path.join(os.path.dirname(__file__), "..", "..", "walker", "ppo_bipedal_walker.zip"),
    ]

    model_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_path = p
            break

    env = NadirBipedalWalkerEnv(render_mode="human")

    if model_path is None:
        print("[Nadir Viewer] Warning: No trained model checkpoint found. Running with random policy demo.")
        model = None
    else:
        print(f"[Nadir Viewer] Loading trained model from: {model_path}")
        try:
            model = PPO.load(model_path, env=env)
            print("[Nadir Viewer] Model loaded successfully.")
        except Exception as e:
            print(f"[Nadir Viewer] Failed to load model ({e}). Using random actions.")
            model = None

    frame_delay = 1.0 / max(args.fps, 1)

    print("\n[Nadir Viewer] Simulation running! Press Ctrl+C in terminal to exit.\n")
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
                obs, info = env.reset()

            elapsed = time.time() - t0
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Nadir Viewer] Visualizer closed by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
