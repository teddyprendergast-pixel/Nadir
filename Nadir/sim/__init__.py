from sim.env import NadirBipedalWalkerEnv
import gymnasium as gym

gym.register(
    id="NadirBipedalWalker-v0",
    entry_point="sim.env:NadirBipedalWalkerEnv",
    max_episode_steps=1000,
)

__all__ = ["NadirBipedalWalkerEnv"]
