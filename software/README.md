# Software

Core algorithms, neural networks, training pipelines, perception, navigation, and ecological monitoring.

## Modules

### `training/` — Reinforcement Learning (PPO)
JAX/Flax PPO pipeline with Generalized Advantage Estimation. Trains an asymmetric actor-critic (256-256-128 hidden layers, ELU) across 4,096 parallel simulated robots. Supports both GPU cluster (`train.py`) and multi-device CPU (`train_cpu.py`) training.

### `vision/` — Depth Perception & Visual Encoding
Interfaces with the dual 8MP stereo camera setup (`depth_provider.py`). Converts depth maps to 3D point clouds and computes traversability cost from slope, step height, roughness, and overhead clearance (`traversability.py`). Includes a compact 4-stage CNN encoder compressing 64×64 depth to a 32-dim latent (`visual_encoder.py`).

### `navigation/` — Hierarchical Path Planning
Two-tier coarse-to-fine navigation: a 5 m × 5 m rolling costmap at 5 cm resolution (`costmap.py`), Fast Marching Method global planner (`planner.py`), and a 50 cm micro-corridor refiner at 2.5 cm resolution for foothold stability (`corridor_refinement.py`). Orchestrated by `navigator.py`.

### `biodiversity/` — Edge AI Ecological Monitoring
Background-thread ONNX classifiers for plant species (`plant_classifier.py`) and bird/wildlife audio (`bioacoustics.py`). All detections are spatially tagged and logged to a SQLite database (`logger.py`).

### `scripts/` — Utilities
Simulation throughput benchmarks (`benchmark_envs.py`, `benchmark_envs_cpu.py`) and SLURM cluster launcher (`train.sh`).

### `tests/` — Automated Tests
Validates MJCF model loading, position-actuator constraints, observation shapes, and reward functions.
