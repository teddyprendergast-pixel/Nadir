# Digital Twin Simulation

Fully vectorized MuJoCo/MJX (JAX) simulation running 4,096 parallel robots on GPU. This is where the locomotion policy learns to walk before ever touching the physical robot.

## Files

| File | Description |
|:---|:---|
| `env_mjx.py` | Vectorized JAX environment — builds the 41-dim observation (projected gravity, angular velocity, joint positions/velocities, previous action, velocity command, gait phase) and 98-dim privileged observation for the critic (adds true base velocity, 50-point terrain scan, friction, external forces). Loads the robot model from `../hardware/nadir.xml`. |
| `rewards.py` | Shaped reward formulation — velocity tracking, upright posture, target height (32 cm), foot air time/clearance. Penalties: action rate, action jerk (2nd derivative), torso angular/linear acceleration, joint torque, foot impact, slip, and self-collision. |
| `terrains.py` | Procedural heightfield generator — flat, rough noise, sloped ramps, staircases, stepping stones, and forest root/log obstacles with difficulty curriculum. |
| `reference_motion.py` | Sinusoidal kinematic gait reference trajectory generator for biasing the policy toward natural walking. |
| `visualize_sim.py` | Interactive 3D MuJoCo viewer for inspecting the robot model or watching a trained ONNX policy walk. |
| `__init__.py` | Re-exports `NadirEnv`, `EnvState`, `RewardConfig`, `TerrainCurriculum`, `DomainRandParams`, `randomize_domain`. |
