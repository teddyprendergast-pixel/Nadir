# Hardware & CAD Modeling

Physical robot specification and bench-tested system identification — the **single source of truth** for Nadir's mechanical properties.

## Files

| File | Description |
|:---|:---|
| `nadir.xml` | MuJoCo MJCF robot model — defines the 35 cm tall, 1.5 kg biped's link geometry, masses, inertias, joint axes, collision capsules, and 10 position-controlled actuators with `forcerange="-3 3"` N·m |
| `measured/actuators.yaml` | Bench-tested Feetech STS3215 servo parameters: fitted PD gains (kp=18.0, kd=1.2), backlash (0.87°), bus latency (3.2 ms read / 2.1 ms write), torque limit (2.5 N·m), and ankle compliance overrides (kp=12.0, kd=1.5) |
| `measured/README.md` | Protocol for step-response testing, frequency sweeps, backlash measurement, and bus latency tail distributions |

## Role in the Pipeline

This folder feeds directly into:
- **`simulation/`** — `env_mjx.py` loads `nadir.xml` to construct the physics twin
- **`sim_to_real/`** — `domain_rand.py` reads `actuators.yaml` to set randomization bounds around measured values
