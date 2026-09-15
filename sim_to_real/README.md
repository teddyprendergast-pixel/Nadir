# Sim to Real

Bridges the gap between the simulated digital twin and the physical robot. Contains two key components:

## Files

| File | Description |
|:---|:---|
| `domain_rand.py` | JAX-vectorized domain randomization engine. Randomizes mass (±15%), center of mass offset, ground friction (0.3–0.9 for wet forest), actuator PD gains (±20%), control latency (0–2 steps), and random push disturbances (up to 2.0 N). Reads measured hardware bounds from `../hardware/measured/actuators.yaml`. |
| `export_onnx.py` | Policy export pipeline: extracts trained weights from JAX/Flax parameter trees, maps them into an equivalent PyTorch `nn.Module`, and exports a standalone `.onnx` graph runnable on ARM Linux SBCs via ONNX Runtime. |

## Design Principles

1. **Grounded randomization** — Randomization ranges are anchored to real bench measurements (backlash, PD gains, bus latency) from `hardware/measured/actuators.yaml`, not arbitrary guesses.
2. **Action smoothness** — Reward penalties in `simulation/rewards.py` (jerk, action rate) produce policies that transfer cleanly without gear stripping.
3. **Lightweight export** — The ONNX model runs inference in <2 ms on ARM, fitting comfortably inside the 20 ms (50 Hz) control budget.
