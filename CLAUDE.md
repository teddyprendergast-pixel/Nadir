# Nadir — project context

Ground truth for anyone working in this repo, human or agent. Read this before
proposing structure.

This file exists because the repository was previously scaffolded by an agent
with no project information. It produced a professional-looking structure for a
completely different robot, which silently encoded the wrong control paradigm.
See [docs/audit-2026-08-21.md](docs/audit-2026-08-21.md). Do not let that
happen again.

---

## What this is

A desktop-scale bipedal walking robot, ~30–40 cm tall, ~1.5 kg, on a student
budget of ~€700, pairing a 50 Hz Reinforcement Learning balance loop with
edge spatial AI.

## Non-negotiables

Violating any of these makes the work wrong, not merely suboptimal.

1. **The servos are position-controlled.** Feetech STS3215s close their own
   loop internally and accept goal positions. Any code assuming torque control,
   effort interfaces, or an ideal actuator model is wrong by construction. In
   MuJoCo this means `<position>` actuators, never `<motor>`.
2. **Actuator system identification is a first-class component.** Measured
   servo lag, backlash (~0.87°), effective PD gains and bus latency get fed
   into the MJCF. This determines sim-to-real success more than anything else.
   Measurements live in `hardware/measured/` and are either real or `null`.
3. **Latency modelling and domain randomisation belong in the training config**,
   with ranges derived from measured `unit_variation` — not from a percentage
   someone picked.
4. **The 50 Hz control loop and the perception process are separate concerns.**
   Perception (Phase 4) runs as its own 5–15 Hz process over UART/IPC and must
   never be able to block or preempt the control loop.
5. **No ROS.** A large unnecessary dependency for a 50 Hz loop on a 1 GB Pi.

## Stack

| | Use | Do **not** use |
| :--- | :--- | :--- |
| Sim / RL | MuJoCo Playground, MJX, JAX | Isaac Lab, Isaac Gym, Gazebo, PyBullet |
| Training | PPO on GPU cluster (INFN-Pisa, SLURM/LSF) | Stable-Baselines3, CPU `SubprocVecEnv` |
| Policy output | Joint **position targets** | Torques |
| Onboard | Raspberry Pi, ONNX Runtime @ 50 Hz, headless Pi OS Lite | ESP32, MicroPython, PWM/PCA9685 |
| Middleware | none | ROS / ROS 2 |

Pipeline: **train in MJX → validate sim2sim in vanilla MuJoCo → export ONNX → deploy to Pi.**

## Hardware

- **Actuators:** 12–14× Feetech STS3215, half-duplex TTL serial, daisy-chained,
  12-bit encoders. Report position/load/current/temperature — use that
  telemetry rather than adding sensors.
- **IMU:** BNO085 — projected gravity + angular velocity.
- **Structure:** 3D-printed PETG.
- **Power:** 2S LiPo direct to servo bus; separate 5 V/5 A UBEC for logic.
- **Compute:** Pi 4 1 GB now (Pi 5 4 GB when vision lands). **Memory is tight.**
  A Mac mini is used for development and telemetry — it is *not* in the control
  loop and *not* mounted on the robot.

## Phases

Currently **Phase 0 → 1**.

| Phase | Goal |
| :--- | :--- |
| 0 | Validate baseline sim training on the cluster. No hardware purchased. |
| 1 | Pi + IMU + 2 servos. Deterministic 50 Hz loop. **Actuator system-ID.** |
| 2 | Print structure, one full leg, match MJCF to measured reality. |
| 3 | Full assembly, tethered walking, then unsupported steps. |
| 4 | Vision (Waveshare IMX219-83 stereo). |

Do not build for a later phase during an earlier one. Vision code, in
particular, is out of scope until Phase 4.

## Modules & Architecture

- **`nadir/deploy/`**: Pi-side servo bus driver (`servo_bus.py`), IMU reading, and deterministic 50 Hz real-time control loop (`control_loop.py`).
- **`nadir/sim/` & `nadir/training/`**: MuJoCo MJX parallel simulation, kinematic reference motions, and PPO training stack.
- **`nadir/navigation/`**: Asynchronous 2.5D elevation costmap and local path planning.
- **`nadir/biodiversity/`**: Edge bioacoustic audio analysis and flora vision classification.

## Conventions

- **No speculative structure.** Do not create directories, config files or
  abstractions for things that do not exist yet. One implementation does not
  need an interface.
- **Documentation describes what exists.** If a feature is planned, mark it
  planned. The previous scaffold documented domain randomisation, latency
  modelling and a CI pipeline that were entirely absent — that is worse than
  saying nothing.
- **Measured values are never guessed.** Datasheet figures stay in a separate
  block from fitted ones.
- **Report honestly.** If something is untested, say so.
