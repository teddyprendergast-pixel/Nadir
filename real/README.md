# Only Real — Embedded Deployment

Code that runs exclusively on the physical robot's onboard computer (Arduino Uno Q 4GB / Linux ARM SBC). The deterministic 50 Hz balance loop and all hardware drivers live here.

## Files

| File | Description |
|:---|:---|
| `control_loop.py` | Main 50 Hz real-time control loop. Uses monotonic clock scheduling to guarantee 20 ms cycle time. Reads IMU orientation, queries joint positions over serial, runs ONNX policy inference, and sync-writes target positions. Includes a tilt safety watchdog — if the robot tilts beyond 1 radian (~57°), motor torque is immediately cut. |
| `servo_bus.py` | Low-level half-duplex 1 Mbps TTL serial bus driver for 10–12× Feetech STS3215 servos. Handles packet construction, checksums, sync-write, and position/load/temperature telemetry readback. |
| `imu.py` | Hardware abstraction layer for BNO085 and BNO055 orientation sensors over I2C. Converts quaternion telemetry into projected gravity vectors and angular rates for the policy observation. |
| `onnx_infer.py` | ONNX Runtime inference wrapper. Includes an Exponential Moving Average (EMA) low-pass filter (α=0.7) to eliminate micro-jitter and gear chatter on physical servos. |
| `vision_process.py` | Launches the vision/navigation stack in an isolated OS `multiprocessing.Process`. Polls depth at 5–15 Hz and pipes velocity commands (vx, vy, yaw_rate) non-blockingly to the control loop via `struct`-packed IPC. |

## Architectural Rule

> The 50 Hz control loop **never** imports or calls heavy perception code directly. Vision runs in a separate process and communicates via a non-blocking pipe. This guarantees that a camera stall or deep neural network spike can never freeze the balance loop.
