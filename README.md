# Nadir

A desktop-scale bipedal walking robot, ~30–40 cm tall, built on a student
budget of roughly €700.

This is a fork-in-spirit of [Open Duck Mini V2](https://github.com/apirrone/Open_Duck_Mini).
We intend to inherit its training and runtime stack rather than reimplement it.

> **Status: Phase 0.** The repository was reset on 2026-08-21. The previous
> contents were AI-generated scaffolding for a different robot and have been
> removed — see [docs/audit-2026-08-21.md](docs/audit-2026-08-21.md) for what
> was there and why it went. The full scaffold remains in git history at tag
> `scaffold-archive` (commit `31ebc64`); nothing was destroyed.
>
> **There is no working code here yet.** This README describes what is planned.
> Anything not marked as existing does not exist.

---

## Hardware

| | |
| :--- | :--- |
| **Actuators** | 12–14× Feetech STS3215 serial-bus servos (half-duplex TTL, daisy-chained, 12-bit magnetic encoders) |
| **Control mode** | **Position** — the servo closes its own loop internally. We never command torque. |
| **IMU** | BNO085 — provides projected gravity + angular velocity |
| **Structure** | 3D-printed PETG, self-manufactured |
| **Power** | 2S LiPo direct to the servo bus; separate 5 V / 5 A UBEC for logic |

### Compute

The control loop and everything else are deliberately separate concerns.

- **Onboard:** Raspberry Pi (Pi 4 1 GB now, Pi 5 4 GB when vision lands).
  Runs the policy through ONNX Runtime at **50 Hz**, drives the servo bus,
  reads the IMU. Headless Raspberry Pi OS Lite. **Memory budget is tight — ~1 GB.**
- **Off-board:** a Mac mini for development, monitoring and telemetry. It is
  **not** in the control loop and is **not** mounted on the robot.
- **Training:** HPC cluster (INFN-Pisa, SLURM/LSF batch queues).

---

## Simulation and training

**MuJoCo Playground / MJX (JAX).** Not Isaac Lab, not Isaac Gym, not Gazebo —
chosen for portability and no NVIDIA lock-in.

The policy outputs **joint position targets**, not torques.

```
train in MJX  ->  validate sim2sim in vanilla MuJoCo  ->  export ONNX  ->  deploy to Pi
```

Latency modelling and domain randomisation belong in the training config, and
their ranges come from measured hardware — see
[hardware/measured/](hardware/measured/).

---

## Phases

Current position: **Phase 0 → 1.**

| Phase | Goal |
| :--- | :--- |
| **0** | Reproduce upstream's sim training on the cluster, unmodified. No hardware purchased. |
| **1** | Pi + IMU + 2 servos. Deterministic 50 Hz loop. **Actuator system-ID.** |
| **2** | Print structure, one full leg, match MJCF to measured reality. |
| **3** | Full assembly, tethered walking, then unsupported steps. |
| **4** | Vision. |

**Vision is deferred to Phase 4.** When it arrives, perception runs as a
separate 5–15 Hz process communicating over UART/IPC. It must never be able to
block or preempt the 50 Hz balance loop.

---

## Non-negotiables

These are the constraints that the deleted scaffolding violated. They are
recorded here so that no future contributor — human or agent — has to
rediscover them.

1. **The servos are position-controlled.** Any code assuming torque control,
   effort interfaces, or an ideal actuator model is wrong by construction.
2. **Actuator system identification is a first-class component.** Real servo
   lag, backlash (~0.87° on the STS3215), PD gains and command latency get
   measured and fed into the MJCF. This determines sim-to-real success more
   than anything else.
3. **Latency modelling and domain randomisation belong in the training config**,
   with ranges derived from measurement rather than intuition.
4. **The 50 Hz control loop and the perception process are separate concerns.**
5. **No ROS.** It is a large unnecessary dependency for a 50 Hz loop on a 1 GB Pi.

---

## Repository layout

Only what actually exists is listed.

```
Nadir/
├── docs/
│   └── audit-2026-08-21.md     # why the previous scaffold was removed
├── hardware/
│   └── measured/               # measured hardware parameters (schema, currently null)
│       ├── README.md
│       └── actuators.yaml
├── .gitignore
├── LICENSE
└── README.md
```

---

## Next steps

1. Fork [`apirrone/Open_Duck_Playground`](https://github.com/apirrone/Open_Duck_Playground)
   and get its example training running on the INFN cluster **unmodified**,
   with upstream's duck, before changing a line. This is the Phase 0 exit
   criterion and doubles as a check on the SLURM/JAX/GPU setup.
2. Add [`apirrone/Open_Duck_Mini_Runtime`](https://github.com/apirrone/Open_Duck_Mini_Runtime)
   as a pinned submodule — this is where the Feetech bus driver and the
   real-time loop live, and it is the piece least worth reimplementing.
3. Build the Phase 1 system-ID harness and populate
   `hardware/measured/actuators.yaml`.

## Upstream

| Repo | Role | How we consume it |
| :--- | :--- | :--- |
| [`Open_Duck_Playground`](https://github.com/apirrone/Open_Duck_Playground) | MJX/Playground training | Hard fork — we change MJCF, rewards, DR continuously |
| [`Open_Duck_Mini_Runtime`](https://github.com/apirrone/Open_Duck_Mini_Runtime) | On-robot Pi runtime, servo bus | Pinned submodule |
| [`Open_Duck_reference_motion_generator`](https://github.com/apirrone/Open_Duck_reference_motion_generator) | Reference motions | Submodule or pip |
| [`Open_Duck_Mini`](https://github.com/apirrone/Open_Duck_Mini) | Hardware / CAD | Vendor only the STLs and MJCF we print |

## License

MIT — see [LICENSE](LICENSE).
