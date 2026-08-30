# Nadir

A desktop-scale bipedal walking robot, ~30–40 cm tall, built on a student
budget of roughly €700.

This is a fork-in-spirit of [Open Duck Mini V2](https://github.com/apirrone/Open_Duck_Mini).
We intend to inherit its training and runtime stack rather than reimplement it.

> **Status: Phase 0.** The repository was reset on 2026-08-21 after the
> previous contents turned out to be AI-generated scaffolding for a different
> robot — see [docs/audit-2026-08-21.md](docs/audit-2026-08-21.md). The
> scaffold remains in history at tag `scaffold-archive` (commit `31ebc64`).
>
> An MJX training stack was then merged in `f751b86`. It had never been run:
> it could not import on a case-sensitive filesystem, and underneath that its
> upright reward was sign-inverted, every episode terminated on step 1, and
> nothing ever reset a terminated environment. See
> [docs/audit-2026-08-30.md](docs/audit-2026-08-30.md).
>
> **Branch `rob` repairs it and trains on CINECA Leonardo.** What exists is
> listed below; anything not listed does not exist. Domain randomisation and
> latency modelling remain **off**, blocked on Phase 1 measurements — see
> §4 of that audit.

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
├── Nadir/
│   ├── sim/
│   │   ├── nadir.xml           # 10-DOF MJCF, <position> actuators
│   │   ├── env_mjx.py          # MJX training env (auto-resetting)
│   │   ├── rewards.py          # reward terms + weights (single source of truth)
│   │   ├── sim2sim.py          # run an ONNX policy in vanilla MuJoCo
│   │   ├── domain_rand.py      # NOT WIRED IN — blocked on Phase 1 measurement
│   │   └── terrains.py         # NOT WIRED IN
│   ├── training/               # PPO (config, networks, ppo, train)
│   ├── export/export_onnx.py   # Flax actor -> ONNX, no torch/TF
│   ├── deploy/                 # Pi-side loop — NOT RUN, see audit §3
│   ├── vision/                 # Phase 4 — out of scope, see audit §3
│   └── navigation/             # not in the phase plan at all, see audit §3
├── docs/
│   ├── audit-2026-08-21.md     # why the original scaffold was removed
│   └── audit-2026-08-30.md     # why the f751b86 training stack could not learn
├── hardware/
│   └── measured/               # measured hardware parameters (schema, currently null)
│       ├── README.md
│       └── actuators.yaml
├── scripts/
│   ├── train_leonardo.sbatch   # SLURM for CINECA Leonardo (boost, A100)
│   └── visualize_sim.py        # interactive viewer (needs a display)
├── tests/                      # 21 tests; regression guards for the audit findings
├── upstream/
│   └── runtime/                # submodule: Open_Duck_Mini_Runtime @ v2 (pinned)
├── .gitignore
├── .gitmodules
├── CLAUDE.md                   # project ground truth — read before adding structure
├── LICENSE
└── README.md
```

Clone with submodules:

```bash
git clone --recurse-submodules <url>
# or, in an existing clone:
git submodule update --init --recursive
```

---

## Running the training

Tested on **CINECA Leonardo** (`boost_usr_prod`, A100-SXM-64GB, account
`INF26_npqcd`). The environment is a plain venv on scratch — no conda:

```bash
module load python/3.11.7
python3 -m venv <path>/nadir-venv && source <path>/nadir-venv/bin/activate
pip install "jax[cuda12]" mujoco mujoco-mjx flax optax orbax-checkpoint \
            numpy pyyaml onnx onnxruntime imageio imageio-ffmpeg pytest
```

**Do not `module load cuda`.** The `jax[cuda12]` wheels bundle their own CUDA
runtime and only need `libcuda.so` from the node driver. Loading Leonardo's
CUDA module shadows them and JAX fails with `Unable to load cuSPARSE`, then
falls back to CPU.

```bash
sbatch scripts/train_leonardo.sbatch                    # full run
python -m pytest tests/ -q                              # 21 tests, CPU, ~50 s
python -m Nadir.export.export_onnx --checkpoint runs/<run>/best \
       --output exported_models/nadir_policy.onnx
MUJOCO_GL=egl python -m Nadir.sim.sim2sim \
       --onnx exported_models/nadir_policy.onnx --video docs/walk.mp4
```

Paths in the sbatch script are absolute and Leonardo-specific; INFN-Pisa needs
its own partition, account and venv path.

---

## Open decisions

Resolve these before Phase 1 hardware is ordered.

- **IMU: BNO085 or BNO055?** Our spec says BNO085; upstream's runtime is
  **BNO055-only** (`adafruit_bno055`, with a working calibration script). They
  are different chips with different drivers — not drop-in. Adopting BNO055
  inherits the IMU stack for free; keeping BNO085 means budgeting a port.
- **Upstream's `openai==1.70.0` dependency** is unnecessary for us and unwanted
  on a 1 GB Pi. Strip it when vendoring or configuring.

## Next steps

1. ~~Add `Open_Duck_Mini_Runtime` as a pinned submodule~~ — done, see
   `upstream/runtime`.
2. Fork [`apirrone/Open_Duck_Playground`](https://github.com/apirrone/Open_Duck_Playground)
   and get its example training running on the cluster **unmodified**, with
   upstream's duck, before changing a line. This is the Phase 0 exit
   criterion. **Still open.** Branch `rob` repaired and ran the *custom* stack
   instead, which validated the SLURM/JAX/GPU setup but is not the same thing
   — it has no reference that is known to walk.
3. Read `upstream/runtime/mini_bdx_runtime/` — particularly
   `rustypot_position_hwi.py` and `onnx_infer.py` — before writing anything
   that talks to a servo.
4. Build the Phase 1 system-ID harness and populate
   `hardware/measured/actuators.yaml`. Upstream's `scripts/record_data.py` is
   a useful starting point.

## Upstream

| Repo | Role | How we consume it |
| :--- | :--- | :--- |
| [`Open_Duck_Playground`](https://github.com/apirrone/Open_Duck_Playground) | MJX/Playground training | Hard fork — we change MJCF, rewards, DR continuously |
| [`Open_Duck_Mini_Runtime`](https://github.com/apirrone/Open_Duck_Mini_Runtime) | On-robot Pi runtime, servo bus | Pinned submodule |
| [`Open_Duck_reference_motion_generator`](https://github.com/apirrone/Open_Duck_reference_motion_generator) | Reference motions | Submodule or pip |
| [`Open_Duck_Mini`](https://github.com/apirrone/Open_Duck_Mini) | Hardware / CAD | Vendor only the STLs and MJCF we print |

## License

MIT — see [LICENSE](LICENSE).
