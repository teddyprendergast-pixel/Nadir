# Measured hardware parameters

This directory exists because **actuator system identification is a first-class
component of this project, not an afterthought.** Measuring what the real
STS3215 servos actually do — lag, backlash, effective PD gains, bus latency —
and feeding those numbers into the MJCF is the single biggest determinant of
whether a policy trained in simulation survives contact with the robot.

The previous version of this repository had nowhere for these numbers to live.
That was the largest gap in it.

## The rule

**A value here is either measured, or it is `null`.**

Datasheet figures live in a separate `datasheet:` block, never mixed with
fitted values. This is deliberate: the failure mode we are guarding against is
reading a plausible number six months from now and assuming it came from the
bench when it was actually a guess someone typed in.

## Files

| File | Contents |
| :--- | :--- |
| `actuators.yaml` | Servo PD gains, backlash, torque limits, bus latency, per-joint mapping |
| `links.yaml` | Measured link masses, lengths, CoM offsets (added in Phase 2) |
| `raw/` | Raw system-ID captures — **gitignored**, these are large |

## Flow

```
bench measurement  ->  raw/*.csv  ->  fit  ->  actuators.yaml  ->  MJCF + DR ranges
     (Phase 1)          (untracked)            (committed)         (training)
```

Note the direction. The MJCF `<position>` actuator gains and the
domain-randomisation ranges are **derived from** these measurements. They are
not chosen first and validated later.

Domain randomisation ranges should be set from `unit_variation` — the spread
actually observed across the servos we bought — rather than from a percentage
picked by intuition.

## What to measure in Phase 1

Against two servos on the bench, before any structure is printed:

1. **Step response** — command a position step, log actual position at full
   rate. Fit `kp`, `kd`, and `response_lag_ms`.
2. **Frequency sweep** — chirp the goal position, find the bandwidth beyond
   which the servo stops tracking. This bounds what the policy can ask for.
3. **Backlash** — approach the same target from both directions, measure the
   dead zone. Nominal is ~0.87° but verify per unit.
4. **Bus timing** — on the Pi, at 50 Hz, log the wall-clock cost of a full
   read+write cycle. Record p99, not just the mean; the tail is what breaks
   the control loop.
5. **Load droop** — how much does holding torque sag as the battery drains
   from full to nominal?

The servos report position, load, current and temperature over the bus. Use
that telemetry rather than adding external sensors.
