# Sim-to-Real Transfer Techniques for Nadir

Transferring reinforcement learning policies trained in physics simulators (MuJoCo) onto physical bipedal hardware requires bridging the **reality gap**.

---

## 1. Primary Sources of Reality Gap

1. **Actuator Dynamics & Latency**: Real metal-gear servos have non-instantaneous response times, gear backlash, friction, and voltage drop under load.
2. **Ground Contact Physics**: Real flooring exhibits non-uniform friction, compliance, and micro-slippage compared to rigid plane contact models.
3. **Sensor Noise & Drift**: IMUs experience vibration noise, gyroscope drift, and quantization errors.
4. **Mass & Inertia Uncertainty**: Real 3D printed parts and wiring deviate slightly from CAD nominal masses.

---

## 2. Strategies Implemented in Nadir

### A. Domain Randomization (DR)
During training, randomize physical properties across episodes:
- **Link Masses**: $\pm 15\%$ variation per body segment.
- **Ground Friction**: $\mu \in [0.6, 1.4]$.
- **Motor Strength / Gear Ratio**: $\pm 10\%$ scaling.
- **Observation Noise**: Gaussian noise $\mathcal{N}(0, \sigma^2)$ injected into joint angles and pitch rates.

### B. Action Filtering & Slew Rate Limiting
To prevent high-frequency servo jitter and gear wear, actions are low-pass filtered:

$$a_t^{\text{applied}} = \beta a_t + (1 - \beta) a_{t-1}^{\text{applied}}, \quad \beta \in [0.4, 0.7]$$

### C. Latency Modeling
A 1–2 step queue delay ($20 - 40\,\text{ms}$) is added in simulation so the policy learns to anticipate control latency.

### D. Hardware Watchdog & Safety Centering
In `firmware/main_esp32.py`, if the estimated pitch angle exceeds $0.8\,\text{rad}$ ($> 45^\circ$), the control loop disables aggressive torques and returns all servos to neutral standing angles to prevent hardware damage.
