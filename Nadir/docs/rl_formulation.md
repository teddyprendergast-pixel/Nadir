# Reinforcement Learning Formulation: Nadir Locomotion

This document describes the Markov Decision Process (MDP) formulation, reward shaping, and PPO algorithmic setup for **Nadir**.

---

## 1. MDP Definition

Locomotion is formulated as a continuous infinite-horizon discounted Markov Decision Process:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

### State Space ($\mathcal{S} \subset \mathbb{R}^{17}$)
To ensure translation invariance, the absolute horizontal displacement $x$ is omitted:

| Index | Symbol | Description | Units |
| :---: | :--- | :--- | :---: |
| 0 | $z - z_0$ | Torso vertical height offset relative to standing | $\text{m}$ |
| 1 | $\theta_{\text{pitch}}$ | Torso pitch angle | $\text{rad}$ |
| 2..4 | $q_{1..3}$ | Right leg joint positions (Hip, Knee, Ankle) | $\text{rad}$ |
| 5..7 | $q_{4..6}$ | Left leg joint positions (Hip, Knee, Ankle) | $\text{rad}$ |
| 8 | $v_x$ | Torso forward linear velocity | $\text{m/s}$ |
| 9 | $v_z$ | Torso vertical velocity | $\text{m/s}$ |
| 10 | $\omega_y$ | Torso pitch angular velocity | $\text{rad/s}$ |
| 11..13 | $\dot{q}_{1..3}$ | Right leg joint angular velocities | $\text{rad/s}$ |
| 14..16 | $\dot{q}_{4..6}$ | Left leg joint angular velocities | $\text{rad/s}$ |

### Action Space ($\mathcal{A} = [-1.0, 1.0]^6$)
Continuous control targets normalized to $[-1, 1]$ corresponding to motor torques/target angles for:
1. `thigh_right_motor`
2. `leg_right_motor`
3. `foot_right_motor`
4. `thigh_left_motor`
5. `leg_left_motor`
6. `foot_left_motor`

---

## 2. Reward Shaping

The step reward $R_t$ incentivizes steady forward motion while penalizing excessive torque and abrupt falls:

$$R_t = w_{\text{fwd}} v_x - w_{\text{ctrl}} \|\mathbf{a}_t\|_2^2 + r_{\text{alive}} - \mathbb{I}_{\text{fall}} p_{\text{fall}}$$

### Default Weights:
- $w_{\text{fwd}} = 1.2$: Forward velocity encouragement
- $w_{\text{ctrl}} = 0.05$: Control cost / energy minimization
- $r_{\text{alive}} = 1.0$: Baseline reward for staying upright per step
- $p_{\text{fall}} = 10.0$: Penalty applied upon early termination

### Termination Conditions:
The episode terminates if:
1. Torso height $z_t < 0.60\,\text{m}$ (Ground contact by torso).
2. $|\theta_{\text{pitch}}| > 1.2\,\text{rad}$ ($> 68^\circ$ pitch tilt).
3. Max episode step limit reached ($T = 1000$ steps).

---

## 3. PPO Training Objectives

We maximize the clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$
- $\hat{A}_t$ is computed via Generalized Advantage Estimation (GAE) with $\lambda = 0.95$ and $\gamma = 0.99$.
