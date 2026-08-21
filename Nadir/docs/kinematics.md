# Kinematics and Dynamics of Nadir

This document outlines the kinematic structure, coordinate frame transformations, Denavit-Hartenberg (DH) parameters, and equations of motion for the **Nadir Bipedal Walker**.

---

## 1. Kinematic Model

Nadir operates as a planar biped with 6 actuated revolute joints and an unactuated floating base (torso).

```
          [Torso / Base]  (x_t, z_t, theta_t)
                |
       +--------+--------+
       |                 |
  [Right Hip q_1]   [Left Hip q_4]
       |                 |
  [Right Thigh L_1] [Left Thigh L_1]
       |                 |
  [Right Knee q_2]  [Left Knee q_5]
       |                 |
  [Right Shank L_2] [Left Shank L_2]
       |                 |
  [Right Ankle q_3] [Left Ankle q_6]
       |                 |
  [Right Foot L_3]  [Left Foot L_3]
```

### Link Lengths and Masses
- **Thigh Length ($L_1$)**: $0.40\,\text{m}$, Mass: $1.5\,\text{kg}$
- **Shank Length ($L_2$)**: $0.40\,\text{m}$, Mass: $1.2\,\text{kg}$
- **Foot Length ($L_3$)**: $0.20\,\text{m}$, Mass: $0.5\,\text{kg}$
- **Torso Height**: $0.40\,\text{m}$, Mass: $4.0\,\text{kg}$

---

## 2. Forward Kinematics (Single Leg)

Given hip joint coordinate $(x_h, z_h)$ and absolute orientation angles:
- Thigh angle: $\alpha_1 = \theta_{\text{torso}} + q_1$
- Shank angle: $\alpha_2 = \alpha_1 + q_2$
- Foot angle: $\alpha_3 = \alpha_2 + q_3$

The positions of the knee, ankle, and toe are:

$$x_{\text{knee}} = x_h + L_1 \sin(\alpha_1)$$
$$z_{\text{knee}} = z_h - L_1 \cos(\alpha_1)$$

$$x_{\text{ankle}} = x_{\text{knee}} + L_2 \sin(\alpha_2)$$
$$z_{\text{ankle}} = z_{\text{knee}} - L_2 \cos(\alpha_2)$$

$$x_{\text{toe}} = x_{\text{ankle}} + L_3 \cos(\alpha_3)$$
$$z_{\text{toe}} = z_{\text{ankle}} - L_3 \sin(\alpha_3)$$

---

## 3. Equations of Motion

The dynamic equations of the biped are derived using the standard rigid-body manipulator equation:

$$M(\mathbf{q}) \ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + G(\mathbf{q}) = B \mathbf{u} + J_c(\mathbf{q})^T \mathbf{\lambda}$$

Where:
- $\mathbf{q} \in \mathbb{R}^9$: Generalized coordinates $[x, z, \theta, q_1, q_2, q_3, q_4, q_5, q_6]^T$
- $M(\mathbf{q})$: Positive-definite Symmetric Inertia Matrix
- $C(\mathbf{q}, \dot{\mathbf{q}})$: Coriolis and Centrifugal matrix
- $G(\mathbf{q})$: Gravity vector
- $B$: Actuator selection matrix ($9 \times 6$)
- $\mathbf{u} \in \mathbb{R}^6$: Motor joint torques
- $J_c(\mathbf{q})$: Contact Jacobian for ground reaction forces $\mathbf{\lambda}$
