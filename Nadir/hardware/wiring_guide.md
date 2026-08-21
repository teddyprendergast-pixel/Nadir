# Nadir Hardware Wiring & Pinout Guide

This guide details the electrical connections, signal routing, and power isolation architecture for the physical **Nadir Bipedal Walker**.

---

## ⚡ Power Architecture & Isolation

> [!WARNING]
> **Never power digital high-torque servos directly from the ESP32 3.3V or 5V VIN pin!** Servo current spikes (up to 8A peak during aggressive walking maneuvers) will trigger brownout resets on the ESP32 or permanently damage the MCU.

```mermaid
graph TD
    LiPo[2S 7.4V LiPo Battery] -->|High Current Rail| UBEC[6V 10A UBEC Step-Down]
    LiPo -->|Power In| Buck[5V Step-Down / ESP32 VIN]
    UBEC -->|Servo V+ (6V Rail)| PCA[PCA9685 V+ Power Terminal]
    Buck -->|5V / 3.3V Logic| ESP32[ESP32-S3 Microcontroller]
    ESP32 -->|3.3V VCC & GND| MPU[MPU6050 IMU]
    ESP32 -->|3.3V VCC & GND| PCA_Logic[PCA9685 VCC Logic]
    ESP32 -->|I2C: GPIO 21 SDA / GPIO 22 SCL| PCA_Logic
    ESP32 -->|I2C: GPIO 21 SDA / GPIO 22 SCL| MPU
    PCA -->|PWM Channels 0-5| Servos[6x Metal Gear Servos]
    GND1[Battery GND] --- GND2[ESP32 GND] --- GND3[PCA9685 GND] --- GND4[IMU GND]
```

---

## 📌 Pinout Connections

### 1. ESP32-S3 to I2C Bus (PCA9685 + MPU6050)
Both devices share the common I2C bus:

| ESP32-S3 Pin | PCA9685 Pin | MPU6050 Pin | Function |
| :--- | :--- | :--- | :--- |
| **GPIO 21** | SDA | SDA | I2C Data Line (400 kHz) |
| **GPIO 22** | SCL | SCL | I2C Clock Line (400 kHz) |
| **3V3** | VCC | VCC | Logic Power (3.3V) |
| **GND** | GND | GND | Common Logic Ground |

### 2. PCA9685 Servo Channel Mapping

| Channel | Joint | Side | Servo Model | Normal Range |
| :---: | :--- | :--- | :--- | :---: |
| **0** | Hip Pitch | Right | RDS3225 (25kg·cm) | $45^\circ - 135^\circ$ |
| **1** | Knee Pitch | Right | RDS3225 (25kg·cm) | $45^\circ - 135^\circ$ |
| **2** | Ankle Pitch | Right | MG996R (15kg·cm) | $65^\circ - 115^\circ$ |
| **3** | Hip Pitch | Left | RDS3225 (25kg·cm) | $45^\circ - 135^\circ$ |
| **4** | Knee Pitch | Left | RDS3225 (25kg·cm) | $45^\circ - 135^\circ$ |
| **5** | Ankle Pitch | Left | MG996R (15kg·cm) | $65^\circ - 115^\circ$ |

### 3. Foot Contact Switches

| Sensor | ESP32 Pin | Configuration |
| :--- | :--- | :--- |
| **Right Foot Contact** | **GPIO 18** | Input with Internal Pull-up (Active LOW on ground contact) |
| **Left Foot Contact** | **GPIO 19** | Input with Internal Pull-up (Active LOW on ground contact) |

---

## 🛠️ Step-by-Step Wiring Procedure

1. **Common Ground**: Connect the negative ground lead from the LiPo battery, UBEC output, ESP32, PCA9685, and IMU together to create a unified reference ground.
2. **Logic Power**: Connect the 3.3V pin of the ESP32 to the `VCC` header on both the PCA9685 and MPU6050 boards.
3. **Servo High-Current Power**: Connect the 6V / 10A output from the UBEC directly to the high-current screw terminal (`V+` and `GND`) on the PCA9685 board.
4. **Servo Signal Leads**: Connect the 3-wire servo connectors to channels 0 through 5, matching `GND (Brown/Black)`, `V+ (Red)`, and `PWM (Orange/Yellow)`.
