# Nadir Bipedal Walker: Bill of Materials (BOM)

A complete list of electronic, mechanical, and structural components for building the physical Nadir robot.

---

## 1. Electronics & Compute

| Component | Description / Model | Qty | Est. Unit Price ($) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Microcontroller** | ESP32-S3 DevKit-C (Dual-core 240MHz, 8MB PSRAM) | 1 | $7.00 | Handles I2C, IMU filter, policy inference |
| **Servo Driver** | PCA9685 16-Channel 12-bit PWM I2C Module | 1 | $4.50 | Controls 6 servo channels via single I2C bus |
| **IMU Sensor** | MPU6050 (or BNO055) 6-Axis Gyro + Accelerometer | 1 | $3.50 | Measures pitch, roll, and angular rates |
| **Actuators (Hips & Knees)** | RDS3225 / DS3235 25kg·cm High Torque Metal Gear Servos | 4 | $14.00 | 4 for hips and knees |
| **Actuators (Ankles)** | MG996R / DS3218 15kg·cm Digital Metal Gear Servos | 2 | $8.00 | 2 for ankle pitch joints |
| **Foot Contact Sensors** | SPDT Roller Lever Microswitches (or FSR 402) | 2 | $1.50 | Ground detection at heel/sole |
| **Power Distribution** | UBEC 5V/6V 10A High-Current Step-Down Converter | 1 | $9.00 | Isolates servo power from ESP32 logic |
| **Battery Pack** | 2S (7.4V) 2200mAh 35C LiPo Battery | 1 | $18.00 | Main high-current power source |

---

## 2. Structural & Mechanical Parts

| Item | Specification | Qty | Notes |
| :--- | :--- | :---: | :--- |
| **Chassis & Leg Brackets** | 3D Printed PETG / PLA+ or Aluminum 6061 U-brackets | 1 set | Thighs (400mm sim equiv), Shanks, Torso spine |
| **Servo Horns** | 25T CNC Aluminum Dual-Arm Servo Horns | 6 | High rigidity to prevent backlash |
| **Flange Bearings** | F624ZZ (4x13x5mm) or F684ZZ Flange Bearings | 6 | Smooth joint rotation opposite servo shafts |
| **Fasteners** | M3 & M4 Socket Cap Screws + Locknuts Assortment | 1 kit | Stainless steel, 8mm to 25mm lengths |
| **Foot Grips** | High-friction textured rubber / silicone padding | 2 | Improves traction and reduces impact bounce |

---

## 3. Estimated Total Cost

- **Electronics & Servos**: ~$110 USD
- **Chassis, Bearings & Hardware**: ~$35 USD
- **Total Estimated Build Cost**: **~$145 USD**
