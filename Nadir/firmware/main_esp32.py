"""
Main MicroPython Control Loop for Nadir Bipedal Walker on ESP32-S3
"""

import time
from machine import Pin, I2C
from servo_driver import NadirServoController
from imu_filter import MPU6050

# Try importing trained weights if available
try:
    from nadir_weights import predict_action
    HAS_WEIGHTS = True
except ImportError:
    HAS_WEIGHTS = False
    print("[Nadir ESP32] nadir_weights.py not found. Running in calibration/standby mode.")


def main():
    print("[Nadir ESP32] Initializing hardware interfaces...")

    # I2C on GPIO 21 (SDA) and GPIO 22 (SCL)
    i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)

    # Scan I2C bus
    devices = i2c.scan()
    print(f"[Nadir ESP32] I2C devices detected: {[hex(d) for d in devices]}")

    servo_ctrl = NadirServoController(i2c)
    imu = MPU6050(i2c)

    # Foot touch contact switches (active low with pull-up)
    foot_right_sw = Pin(18, Pin.IN, Pin.PULL_UP)
    foot_left_sw = Pin(19, Pin.IN, Pin.PULL_UP)

    # Initial standing posture
    neutral_actions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    servo_ctrl.apply_actions(neutral_actions)
    print("[Nadir ESP32] Servos set to neutral standing posture.")
    time.sleep(1.0)

    loop_dt = 0.02  # 50 Hz control loop
    print("[Nadir ESP32] Control loop starting (50 Hz)...")

    # Simple state vector
    joint_angles = [0.0] * 6
    joint_vels = [0.0] * 6

    while True:
        t_start = time.ticks_ms()

        # 1. Read IMU
        pitch, pitch_rate = imu.update(loop_dt)

        # 2. Read Foot Contacts
        right_contact = 1.0 if foot_right_sw.value() == 0 else 0.0
        left_contact = 1.0 if foot_left_sw.value() == 0 else 0.0

        # 3. Formulate Observation Vector (17 dims)
        # [torso_z_offset, pitch, q1..q6, vx, vz, pitch_rate, vq1..vq6]
        obs = [0.0, pitch] + joint_angles + [0.0, 0.0, pitch_rate] + joint_vels

        # 4. Compute Control Action
        if HAS_WEIGHTS:
            actions = predict_action(obs)
        else:
            # Fallback gentle balance stabilization based on pitch
            p_gain = 0.5
            d_gain = 0.05
            correction = -(p_gain * pitch + d_gain * pitch_rate)
            actions = [correction, -correction, 0.0, correction, -correction, 0.0]

        # 5. Safety Watchdog: If tilt is too extreme (> 45 deg), disable actuators
        if abs(pitch) > 0.8:
            print(f"[Nadir ESP32] WARNING: Fall detected! Tilt = {pitch:.2f} rad. Centering servos.")
            servo_ctrl.apply_actions([0.0] * 6)
            time.sleep(0.5)
            continue

        # 6. Apply Servo Actions
        servo_ctrl.apply_actions(actions)

        # 7. Regulate 50 Hz loop frequency
        elapsed = time.ticks_diff(time.ticks_ms(), t_start)
        delay_ms = int(max(0, (loop_dt * 1000) - elapsed))
        time.sleep_ms(delay_ms)


if __name__ == "__main__":
    main()
