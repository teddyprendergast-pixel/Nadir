"""
MPU6050 IMU Reader and Complementary Filter for Pitch Estimation
"""

import math
import struct
from machine import I2C

MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B


class MPU6050:
    def __init__(self, i2c: I2C, addr: int = MPU6050_ADDR):
        self.i2c = i2c
        self.addr = addr
        # Wake up MPU6050
        self.i2c.writeto_mem(self.addr, PWR_MGMT_1, bytes([0x00]))
        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.alpha = 0.96  # Complementary filter weight

    def read_raw(self):
        data = self.i2c.readfrom_mem(self.addr, ACCEL_XOUT_H, 14)
        ax, ay, az, temp, gx, gy, gz = struct.unpack(">hhhhhhh", data)
        # Scaled values (±2g, ±250 deg/s)
        accel = (ax / 16384.0, ay / 16384.0, az / 16384.0)
        gyro = (gx / 131.0, gy / 131.0, gz / 131.0)
        return accel, gyro

    def update(self, dt: float = 0.02):
        accel, gyro = self.read_raw()
        ax, ay, az = accel
        gx, gy, gz = gyro

        # Pitch from accelerometer (radians)
        accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        gyro_pitch_rate = math.radians(gy)

        # Complementary filter update
        self.pitch = self.alpha * (self.pitch + gyro_pitch_rate * dt) + (1.0 - self.alpha) * accel_pitch
        self.pitch_rate = gyro_pitch_rate

        return self.pitch, self.pitch_rate
