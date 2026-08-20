"""
MicroPython PCA9685 16-Channel PWM Servo Driver for Nadir
"""

import time
from machine import I2C

PCA9685_ADDRESS = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


class PCA9685:
    def __init__(self, i2c: I2C, address: int = PCA9685_ADDRESS):
        self.i2c = i2c
        self.address = address
        self.reset()
        self.set_pwm_freq(50)  # 50 Hz standard analog/digital servo frequency

    def _write(self, reg: int, val: int):
        self.i2c.writeto_mem(self.address, reg, bytes([val]))

    def _read(self, reg: int) -> int:
        return self.i2c.readfrom_mem(self.address, reg, 1)[0]

    def reset(self):
        self._write(MODE1, 0x00)

    def set_pwm_freq(self, freq_hz: float):
        prescaleval = 25000000.0  # 25MHz internal oscillator
        prescaleval /= 4096.0
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = int(prescaleval + 0.5)

        oldmode = self._read(MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # Sleep mode to set prescaler
        self._write(MODE1, newmode)
        self._write(PRESCALE, prescale)
        self._write(MODE1, oldmode)
        time.sleep_ms(5)
        self._write(MODE1, oldmode | 0xA1)  # Auto-increment enable

    def set_pwm(self, channel: int, on: int, off: int):
        reg = LED0_ON_L + 4 * channel
        data = bytes([on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF])
        self.i2c.writeto_mem(self.address, reg, data)

    def set_servo_angle(self, channel: int, angle_deg: float, min_us: int = 500, max_us: int = 2500):
        """
        Set servo angle in degrees (0 to 180).
        """
        angle_deg = max(0.0, min(180.0, angle_deg))
        pulse_us = min_us + (angle_deg / 180.0) * (max_us - min_us)
        # 50 Hz period = 20,000 us. Resolution = 4096 ticks.
        off_tick = int((pulse_us / 20000.0) * 4096)
        self.set_pwm(channel, 0, off_tick)


class NadirServoController:
    """
    Channel Map:
      Ch 0: Right Hip
      Ch 1: Right Knee
      Ch 2: Right Ankle
      Ch 3: Left Hip
      Ch 4: Left Knee
      Ch 5: Left Ankle
    """
    def __init__(self, i2c: I2C):
        self.driver = PCA9685(i2c)
        self.channels = [0, 1, 2, 3, 4, 5]
        # Neutral center angle offsets (deg)
        self.centers = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
        # Gain multiplier from [-1.0, 1.0] normalized action to degrees
        self.action_scales = [45.0, 45.0, 25.0, 45.0, 45.0, 25.0]

    def apply_actions(self, actions):
        for ch, act, center, scale in zip(self.channels, actions, self.centers, self.action_scales):
            target_angle = center + (act * scale)
            self.driver.set_servo_angle(ch, target_angle)
