import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class IMUReader(ABC):
    @abstractmethod
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read IMU data.
        Returns:
            projected_gravity: (3,) gravity vector in body frame (normalized)
            angular_velocity: (3,) rad/s in body frame
        """
        pass

class BNO085Reader(IMUReader):
    """Adafruit BNO085 via I2C using adafruit_bno08x library."""
    def __init__(self, i2c_bus: int = 1, address: int = 0x4A):
        import board
        import busio
        from adafruit_bno08x.i2c import BNO08X_I2C
        from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, BNO_REPORT_GYROSCOPE
        
        # Ensure i2c object exists based on given bus
        i2c = busio.I2C(board.SCL, board.SDA)
        self.bno = BNO08X_I2C(i2c, address=address)
        self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
    
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        # Get quaternion (i, j, k, real) -> compute gravity projection
        quat = self.bno.quaternion
        if quat is None or any(q is None for q in quat):
            quat = (0.0, 0.0, 0.0, 1.0)
            
        qi, qj, qk, qr = quat
        
        # Projected gravity formula from quaternion
        gx = 2.0 * (qi * qk - qr * qj)
        gy = 2.0 * (qj * qk + qr * qi)
        gz = 1.0 - 2.0 * (qi * qi + qj * qj)
        
        projected_gravity = np.array([gx, gy, gz], dtype=np.float32)
        
        # Get gyroscope -> angular velocity
        gyro = self.bno.gyro
        if gyro is None or any(g is None for g in gyro):
            gyro = (0.0, 0.0, 0.0)
            
        angular_velocity = np.array(gyro, dtype=np.float32)
        
        return projected_gravity, angular_velocity

class BNO055Reader(IMUReader):
    """Adafruit BNO055 via I2C (upstream compatible)."""
    def __init__(self, i2c_bus: int = 1, address: int = 0x28):
        import board
        import busio
        import adafruit_bno055
        
        i2c = busio.I2C(board.SCL, board.SDA)
        self.bno = adafruit_bno055.BNO055_I2C(i2c, address=address)
        
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        # BNO055 quaternion is (w, x, y, z)
        quat = self.bno.quaternion
        if quat is None or any(q is None for q in quat):
            quat = (1.0, 0.0, 0.0, 0.0)
            
        qr, qi, qj, qk = quat
        
        gx = 2.0 * (qi * qk - qr * qj)
        gy = 2.0 * (qj * qk + qr * qi)
        gz = 1.0 - 2.0 * (qi * qi + qj * qj)
        
        projected_gravity = np.array([gx, gy, gz], dtype=np.float32)
        
        gyro = self.bno.gyro
        if gyro is None or any(g is None for g in gyro):
            gyro = (0.0, 0.0, 0.0)
            
        angular_velocity = np.array(gyro, dtype=np.float32)
        
        return projected_gravity, angular_velocity
