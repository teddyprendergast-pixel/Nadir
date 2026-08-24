import time
import signal
import sys
import struct
import numpy as np
import argparse
from typing import Optional

from Nadir.deploy.servo_bus import STS3215Bus
from Nadir.deploy.imu import IMUReader, BNO085Reader, BNO055Reader
from Nadir.deploy.onnx_infer import PolicyInference

class ControlLoop:
    """Main 50 Hz control loop for the Nadir biped.
    
    This loop:
    1. Reads IMU (projected gravity + angular velocity)
    2. Reads servo positions and velocities
    3. Runs ONNX policy inference
    4. Writes goal positions to servos
    5. Monitors timing jitter
    
    NEVER imports or touches vision/navigation code.
    The vision process communicates via shared memory or pipe.
    """
    
    def __init__(self, policy_path: str, servo_port: str = '/dev/ttyAMA0',
                 servo_ids: list = None, dt: float = 0.02,
                 max_tilt_rad: float = 1.0, command_pipe=None, imu_type: str = 'bno085'):
        self.dt = dt
        self.max_tilt_rad = max_tilt_rad
        self.running = False
        self.command_pipe = command_pipe
        self.current_command = np.zeros(3, dtype=np.float32)
        
        # Initialize hardware and policy
        self.servo_ids = servo_ids or list(range(1, 11))
        self.bus = STS3215Bus(port=servo_port, servo_ids=self.servo_ids)
        
        if imu_type == 'bno085':
            self.imu = BNO085Reader()
        else:
            self.imu = BNO055Reader()
            
        self.policy = PolicyInference(model_path=policy_path, num_joints=len(self.servo_ids))
        
        # State tracking
        self.last_positions = np.zeros(len(self.servo_ids), dtype=np.float32)
        
    def _check_command_pipe(self) -> np.ndarray:
        """Non-blocking check for velocity commands from the vision process.
        Returns current command (defaults to zero/standing if no vision process)."""
        if self.command_pipe is None:
            return self.current_command
            
        try:
            while self.command_pipe.poll():
                cmd_bytes = self.command_pipe.recv_bytes()
                if len(cmd_bytes) == 12:
                    vx, vy, yaw_rate = struct.unpack('fff', cmd_bytes)
                    self.current_command = np.array([vx, vy, yaw_rate], dtype=np.float32)
        except Exception:
            pass
            
        return self.current_command
        
    def _handle_signal(self, sig, frame):
        print("\nShutdown signal received. Stopping control loop.")
        self.running = False

    def run(self):
        """Run the control loop until interrupted."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        print("Enabling torque on all servos...")
        self.bus.enable_all_torque(True)
        self.running = True
        
        print("Starting 50 Hz control loop...")
        try:
            while self.running:
                t_start = time.monotonic_ns()
                
                # 1. Read IMU
                proj_gravity, ang_vel = self.imu.read()
                
                # Safety check: tilt watchdog
                tilt = np.arccos(np.clip(proj_gravity[2], -1.0, 1.0))
                if tilt > self.max_tilt_rad:
                    print(f"Safety trigger: Tilt ({tilt:.2f} rad) exceeded limit ({self.max_tilt_rad:.2f} rad)")
                    break
                    
                # 2. Read servo state
                positions_dict = self.bus.read_all_positions()
                joint_positions = np.array([positions_dict.get(i, self.policy.default_positions[idx]) 
                                            for idx, i in enumerate(self.servo_ids)], dtype=np.float32)
                
                joint_velocities = (joint_positions - self.last_positions) / self.dt
                self.last_positions = joint_positions.copy()
                
                # 3. Get command
                command = self._check_command_pipe()
                
                # 4. Build observation & inference
                obs = self.policy.build_observation(joint_positions, joint_velocities, 
                                                    proj_gravity, ang_vel, command)
                
                target_positions = self.policy.predict(obs, self.dt)
                
                # 5. Write goal positions
                target_dict = {i: float(target_positions[idx]) for idx, i in enumerate(self.servo_ids)}
                self.bus.sync_write_positions(target_dict)
                
                # 6. Monitor timing jitter and sleep
                t_end = time.monotonic_ns()
                elapsed = (t_end - t_start) / 1e9
                
                if elapsed > self.dt:
                    print(f"Timing Warning: Control cycle took {elapsed:.4f}s (> {self.dt}s)")
                    
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        finally:
            self.shutdown()

    def shutdown(self):
        """Safely disable all servos and close hardware."""
        print("Shutting down... disabling torques.")
        self.bus.enable_all_torque(False)
        self.bus.close()
        print("Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Nadir Real-Time Control Loop")
    parser.add_argument('--policy', type=str, required=True, help="Path to ONNX policy file")
    parser.add_argument('--port', type=str, default='/dev/ttyAMA0', help="Serial port for servos")
    parser.add_argument('--imu', type=str, choices=['bno085', 'bno055'], default='bno085', help="IMU type")
    args = parser.parse_args()
    
    loop = ControlLoop(policy_path=args.policy, servo_port=args.port, imu_type=args.imu)
    loop.run()

if __name__ == '__main__':
    main()
