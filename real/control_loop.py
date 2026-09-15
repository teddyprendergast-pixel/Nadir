"""
===============================================================================
Nadir Robot — Onboard 50 Hz Real-Time Control Loop
===============================================================================
What this file does:
  This script is the physical robot's "central nervous system".
  It runs directly on the robot's onboard mini-computer (e.g., Raspberry Pi,
  Arduino Uno Q, or ARM Linux SBC).

The 50 Hz "Heartbeat" (for non-coders / beginners):
  Every 20 milliseconds (which is 50 times per second), this loop executes
  a cycle like clockwork:
  1. Sense Balance: Reads the IMU sensor (gyroscope and gravity tilt).
  2. Safety Watchdog: If the robot tilts too far or falls, cut motor power
     immediately so nothing breaks or overheats!
  3. Sense Joints: Asks the electric servo motors where each leg joint is.
  4. Listen to Orders: Checks if the camera / navigation system requested a change
     in walking direction (e.g. turn left, step forward).
  5. AI Decision: Passes all sensory readings through the fast ONNX neural network.
  6. Move Motors: Sends new target angles to all Feetech servo motors over the serial bus.
  7. Rest & Repeat: Sleeps any remaining fraction of the 20ms window to keep the
     timing rock-solid.
===============================================================================
"""

import time
import signal
import sys
import struct
import numpy as np
import argparse
from typing import Optional

# Internal hardware drivers and neural network inference engine
from real.servo_bus import STS3215Bus
from real.imu import IMUReader, BNO085Reader, BNO055Reader
from real.onnx_infer import PolicyInference


class ControlLoop:
    """
    Main 50 Hz control loop for the Nadir bipedal robot.
    
    Architectural rule:
      This loop strictly handles balance and movement.
      It NEVER touches heavy camera image processing or audio detection directly!
      Perception runs as a separate background process and talks to this loop
      through a non-blocking pipe, ensuring the balance loop NEVER freezes or lags.
    """
    
    def __init__(self, policy_path: str, servo_port: str = '/dev/ttyAMA0',
                 servo_ids: list = None, dt: float = 0.02,
                 max_tilt_rad: float = 1.0, command_pipe=None, imu_type: str = 'bno085'):
        # dt = 0.02 seconds = 20 milliseconds = 50 Hz frequency
        self.dt = dt
        # If the robot leans further than 1.0 radian (~57 degrees), trigger safety shutoff
        self.max_tilt_rad = max_tilt_rad
        self.running = False
        self.command_pipe = command_pipe
        # Default walking command: [vx = 0, vy = 0, yaw_rate = 0] (stand stably in place)
        self.current_command = np.zeros(3, dtype=np.float32)
        
        # -------------------------------------------------------------------
        # Hardware Setup:
        # 1. Connect to the Feetech STS3215 servo daisy-chain serial bus
        # -------------------------------------------------------------------
        self.servo_ids = servo_ids or list(range(1, 11))
        self.bus = STS3215Bus(port=servo_port, servo_ids=self.servo_ids)
        
        # 2. Connect to the IMU balance sensor (BNO085 or BNO055)
        if imu_type == 'bno085':
            self.imu = BNO085Reader()
        else:
            self.imu = BNO055Reader()
            
        # 3. Load the fast ONNX AI walking brain
        self.policy = PolicyInference(model_path=policy_path, num_joints=len(self.servo_ids))
        
        # Memory to calculate joint speeds: tracks previous joint angles
        self.last_positions = np.zeros(len(self.servo_ids), dtype=np.float32)
        
    def _check_command_pipe(self) -> np.ndarray:
        """
        Non-blocking check for velocity commands from the navigation/vision process.
        Returns the current command [vx, vy, yaw_rate].
        If no new command arrived, it smoothly maintains the previous command.
        """
        if self.command_pipe is None:
            return self.current_command
            
        try:
            # Poll non-blockingly so we never wait or freeze if the camera is slow
            while self.command_pipe.poll():
                cmd_bytes = self.command_pipe.recv_bytes()
                # 3 floating point numbers (4 bytes each = 12 bytes total)
                if len(cmd_bytes) == 12:
                    vx, vy, yaw_rate = struct.unpack('fff', cmd_bytes)
                    self.current_command = np.array([vx, vy, yaw_rate], dtype=np.float32)
        except Exception:
            pass
            
        return self.current_command
        
    def _handle_signal(self, sig, frame):
        """Cleanly catch Ctrl+C or kill signals to safely shut down motors."""
        print("\nShutdown signal received. Stopping control loop safely...")
        self.running = False

    def run(self):
        """
        Starts and runs the 50 Hz control loop.
        Loops continuously until interrupted by user (Ctrl+C) or safety trigger.
        """
        # Register system signals so Ctrl+C turns off motors gently
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        print("Enabling electric torque on all leg servos...")
        self.bus.enable_all_torque(True)
        self.running = True
        
        print("Starting 50 Hz real-time balance loop (20ms cycle)...")
        try:
            while self.running:
                # Mark loop start timestamp in nanoseconds for high precision
                t_start = time.monotonic_ns()
                
                # -----------------------------------------------------------
                # Step 1: Read IMU Balance Sensor
                # -----------------------------------------------------------
                proj_gravity, ang_vel = self.imu.read()
                
                # -----------------------------------------------------------
                # Step 2: Safety Watchdog (Tilt Protection)
                # If the robot has tipped over, immediately stop so the motors
                # don't grind against the floor or strip gears.
                # -----------------------------------------------------------
                tilt = np.arccos(np.clip(proj_gravity[2], -1.0, 1.0))
                if tilt > self.max_tilt_rad:
                    print(f"SAFETY TRIGGER: Tilt ({tilt:.2f} rad / {np.degrees(tilt):.1f}°) exceeded safe limit ({self.max_tilt_rad:.2f} rad)!")
                    break
                    
                # -----------------------------------------------------------
                # Step 3: Read Physical Leg Joint Angles
                # -----------------------------------------------------------
                positions_dict = self.bus.read_all_positions()
                joint_positions = np.array([
                    positions_dict.get(i, self.policy.default_positions[idx]) 
                    for idx, i in enumerate(self.servo_ids)
                ], dtype=np.float32)
                
                # Calculate joint speeds: (change in angle) / (time interval)
                joint_velocities = (joint_positions - self.last_positions) / self.dt
                self.last_positions = joint_positions.copy()
                
                # -----------------------------------------------------------
                # Step 4: Check for Navigation Commands
                # -----------------------------------------------------------
                command = self._check_command_pipe()
                
                # -----------------------------------------------------------
                # Step 5: Ask the AI Brain for Next Joint Positions
                # -----------------------------------------------------------
                obs = self.policy.build_observation(
                    joint_positions, joint_velocities, 
                    proj_gravity, ang_vel, command
                )
                target_positions = self.policy.predict(obs, self.dt)
                
                # -----------------------------------------------------------
                # Step 6: Write Target Angles to Servos
                # -----------------------------------------------------------
                target_dict = {i: float(target_positions[idx]) for idx, i in enumerate(self.servo_ids)}
                self.bus.sync_write_positions(target_dict)
                
                # -----------------------------------------------------------
                # Step 7: Precise Timing & Sleep
                # -----------------------------------------------------------
                t_end = time.monotonic_ns()
                elapsed = (t_end - t_start) / 1e9  # Convert nanoseconds to seconds
                
                # If calculations took longer than 20ms, warn about jitter
                if elapsed > self.dt:
                    print(f"Timing Warning: Control cycle took {elapsed * 1000:.1f}ms (> {self.dt * 1000:.1f}ms target)")
                    
                # Sleep the remaining time to keep the 50 Hz rhythm steady
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        finally:
            # Whenever the loop exits, always run shutdown to safely disable motors
            self.shutdown()

    def shutdown(self):
        """
        Safely disables torque on all motors and closes the serial bus connection.
        Ensures the robot goes limp gently rather than holding high torque when turned off.
        """
        print("Shutting down... disabling servo torques.")
        try:
            self.bus.enable_all_torque(False)
            self.bus.close()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        print("Hardware shutdown complete.")


def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Nadir Onboard Real-Time Control Loop")
    parser.add_argument('--policy', type=str, required=True,
                        help="Path to the exported ONNX AI brain file (e.g. models/nadir_policy.onnx)")
    parser.add_argument('--port', type=str, default='/dev/ttyAMA0',
                        help="Serial port connected to the Feetech servo bus (e.g. /dev/ttyAMA0 or COM3)")
    parser.add_argument('--imu', type=str, choices=['bno085', 'bno055'], default='bno085',
                        help="Model of IMU orientation sensor installed on the robot")
    args = parser.parse_args()
    
    # Initialize and run the control loop
    loop = ControlLoop(policy_path=args.policy, servo_port=args.port, imu_type=args.imu)
    loop.run()


if __name__ == '__main__':
    main()
