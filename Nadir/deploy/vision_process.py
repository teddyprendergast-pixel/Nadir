import multiprocessing
import multiprocessing.connection
import time
import struct
import numpy as np
from typing import Optional, Tuple

class VisionProcess:
    """Runs stereo vision, traversability mapping, and pathfinding in a separate OS process.
    Communicates velocity commands to the control loop via a pipe.
    
    Runs at 5-15 Hz. CANNOT block or preempt the 50 Hz balance loop.
    """
    
    def __init__(self, command_pipe, goal: Optional[Tuple[float, float]] = None):
        self.command_pipe = command_pipe
        self.goal = goal
        self._running = False
    
    def run(self):
        """Main vision loop."""
        try:
            # We delay imports so they don't impact the main control process
            from Nadir.vision.depth_provider import OakDLiteProvider
            from Nadir.navigation.navigator import Navigator
        except ImportError:
            print("Vision modules not found. Exiting vision process.")
            return
            
        depth = OakDLiteProvider(fps=15)
        nav = Navigator(depth)
        if self.goal:
            nav.set_goal(*self.goal)
        
        self._running = True
        depth.start()
        
        try:
            while self._running:
                vx, vy, yaw_rate = nav.step()
                # Send command through pipe (non-blocking)
                cmd_bytes = struct.pack('fff', float(vx), float(vy), float(yaw_rate))
                try:
                    self.command_pipe.send_bytes(cmd_bytes)
                except BrokenPipeError:
                    print("Command pipe broken, exiting vision process.")
                    break
                except Exception as e:
                    print(f"Error sending command: {e}")
                    
                time.sleep(0.066)  # ~15 Hz
        finally:
            depth.stop()
            self._running = False

def launch_vision_process(goal: Optional[Tuple[float, float]] = None) -> Tuple[multiprocessing.connection.Connection, multiprocessing.Process]:
    """Launch the vision process and return the command pipe and process handle."""
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    vp = VisionProcess(child_conn, goal=goal)
    proc = multiprocessing.Process(target=vp.run, daemon=True)
    proc.start()
    return parent_conn, proc
