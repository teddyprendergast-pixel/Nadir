import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import os

class DepthProvider(ABC):
    @abstractmethod
    def get_depth(self) -> Optional[np.ndarray]:
        """Return a 64x64 float32 depth image in meters, or None if unavailable."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the depth provider."""
        pass
        
    @abstractmethod  
    def stop(self) -> None:
        """Stop the depth provider."""
        pass

class OakDLiteProvider(DepthProvider):
    """Luxonis OAK-D Lite depth provider using DepthAI."""
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 15, output_size: Tuple[int, int] = (64, 64), max_depth_m: float = 5.0):
        self.resolution = resolution
        self.fps = fps
        self.output_size = output_size
        self.max_depth_m = max_depth_m
        self.device = None
        self.depth_queue = None
    
    def start(self) -> None:
        """Create DepthAI pipeline and start the device."""
        import depthai as dai
            
        pipeline = dai.Pipeline()
        # Create stereo depth node
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        xout_depth = pipeline.create(dai.node.XLinkOut)
        
        # Configure mono cameras
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setCamera('left')
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setCamera('right')
        mono_left.setFps(self.fps)
        mono_right.setFps(self.fps)
        
        # Configure stereo depth
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to left camera
        
        # IMPORTANT: Lock auto-focus for vibration resistance
        # OAK-D Lite AF version needs manual focus lock
        color = pipeline.create(dai.node.ColorCamera)
        color.initialControl.setManualFocus(130)  # ~1.5m focus distance
        
        # Link nodes
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        xout_depth.setStreamName('depth')
        stereo.depth.link(xout_depth.input)
        
        # Start device
        self.device = dai.Device(pipeline)
        self.depth_queue = self.device.getOutputQueue('depth', maxSize=2, blocking=False)
    
    def get_depth(self) -> Optional[np.ndarray]:
        """Get latest depth frame, resize to 64x64, normalize to meters."""
        if self.depth_queue is None:
            return None
            
        depth_msg = self.depth_queue.tryGet()
        if depth_msg is None:
            return None
            
        depth_raw = depth_msg.getFrame()  # uint16 in mm
        depth_m = depth_raw.astype(np.float32) / 1000.0
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        
        # Resize to output_size
        import cv2
        depth_resized = cv2.resize(depth_m, self.output_size, interpolation=cv2.INTER_AREA)
        return depth_resized
    
    def stop(self) -> None:
        if self.device is not None:
            self.device.close()
            self.device = None
            self.depth_queue = None

class SimulatedDepthProvider(DepthProvider):
    """For testing: loads depth frames from a directory or generates synthetic data."""
    def __init__(self, directory: Optional[str] = None, output_size: Tuple[int, int] = (64, 64)):
        self.directory = directory
        self.output_size = output_size
        self.frames = []
        self.current_idx = 0
        
    def start(self) -> None:
        if self.directory and os.path.exists(self.directory):
            self.frames = sorted([os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.endswith('.npy')])
        else:
            self.frames = []
            
    def get_depth(self) -> Optional[np.ndarray]:
        if self.frames:
            if self.current_idx >= len(self.frames):
                self.current_idx = 0
            frame = np.load(self.frames[self.current_idx])
            self.current_idx += 1
            return frame
        else:
            # Generate flat plane with noise
            depth = np.ones(self.output_size, dtype=np.float32) * 1.5
            depth += np.random.normal(0, 0.05, self.output_size)
            return depth
            
    def stop(self) -> None:
        pass
