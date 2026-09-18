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

class Dual8MPCameraProvider(DepthProvider):
    """Dual 8MP stereo camera depth provider using stereo disparity matching."""
    def __init__(
        self,
        left_camera_id: int = 0,
        right_camera_id: int = 1,
        capture_resolution: Tuple[int, int] = (1280, 720),
        fps: int = 15,
        output_size: Tuple[int, int] = (64, 64),
        baseline_m: float = 0.075,
        focal_length_px: float = 320.0,
        max_depth_m: float = 5.0,
    ):
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        self.capture_resolution = capture_resolution
        self.fps = fps
        self.output_size = output_size
        self.baseline_m = baseline_m
        self.focal_length_px = focal_length_px
        self.max_depth_m = max_depth_m
        self.cap_left = None
        self.cap_right = None
        self.stereo = None

    def start(self) -> None:
        """Initialize dual camera captures and stereo disparity matcher."""
        import cv2

        self.cap_left = cv2.VideoCapture(self.left_camera_id)
        self.cap_right = cv2.VideoCapture(self.right_camera_id)

        for cap in (self.cap_left, self.cap_right):
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_resolution[1])
                cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Semi-Global Block Matching (StereoSGBM) for disparity computation
        num_disp = 64
        block_size = 5
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def get_depth(self) -> Optional[np.ndarray]:
        """Capture stereo pair, compute disparity, and project to 64x64 depth in meters."""
        if self.cap_left is None or self.cap_right is None:
            return None
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            return None

        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()
        if not ret_l or not ret_r:
            return None

        import cv2

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # Compute disparity map (fixed-point int16, divided by 16)
        disparity_raw = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # Avoid divide-by-zero: depth = (focal_length * baseline) / disparity
        disparity_valid = np.maximum(disparity_raw, 0.1)
        depth_m = (self.focal_length_px * self.baseline_m) / disparity_valid
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        depth_m[disparity_raw <= 0] = self.max_depth_m

        # Resize to output_size
        depth_resized = cv2.resize(depth_m, self.output_size, interpolation=cv2.INTER_AREA)
        return depth_resized

    def stop(self) -> None:
        """Release camera devices."""
        if self.cap_left is not None:
            self.cap_left.release()
            self.cap_left = None
        if self.cap_right is not None:
            self.cap_right.release()
            self.cap_right = None
        self.stereo = None


# Compatibility alias
OakDLiteProvider = Dual8MPCameraProvider

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
