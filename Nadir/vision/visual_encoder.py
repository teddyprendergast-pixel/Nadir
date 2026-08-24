import numpy as np
from typing import Tuple, Optional

class SpatialSoftmax:
    """Spatial softmax layer that outputs expected 2D coordinates."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        # x shape: (B, C, H, W) or (C, H, W)
        has_batch = len(x.shape) == 4
        if not has_batch:
            x = np.expand_dims(x, axis=0)
            
        b, c, h, w = x.shape
        x_flat = x.reshape((b, c, -1))
        
        # Softmax
        x_max = np.max(x_flat, axis=-1, keepdims=True)
        exp_x = np.exp(x_flat - x_max)
        softmax_weights = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        softmax_weights = softmax_weights.reshape((b, c, h, w))
        
        # Create coordinate grids
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, w),
            np.linspace(-1.0, 1.0, h)
        )
        
        expected_x = np.sum(softmax_weights * pos_x, axis=(2, 3))
        expected_y = np.sum(softmax_weights * pos_y, axis=(2, 3))
        
        coords = np.stack([expected_x, expected_y], axis=-1)  # (b, c, 2)
        out = coords.reshape((b, c * 2))
        
        if not has_batch:
            out = out[0]
        return out

class VisualEncoder:
    """Lightweight depth image encoder.
    Architecture: 4x Conv2D (stride 2) + SpatialSoftmax → 32-dim latent.
    """
    def __init__(self, weights_path: Optional[str] = None):
        self.spatial_softmax = SpatialSoftmax()
        # Initialize with random weights if no path provided
        # Conv weights: (out_c, in_c, kH, kW)
        self.w1 = np.random.randn(16, 1, 3, 3) * 0.1
        self.b1 = np.zeros(16)
        
        self.w2 = np.random.randn(32, 16, 3, 3) * 0.1
        self.b2 = np.zeros(32)
        
        self.w3 = np.random.randn(64, 32, 3, 3) * 0.1
        self.b3 = np.zeros(64)
        
        self.w4 = np.random.randn(32, 64, 3, 3) * 0.1
        self.b4 = np.zeros(32)
        
        self.fc_w = np.random.randn(32, 64) * 0.1
        self.fc_b = np.zeros(32)
        
        if weights_path:
            self.load_weights(weights_path)
            
    def conv2d_stride2(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        c, h, w_in = x.shape
        out_c = w.shape[0]
        out_h, out_w = h // 2, w_in // 2
        
        # Pad x (padding=1)
        x_pad = np.pad(x, ((0,0), (1,1), (1,1)), mode='constant')
        out = np.zeros((out_c, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                region = x_pad[:, i*2:i*2+3, j*2:j*2+3]
                out[:, i, j] = np.tensordot(w, region, axes=([1, 2, 3], [0, 1, 2])) + b
                
        return np.maximum(0, out) # ReLU

    def forward(self, depth_image: np.ndarray) -> np.ndarray:
        """Process 64x64 depth image to 32-dim latent vector."""
        if len(depth_image.shape) == 2:
            x = np.expand_dims(depth_image, axis=0) # (1, 64, 64)
        else:
            x = depth_image
            
        x = self.conv2d_stride2(x, self.w1, self.b1) # 16x32x32
        x = self.conv2d_stride2(x, self.w2, self.b2) # 32x16x16
        x = self.conv2d_stride2(x, self.w3, self.b3) # 64x8x8
        x = self.conv2d_stride2(x, self.w4, self.b4) # 32x4x4
        
        coords = self.spatial_softmax.forward(x) # 64-dim
        
        out = np.dot(coords, self.fc_w.T) + self.fc_b # 32-dim
        return out
        
    def load_weights(self, path: str):
        if not path.endswith('.npz'):
            print(f"Failed to load weights: {path}")
            return
            
        data = np.load(path)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.w3 = data['w3']
        self.b3 = data['b3']
        self.w4 = data['w4']
        self.b4 = data['b4']
        self.fc_w = data['fc_w']
        self.fc_b = data['fc_b']
