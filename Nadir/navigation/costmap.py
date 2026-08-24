import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CostmapConfig:
    size_m: float = 5.0          # 5m x 5m local map
    resolution_m: float = 0.05    # 5cm per cell
    decay_rate: float = 0.98      # per-update decay for unseen cells
    unknown_cost: float = 0.5     # cost for unexplored cells

class LocalCostmap:
    """Rolling 2.5D local costmap centered on the robot."""
    
    def __init__(self, config: CostmapConfig = CostmapConfig()):
        self.config = config
        n = int(config.size_m / config.resolution_m)
        self.ground_cost = np.full((n, n), config.unknown_cost, dtype=np.float32)
        self.ceiling_clearance = np.full((n, n), 1.0, dtype=np.float32)  # 1.0 = clear
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.grid_size = n
    
    def update(self, traversability_grid: np.ndarray, clearance_grid: np.ndarray,
               robot_x: float, robot_y: float, robot_yaw: float):
        """Fuse new observation into the rolling costmap."""
        dx = robot_x - self.robot_x
        dy = robot_y - self.robot_y
        
        cells_x = int(dx / self.config.resolution_m)
        cells_y = int(dy / self.config.resolution_m)
        
        # Shift costmap
        new_ground = np.full_like(self.ground_cost, self.config.unknown_cost)
        new_ceiling = np.full_like(self.ceiling_clearance, 1.0)
        
        src_x_start = max(0, cells_x)
        src_x_end = min(self.grid_size, self.grid_size + cells_x)
        src_y_start = max(0, cells_y)
        src_y_end = min(self.grid_size, self.grid_size + cells_y)
        
        dst_x_start = max(0, -cells_x)
        dst_x_end = min(self.grid_size, self.grid_size - cells_x)
        dst_y_start = max(0, -cells_y)
        dst_y_end = min(self.grid_size, self.grid_size - cells_y)
        
        if (src_x_end > src_x_start and src_y_end > src_y_start and
            dst_x_end > dst_x_start and dst_y_end > dst_y_start):
            new_ground[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = self.ground_cost[src_y_start:src_y_end, src_x_start:src_x_end]
            new_ceiling[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = self.ceiling_clearance[src_y_start:src_y_end, src_x_start:src_x_end]
            
        self.ground_cost = new_ground
        self.ceiling_clearance = new_ceiling
        self.robot_x = robot_x
        self.robot_y = robot_y
        
        obs_h, obs_w = traversability_grid.shape
        start_y = self.grid_size // 2 - obs_h // 2
        start_x = self.grid_size // 2 - obs_w // 2
        
        if start_y >= 0 and start_x >= 0 and start_y + obs_h <= self.grid_size and start_x + obs_w <= self.grid_size:
            # Overwrite observed cells
            mask = traversability_grid != 0.5
            self.ground_cost[start_y:start_y+obs_h, start_x:start_x+obs_w][mask] = traversability_grid[mask]
            
            clearance_mask = clearance_grid != 0.0
            self.ceiling_clearance[start_y:start_y+obs_h, start_x:start_x+obs_w][clearance_mask] = clearance_grid[clearance_mask]
            
        # Decay unseen cells towards unknown
        self.ground_cost = self.ground_cost * self.config.decay_rate + self.config.unknown_cost * (1.0 - self.config.decay_rate)
    
    def get_combined_cost(self) -> np.ndarray:
        """Return combined ground + ceiling cost."""
        return np.maximum(self.ground_cost, 1.0 - self.ceiling_clearance)
    
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int((wx - self.robot_x) / self.config.resolution_m) + self.grid_size // 2
        gy = int((wy - self.robot_y) / self.config.resolution_m) + self.grid_size // 2
        return (gx, gy)
        
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = (gx - self.grid_size // 2) * self.config.resolution_m + self.robot_x
        wy = (gy - self.grid_size // 2) * self.config.resolution_m + self.robot_y
        return (wx, wy)
