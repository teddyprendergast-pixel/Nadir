import numpy as np
from dataclasses import dataclass

@dataclass
class TraversabilityConfig:
    # Camera intrinsics (Dual 8MP stereo camera approximate)
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 32.0  # center of 64x64 image
    cy: float = 32.0
    
    # Grid parameters
    grid_size: float = 3.0        # meters, extent of local grid
    grid_resolution: float = 0.05  # meters per cell
    
    # Traversability weights  
    slope_weight: float = 0.4
    step_height_weight: float = 0.4
    roughness_weight: float = 0.2
    
    # Thresholds
    max_slope_rad: float = 0.6     # ~35 degrees = impassable
    max_step_height_m: float = 0.08 # 8cm step = impassable for 35cm robot
    max_roughness_m: float = 0.04   # very rough
    
    # Robot dimensions for clearance
    robot_height_m: float = 0.35
    robot_width_m: float = 0.15

def depth_to_pointcloud(depth: np.ndarray, config: TraversabilityConfig) -> np.ndarray:
    """Convert 64x64 depth image to Nx3 point cloud in camera frame."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - config.cx) * z / config.fx
    y = (v - config.cy) * z / config.fy
    
    # Stack to Nx3 and filter out zero depths
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid = z.reshape(-1) > 0
    return points[valid]

def compute_traversability_grid(points: np.ndarray, robot_pose_2d: np.ndarray, config: TraversabilityConfig) -> np.ndarray:
    """Compute traversability cost grid from point cloud."""
    grid_cells = int(config.grid_size / config.grid_resolution)
    grid = np.full((grid_cells, grid_cells), 0.5, dtype=np.float32)  # unknown = 0.5
    
    if len(points) == 0:
        return grid
        
    half_grid = config.grid_size / 2.0
    
    # Filter points within local grid bounds
    x_valid = (points[:, 0] >= -half_grid) & (points[:, 0] < half_grid)
    z_valid = (points[:, 2] >= 0) & (points[:, 2] < config.grid_size) # depth is Z
    valid = x_valid & z_valid
    local_pts = points[valid]
    
    if len(local_pts) == 0:
        return grid
        
    cols = np.floor((local_pts[:, 0] + half_grid) / config.grid_resolution).astype(np.int32)
    rows = np.floor(local_pts[:, 2] / config.grid_resolution).astype(np.int32)
    
    cols = np.clip(cols, 0, grid_cells - 1)
    rows = np.clip(rows, 0, grid_cells - 1)
    
    heights = -local_pts[:, 1]
    
    cell_heights = {}
    for i in range(len(heights)):
        r, c = rows[i], cols[i]
        if (r, c) not in cell_heights:
            cell_heights[(r, c)] = []
        cell_heights[(r, c)].append(heights[i])
        
    for (r, c), h_list in cell_heights.items():
        h_arr = np.array(h_list)
        roughness = np.std(h_arr)
        step_h = np.max(h_arr) - np.min(h_arr)
        
        rough_cost = np.clip(roughness / config.max_roughness_m, 0.0, 1.0)
        step_cost = np.clip(step_h / config.max_step_height_m, 0.0, 1.0)
        slope_cost = np.clip(roughness * 2.0 / config.max_slope_rad, 0.0, 1.0) # Proxy
        
        cost = (config.slope_weight * slope_cost + 
                config.step_height_weight * step_cost + 
                config.roughness_weight * rough_cost)
        grid[r, c] = np.clip(cost, 0.0, 1.0)
        
    return grid

def compute_upper_clearance(points: np.ndarray, robot_pose_2d: np.ndarray, config: TraversabilityConfig) -> np.ndarray:
    """Compute overhead clearance map for overhanging branches."""
    grid_cells = int(config.grid_size / config.grid_resolution)
    grid = np.full((grid_cells, grid_cells), 0.0, dtype=np.float32)  # clear = 0.0
    
    if len(points) == 0:
        return grid
        
    half_grid = config.grid_size / 2.0
    
    x_valid = (points[:, 0] >= -half_grid) & (points[:, 0] < half_grid)
    z_valid = (points[:, 2] >= 0) & (points[:, 2] < config.grid_size) 
    valid = x_valid & z_valid
    local_pts = points[valid]
    
    if len(local_pts) == 0:
        return grid
        
    cols = np.floor((local_pts[:, 0] + half_grid) / config.grid_resolution).astype(np.int32)
    rows = np.floor(local_pts[:, 2] / config.grid_resolution).astype(np.int32)
    
    cols = np.clip(cols, 0, grid_cells - 1)
    rows = np.clip(rows, 0, grid_cells - 1)
    
    heights = -local_pts[:, 1]
    obstacle_mask = heights > config.robot_height_m
    
    obs_cols = cols[obstacle_mask]
    obs_rows = rows[obstacle_mask]
    grid[obs_rows, obs_cols] = 1.0
    
    return grid
