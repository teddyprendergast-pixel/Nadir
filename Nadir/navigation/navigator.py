import numpy as np
from typing import Tuple, Optional
from nadir.vision.depth_provider import DepthProvider
from nadir.vision.traversability import TraversabilityConfig, compute_traversability_grid, depth_to_pointcloud, compute_upper_clearance
from nadir.navigation.costmap import LocalCostmap, CostmapConfig
from nadir.navigation.planner import fast_marching_planner, extract_velocity_command
from nadir.navigation.corridor_refinement import CorridorRefiner

class Navigator:
    """High-level navigation: Goal → velocity commands for the locomotion policy.
    Runs at 5 Hz in a separate process from the 50 Hz balance loop."""
    
    def __init__(self, depth_provider: DepthProvider, config=None):
        self.depth = depth_provider
        self.costmap = LocalCostmap()
        self.trav_config = TraversabilityConfig()
        self.corridor_refiner = CorridorRefiner()
        self.goal = None  # (x, y) in world frame
        self.robot_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
    
    def set_goal(self, x: float, y: float):
        self.goal = (x, y)
    
    def update_pose(self, x: float, y: float, yaw: float):
        self.robot_pose = (x, y, yaw)
    
    def step(self) -> Tuple[float, float, float]:
        """Run one navigation cycle. Returns (vx, vy, yaw_rate) command."""
        # 1. Get depth from camera
        depth_img = self.depth.get_depth()
        if depth_img is None:
            return 0.0, 0.0, 0.0
            
        x, y, yaw = self.robot_pose
        
        # 2. Compute traversability grid
        points = depth_to_pointcloud(depth_img, self.trav_config)
        pose_arr = np.array([x, y, yaw])
        trav_grid = compute_traversability_grid(points, pose_arr, self.trav_config)
        clear_grid = compute_upper_clearance(points, pose_arr, self.trav_config)
        
        # 3. Update costmap
        self.costmap.update(trav_grid, clear_grid, x, y, yaw)
        combined_cost = self.costmap.get_combined_cost()
        
        # 4. Run planner toward goal
        if self.goal is None:
            return 0.0, 0.0, 0.0
            
        gx, gy = self.costmap.world_to_grid(self.goal[0], self.goal[1])
        # Check if goal is outside costmap, project to boundary if so
        rows, cols = combined_cost.shape
        gx = int(np.clip(gx, 0, rows - 1))
        gy = int(np.clip(gy, 0, cols - 1))
        
        arrival_field = fast_marching_planner(combined_cost, (gx, gy), self.costmap.config.resolution_m)
        
        # 5. Extract velocity command from coarse 5cm arrival field
        rx, ry = self.costmap.world_to_grid(x, y)
        rx = int(np.clip(rx, 0, rows - 1))
        ry = int(np.clip(ry, 0, cols - 1))
        
        base_cmd = extract_velocity_command(arrival_field, (rx, ry), yaw, self.costmap.config.resolution_m)
        
        # 6. Micro-Corridor Refinement (2.5cm grid for foothold contact stability)
        # Project immediate 1.5m path waypoints in front of the robot
        c, s = np.cos(yaw), np.sin(yaw)
        lookaheads = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        path_waypoints = [(x + d * c, y + d * s) for d in lookaheads]
        
        vx, vy, yaw_rate = self.corridor_refiner.refine_trajectory_commands(base_cmd, points, path_waypoints)
        return vx, vy, yaw_rate
    
    def get_costmap_image(self) -> np.ndarray:
        """Return a color visualization of the costmap (green→red gradient)."""
        cost = self.costmap.get_combined_cost()
        
        # Color mapping: 0.0=green, 0.5=yellow/gray, 1.0=red
        h, w = cost.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Simple gradient:
        # Cost 0.0 -> (0, 255, 0)
        # Cost 1.0 -> (255, 0, 0)
        img[..., 0] = (cost * 255).astype(np.uint8) # R
        img[..., 1] = ((1.0 - cost) * 255).astype(np.uint8) # G
        img[..., 2] = 0 # B
        
        return img
