import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

@dataclass
class CorridorRefinementConfig:
    """Configuration for Hierarchical 2.5cm Path Corridor Refinement."""
    coarse_resolution_m: float = 0.05   # Macro 5cm grid
    fine_resolution_m: float = 0.025    # Micro 2.5cm grid for foothold selection
    corridor_half_width_m: float = 0.25 # 50cm total corridor width
    lookahead_distance_m: float = 1.5   # Evaluate next 1.5 meters of footsteps
    foot_length_m: float = 0.07         # ~7cm foot length
    foot_width_m: float = 0.04          # ~4cm foot width
    max_micro_step_m: float = 0.03      # 3cm step height is hazardous for micro-contact
    lateral_foot_offset_m: float = 0.075 # ~7.5cm distance between left/right foot centerlines

class CorridorRefiner:
    """Hierarchical Coarse-to-Fine Corridor Refiner for Bipedal Foothold Selection.
    
    Takes the macroscopic planned path (planned on a 5cm grid) and re-samples
    a narrow 50cm-wide corridor into a high-density 2.5cm micro-elevation grid.
    Evaluates micro-terrain stability specifically under each projected footstep patch.
    """
    
    def __init__(self, config: CorridorRefinementConfig = CorridorRefinementConfig()):
        self.config = config

    def extract_corridor_bounds(self, path_points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
        """Compute the bounding box of the planned path segment within the lookahead horizon."""
        if not path_points or len(path_points) < 2:
            return None
            
        pts = np.array(path_points)
        min_x = np.min(pts[:, 0]) - self.config.corridor_half_width_m
        max_x = np.max(pts[:, 0]) + self.config.corridor_half_width_m
        min_y = np.min(pts[:, 1]) - self.config.corridor_half_width_m
        max_y = np.max(pts[:, 1]) + self.config.corridor_half_width_m
        return min_x, max_x, min_y, max_y

    def build_fine_corridor_grid(self, points: np.ndarray, 
                                 bounds: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Build a 2.5cm high-resolution micro-elevation & roughness grid within the corridor bounding box.
        
        Returns:
            elevation_grid: (rows, cols) mean height
            roughness_grid: (rows, cols) standard deviation of height
            min_x, min_y: spatial origin of the sub-grid
        """
        min_x, max_x, min_y, max_y = bounds
        res = self.config.fine_resolution_m
        
        cols = int(np.ceil((max_x - min_x) / res))
        rows = int(np.ceil((max_y - min_y) / res))
        
        elevation_grid = np.zeros((rows, cols), dtype=np.float32)
        roughness_grid = np.zeros((rows, cols), dtype=np.float32)
        
        if len(points) == 0:
            return elevation_grid, roughness_grid, min_x, min_y
            
        # Filter points within corridor bounding box
        x, y, z = points[:, 0], -points[:, 1], points[:, 2] # Camera to local: z=forward, x=lateral, -y=height
        valid = (x >= min_x) & (x <= max_x) & (z >= min_y) & (z <= max_y)
        local_x, local_h, local_y = x[valid], y[valid], z[valid]
        
        if len(local_x) == 0:
            return elevation_grid, roughness_grid, min_x, min_y
            
        c_idx = np.clip(np.floor((local_x - min_x) / res).astype(np.int32), 0, cols - 1)
        r_idx = np.clip(np.floor((local_y - min_y) / res).astype(np.int32), 0, rows - 1)
        
        # Bin points into 2.5cm cells
        cell_dict: Dict[Tuple[int, int], List[float]] = {}
        for r, c, h in zip(r_idx, c_idx, local_h):
            idx = (r, c)
            if idx not in cell_dict:
                cell_dict[idx] = []
            cell_dict[idx].append(h)
            
        for (r, c), h_vals in cell_dict.items():
            arr = np.array(h_vals)
            elevation_grid[r, c] = np.mean(arr)
            roughness_grid[r, c] = np.std(arr)
            
        return elevation_grid, roughness_grid, min_x, min_y

    def score_foothold(self, center_x: float, center_y: float,
                       elevation_grid: np.ndarray, roughness_grid: np.ndarray,
                       min_x: float, min_y: float) -> Tuple[float, Tuple[float, float]]:
        """Score a foot contact patch (~7cm x 4cm) in the 2.5cm grid.
        
        Returns:
            stability_score: 0.0 (hazardous edge/root) to 1.0 (flat, stable)
            micro_offset: (dx, dy) suggested shift to find a safer landing patch
        """
        res = self.config.fine_resolution_m
        rows, cols = elevation_grid.shape
        
        # Foot bounding cells: 7cm x 4cm -> ~3 x 2 cells at 2.5cm
        half_l_cells = int(np.ceil((self.config.foot_length_m / 2.0) / res))
        half_w_cells = int(np.ceil((self.config.foot_width_m / 2.0) / res))
        
        cr = int((center_y - min_y) / res)
        cc = int((center_x - min_x) / res)
        
        best_score = -1.0
        best_nudge = (0.0, 0.0)
        
        # Search candidate micro-shifts: 0, +2.5cm, -2.5cm
        for nudge_r in [-1, 0, 1]:
            for nudge_c in [-1, 0, 1]:
                r_start = max(0, cr + nudge_r - half_l_cells)
                r_end = min(rows, cr + nudge_r + half_l_cells + 1)
                c_start = max(0, cc + nudge_c - half_w_cells)
                c_end = min(cols, cc + nudge_c + half_w_cells + 1)
                
                if r_end <= r_start or c_end <= c_start:
                    continue
                    
                patch_elev = elevation_grid[r_start:r_end, c_start:c_end]
                patch_rough = roughness_grid[r_start:r_end, c_start:c_end]
                
                # Height span across the foot contact patch
                height_variance = np.max(patch_elev) - np.min(patch_elev)
                mean_roughness = np.mean(patch_rough)
                
                # Instability penalty: high roughness or large height differences (e.g. stepping on a root edge)
                instability = (height_variance / self.config.max_micro_step_m) + (mean_roughness / 0.02)
                score = float(np.clip(1.0 - instability, 0.0, 1.0))
                
                # Prefer zero nudge if score is already good
                if nudge_r == 0 and nudge_c == 0 and score > 0.8:
                    return score, (0.0, 0.0)
                    
                if score > best_score:
                    best_score = score
                    best_nudge = (nudge_c * res, nudge_r * res)
                    
        return max(best_score, 0.0), best_nudge

    def refine_trajectory_commands(self, base_cmd: Tuple[float, float, float],
                                  points: np.ndarray,
                                  path_waypoints: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """Evaluate upcoming footholds in the 2.5cm corridor and gently adjust velocity commands."""
        bounds = self.extract_corridor_bounds(path_waypoints)
        if bounds is None or len(points) == 0:
            return base_cmd
            
        elev_grid, rough_grid, min_x, min_y = self.build_fine_corridor_grid(points, bounds)
        
        # Check the very next immediate foothold (~25cm ahead)
        if len(path_waypoints) > 1:
            next_pt = path_waypoints[1]
            score, (nudge_x, nudge_y) = self.score_foothold(next_pt[0], next_pt[1], 
                                                           elev_grid, rough_grid, min_x, min_y)
            
            # If foothold is on an unstable edge (score < 0.6), apply a micro-steering correction
            if score < 0.6:
                vx, vy, yaw_rate = base_cmd
                # Nudge lateral velocity and steering to land in the flatter micro-pocket
                adjusted_vy = vy + float(np.clip(nudge_x * 1.5, -0.15, 0.15))
                adjusted_yaw = yaw_rate + float(np.clip(nudge_x * 2.0, -0.3, 0.3))
                return vx, adjusted_vy, adjusted_yaw
                
        return base_cmd
