import numpy as np
from typing import Tuple, Optional, List
import heapq

def weighted_astar(cost_grid: np.ndarray, start: Tuple[int, int], 
                   goal: Tuple[int, int], resolution: float) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding on the traversability costmap.
    Cost of moving to a cell = distance * (1.0 + cell_cost * cost_multiplier)
    Cells with cost >= 0.95 are treated as impassable walls."""
    rows, cols = cost_grid.shape
    
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None
        
    if cost_grid[goal[0], goal[1]] >= 0.95:
        return None
        
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) * resolution
        
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    
    cost_multiplier = 10.0
    
    while open_set:
        current_cost, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
            
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                cell_cost = cost_grid[neighbor[0], neighbor[1]]
                if cell_cost >= 0.95:
                    continue
                    
                dist = np.sqrt(dx**2 + dy**2) * resolution
                move_cost = dist * (1.0 + cell_cost * cost_multiplier)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return None

def fast_marching_planner(cost_grid: np.ndarray, goal_cell: Tuple[int, int],
                          resolution: float) -> np.ndarray:
    """Fast Marching Method: returns a 'time of arrival' field.
    Gradient descent on this field gives the optimal path from any point.
    Uses scikit-fmm if available, otherwise falls back to Dijkstra."""
    rows, cols = cost_grid.shape
    
    try:
        import skfmm
        phi = np.ones_like(cost_grid)
        phi[goal_cell[0], goal_cell[1]] = 0
        speed = 1.0 - np.clip(cost_grid, 0, 0.99)
        speed[cost_grid >= 0.95] = 1e-3
        t = skfmm.travel_time(phi, speed, dx=resolution)
        return t
    except ImportError:
        # Fallback to simple Dijkstra distances
        t = np.full_like(cost_grid, np.inf)
        if not (0 <= goal_cell[0] < rows and 0 <= goal_cell[1] < cols):
            return t
            
        open_set = []
        heapq.heappush(open_set, (0.0, goal_cell))
        t[goal_cell[0], goal_cell[1]] = 0.0
        
        while open_set:
            dist, current = heapq.heappop(open_set)
            
            if dist > t[current[0], current[1]]:
                continue
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    cell_cost = cost_grid[neighbor[0], neighbor[1]]
                    if cell_cost >= 0.95:
                        continue
                        
                    step_dist = np.sqrt(dx**2 + dy**2) * resolution
                    move_time = step_dist / max(1.0 - cell_cost, 0.05)
                    new_dist = dist + move_time
                    
                    if new_dist < t[neighbor[0], neighbor[1]]:
                        t[neighbor[0], neighbor[1]] = new_dist
                        heapq.heappush(open_set, (new_dist, neighbor))
        return t

def extract_velocity_command(arrival_field: np.ndarray, robot_cell: Tuple[int, int],
                             robot_yaw: float, resolution: float,
                             max_speed: float = 0.5) -> Tuple[float, float, float]:
    """Follow gradient of arrival field to compute (vx, vy, yaw_rate) commands."""
    rows, cols = arrival_field.shape
    rx, ry = robot_cell
    
    if not (1 <= rx < rows - 1 and 1 <= ry < cols - 1):
        return 0.0, 0.0, 0.0
        
    if np.isinf(arrival_field[rx, ry]):
        return 0.0, 0.0, 0.0
        
    # Central difference gradient
    dx = (arrival_field[rx+1, ry] - arrival_field[rx-1, ry]) / (2.0 * resolution)
    dy = (arrival_field[rx, ry+1] - arrival_field[rx, ry-1]) / (2.0 * resolution)
    
    # Gradient points uphill, we want to go downhill
    grad_x, grad_y = -dx, -dy
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    
    if grad_norm < 1e-6:
        return 0.0, 0.0, 0.0
        
    # Normalize
    dir_x = grad_x / grad_norm
    dir_y = grad_y / grad_norm
    
    # Convert direction to robot frame using yaw
    c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
    robot_dir_x = dir_x * c - dir_y * s
    robot_dir_y = dir_x * s + dir_y * c
    
    vx = robot_dir_x * max_speed
    vy = robot_dir_y * max_speed
    
    # P-control for yaw towards the gradient direction
    target_yaw = np.arctan2(dir_y, dir_x)
    yaw_error = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = np.clip(yaw_error * 1.5, -1.0, 1.0)
    
    return float(vx), float(vy), float(yaw_rate)
