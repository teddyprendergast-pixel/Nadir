import numpy as np

def flat_terrain(rows, cols, resolution) -> np.ndarray:
    """Generate a flat terrain."""
    return np.zeros((rows, cols), dtype=np.float32)

def rough_terrain(rows, cols, resolution, amplitude, rng) -> np.ndarray:
    """Generate rough terrain using random noise."""
    # Simplistic roughness (normally use Perlin)
    return rng.uniform(-amplitude, amplitude, size=(rows, cols)).astype(np.float32)

def slope_terrain(rows, cols, resolution, angle_deg) -> np.ndarray:
    """Generate a sloped terrain."""
    angle_rad = np.deg2rad(angle_deg)
    slope = np.tan(angle_rad)
    x = np.arange(cols) * resolution
    heights = x * slope
    return np.tile(heights, (rows, 1)).astype(np.float32)

def stairs_terrain(rows, cols, resolution, step_height, step_width) -> np.ndarray:
    """Generate staircase terrain."""
    steps = np.floor(np.arange(cols) * resolution / step_width)
    heights = steps * step_height
    return np.tile(heights, (rows, 1)).astype(np.float32)

def stepping_stones_terrain(rows, cols, resolution, stone_size, gap_size, rng) -> np.ndarray:
    """Generate stepping stones."""
    grid = np.zeros((rows, cols), dtype=np.float32) - 0.5  # pit
    stone_cells = int(stone_size / resolution)
    gap_cells = int(gap_size / resolution)
    
    for i in range(0, rows, stone_cells + gap_cells):
        for j in range(0, cols, stone_cells + gap_cells):
            grid[i:i+stone_cells, j:j+stone_cells] = rng.uniform(-0.05, 0.05)
    return grid

def forest_terrain(rows, cols, resolution, rng) -> np.ndarray:
    """Generate forest-like terrain with roots/logs."""
    base = rough_terrain(rows, cols, resolution, 0.05, rng)
    # Add random logs
    for _ in range(int((rows * cols) / 10000)):
        start = (rng.integers(0, rows), rng.integers(0, cols))
        length = rng.integers(10, 50)
        direction = rng.integers(0, 2)
        if direction == 0 and start[0] + length < rows:
            base[start[0]:start[0]+length, start[1]:start[1]+3] += 0.1
        elif direction == 1 and start[1] + length < cols:
            base[start[0]:start[0]+3, start[1]:start[1]+length] += 0.1
    return base

class TerrainCurriculum:
    """Tracks per-env difficulty level and handles terrain updates."""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.levels = np.zeros(num_envs, dtype=np.int32)
        self.survival_distances = np.zeros(num_envs, dtype=np.float32)
        
    def update(self, done_envs, distances):
        """Promote/demote envs based on performance."""
        promotions = distances > 5.0
        demotions = distances < 1.0
        
        self.levels = np.clip(self.levels + promotions - demotions, 0, 9)
        self.survival_distances[done_envs] = 0.0
