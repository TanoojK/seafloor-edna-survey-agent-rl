import numpy as np
from perlin_noise import PerlinNoise
import seaborn as sns
import matplotlib.pyplot as plt

default_octaves = 6
default_threshold_low = 0.1
default_threshold_high = 0.92

def generate_terrain(grid_size = 50 ,
                     octaves=default_octaves,
                     seed = None):
    if seed is None:
        seed = np.random.randint(0, 10_000)
    
    noise = PerlinNoise(octaves=octaves, seed=seed)
    elevation_map = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            elevation_map[i][j] = noise([i / grid_size, j / grid_size])
    
    #Normalizing the elevation map to be between 0 and 1
    elevation_map = (elevation_map - np.min(elevation_map)) / (np.max(elevation_map) - np.min(elevation_map))

    return elevation_map

def compute_impassable_terrain(elevation_map: np.ndarray, 
                               threshold_low: float = default_threshold_low,
                               threshold_high: float = default_threshold_high):
    return (elevation_map < threshold_low) | (elevation_map > threshold_high)

def elevation_step_cost(elevation_map: np.ndarray,
                        from_pos: tuple,
                        to_pos: tuple,
                        cost_per_unit: float = 1.0) -> float:
    delta = elevation_map[to_pos] - elevation_map[from_pos]
    return delta * cost_per_unit  # delta in [-1,1]-ish range * 0.1 handled by caller scaling

def elevation_step_cost_flat(elevation: np.ndarray,
                            from_pos: tuple,
                            to_pos: tuple,
                            flat_cost: float = 0.1) -> float:
    delta = elevation[to_pos] - elevation[from_pos]
    if delta > 0:
        return flat_cost
    elif delta < 0:
        return -flat_cost
    return 0.0

def terrain_summary(elevation_map: np.ndarray,
                    impassable_mask: np.ndarray) -> dict:
    return {"min": float(elevation_map.min()),
            "max": float(elevation_map.max()),
            "mean": float(elevation_map.mean()),
            "pct_impassable": float(impassable_mask.mean() * 100),
    }
    
