import math
import numpy as np
import itertools
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# --- World Constants ---
FIELD_SIZE       = 250.0   
CELL_SIZE        = 5.0     
GRID_N           = 50      
TARGET_COVERAGE  = 0.10    
FIRE_SPREAD_T    = 10.0    
FIRE_SPREAD_R    = 30.0    
EXTINGUISH_T     = 5.0     
EXTINGUISH_R     = 12.0    
SIM_DURATION     = 3600.0  
DT               = 0.5     

# --- Truck Kinematics ---
TRUCK_W, TRUCK_L = 2.2, 4.9
TRUCK_WB         = 3.0     
TRUCK_MIN_R      = 13.0    
TRUCK_MAX_V      = 10.0    

class State(Enum):
    INTACT       = 0
    BURNING      = 1
    EXTINGUISHED = 2
    BURNED       = 3

# --- Tetromino Generation ---
_BASE_SHAPES = [[(0,0),(0,1),(0,2),(0,3)], [(0,0),(1,0),(1,1),(1,2)], [(0,0),(0,1),(0,2),(1,2)],
                [(0,0),(0,1),(1,0),(1,1)], [(0,1),(0,2),(1,0),(1,1)], [(0,0),(1,0),(1,1),(2,1)],
                [(0,0),(0,1),(0,2),(1,1)]]

def generate_grid(seed=None):
    rng = np.random.default_rng(seed)
    grid = np.zeros((GRID_N, GRID_N), dtype=bool)
    target = int(GRID_N * GRID_N * TARGET_COVERAGE)
    placed_cells = []
    attempts = 0
    while len(placed_cells) < target and attempts < 2000:
        shape = _BASE_SHAPES[rng.integers(len(_BASE_SHAPES))]
        r0, c0 = rng.integers(0, GRID_N, size=2)
        cells = [(r0+dr, c0+dc) for dr, dc in shape]
        if all(0 <= r < GRID_N and 0 <= c < GRID_N and not grid[r,c] for r, c in cells):
            for r, c in cells:
                grid[r, c] = True
                placed_cells.append((r, c))
        attempts += 1
    return grid

class ObstacleManager:
    def __init__(self, grid):
        self.grid = grid
        self.state = {(r, c): State.INTACT for r in range(GRID_N) 
                      for c in range(GRID_N) if grid[r, c]}
        self.burn_start = {}

    def cell_xy(self, r, c):
        return (c * CELL_SIZE + CELL_SIZE/2, r * CELL_SIZE + CELL_SIZE/2)

    def is_passable(self, r, c):
        if not (0 <= r < GRID_N and 0 <= c < GRID_N): return False
        return not self.grid[r, c] or self.state.get((r, c)) in (State.EXTINGUISHED, State.BURNED)

    def ignite(self, r, c, t):
        if (r, c) in self.state and self.state[(r, c)] == State.INTACT:
            self.state[(r, c)] = State.BURNING
            self.burn_start[(r, c)] = t
            return True
        return False

    def update(self, t):
        newly_ignited = []
        for (r, c), s in list(self.state.items()):
            if s == State.BURNING and (t - self.burn_start.get((r, c), t)) >= FIRE_SPREAD_T:
                self.state[(r, c)] = State.BURNED
                cx, cy = self.cell_xy(r, c)
                for (r2, c2), s2 in self.state.items():
                    if s2 == State.INTACT:
                        x2, y2 = self.cell_xy(r2, c2)
                        if math.hypot(cx - x2, cy - y2) <= FIRE_SPREAD_R:
                            if self.ignite(r2, c2, t): newly_ignited.append((r2, c2))
        return newly_ignited

class Visualizer:
    def __init__(self, obs_manager):
        self.obs = obs_manager
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, FIELD_SIZE)
        self.ax.set_ylim(0, FIELD_SIZE)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#2c3e50') # Dark slate background
        
        # Static Obstacle patches
        self.patches = {}
        for (r, c) in self.obs.state:
            x, y = c * CELL_SIZE, r * CELL_SIZE
            rect = mpatches.Rectangle((x, y), CELL_SIZE, CELL_SIZE, color='green', alpha=0.6)
            self.ax.add_patch(rect)
            self.patches[(r, c)] = rect

    def render(self, truck_pos, wumpus_cell, t, w_score, t_score):
        # Update obstacle colors based on state
        for (r, c), state in self.obs.state.items():
            if state == State.BURNING: self.patches[(r, c)].set_color('#e67e22') # Orange
            elif state == State.BURNED: self.patches[(r, c)].set_color('#7f8c8d') # Gray
            elif state == State.EXTINGUISHED: self.patches[(r, c)].set_color('#3498db') # Blue

        # Draw Firetruck as a directed polygon
        x, y, theta = truck_pos
        truck_rect = mpatches.RegularPolygon((x, y), 3, radius=4, orientation=theta, color='red')
        
        # Wumpus
        wx, wy = self.obs.cell_xy(*wumpus_cell)
        wumpus_circ = mpatches.Circle((wx, wy), 3, color='purple')
        
        self.ax.set_title(f"Time: {t:.1f}s | Wumpus: {w_score} | Truck: {t_score}", color='white', fontsize=12)
        
        # Temporary artists for this frame
        frame_artists = [self.ax.add_patch(truck_rect), self.ax.add_patch(wumpus_circ)]
        plt.pause(0.001)
        
        # Clean up temporary artists for next frame
        for a in frame_artists:
            a.remove()