import math
import numpy as np
from enum import Enum

# --- Simulation Constants ---
FIELD_SIZE       = 250.0   # metres
CELL_SIZE        = 5.0     # 5m cells = 50x50 grid
GRID_N           = 50
TARGET_COVERAGE  = 0.10    
FIRE_SPREAD_T    = 10.0    # seconds before spreading
FIRE_SPREAD_R    = 30.0    # spread radius in metres
EXTINGUISH_T     = 5.0     # wait time to extinguish
EXTINGUISH_R     = 12.0    # range of hose
SIM_DURATION     = 3600.0  
DT               = 0.5     

# --- Truck Kinematics (Mercedes Unimog) ---
TRUCK_W          = 2.2
TRUCK_L          = 4.9
TRUCK_WB         = 3.0     # Wheelbase
TRUCK_MIN_R      = 13.0    # Min turning radius
TRUCK_MAX_V      = 10.0    # Max velocity m/s

class State(Enum):
    INTACT       = 0
    BURNING      = 1
    EXTINGUISHED = 2
    BURNED       = 3

class ObstacleManager:
    def __init__(self, grid):
        self.grid = grid.copy()
        self.state = {(r, c): State.INTACT for r in range(GRID_N) 
                      for c in range(GRID_N) if grid[r, c]}
        self.burn_start = {}

    def cell_xy(self, r, c):
        return (c * CELL_SIZE + CELL_SIZE/2, r * CELL_SIZE + CELL_SIZE/2)

    def is_passable(self, r, c):
        if not (0 <= r < GRID_N and 0 <= c < GRID_N): return False
        return not self.grid[r, c] or self.state.get((r, c)) in (State.EXTINGUISHED, State.BURNED)

    def ignite(self, r, c, t):
        if self.state.get((r, c)) == State.INTACT:
            self.state[(r, c)] = State.BURNING
            self.burn_start[(r, c)] = t
            return True
        return False

    def update(self, t):
        newly_burned, newly_ignited = [], []
        for (r, c), s in list(self.state.items()):
            if s == State.BURNING and (t - self.burn_start.get((r, c), t)) >= FIRE_SPREAD_T:
                cx, cy = self.cell_xy(r, c)
                self.state[(r, c)] = State.BURNED
                newly_burned.append((r, c))
                # Spread fire to nearby intact cells
                for (r2, c2), s2 in self.state.items():
                    if s2 == State.INTACT:
                        x2, y2 = self.cell_xy(r2, c2)
                        if math.hypot(cx - x2, cy - y2) <= FIRE_SPREAD_R:
                            if self.ignite(r2, c2, t): newly_ignited.append((r2, c2))
        return newly_burned, newly_ignited