import math
import numpy as np
import itertools
from enum import Enum

# --- Visualization Setup ---
# matplotlib.use("Agg") # Commented out to allow pop-up windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- World constants ---
FIELD_SIZE       = 250.0   # metres
CELL_SIZE        = 5.0     # metres per grid cell
GRID_N           = 50      # cells per side (250/5)
TARGET_COVERAGE  = 0.10    # 10 % obstacle coverage
FIRE_SPREAD_T    = 10.0    # s before burning spreads
FIRE_SPREAD_R    = 30.0    # metres spread radius
EXTINGUISH_T     = 5.0     # s truck must wait within range
EXTINGUISH_R     = 12.0    # metres truck-to-obstacle range
SIM_DURATION     = 3600.0  # simulation seconds
DT               = 0.5     # time step (s)

# --- Firetruck Kinematics (Mercedes Unimog) ---
TRUCK_W          = 2.2
TRUCK_L          = 4.9
TRUCK_WB         = 3.0     # wheelbase
TRUCK_MIN_R      = 13.0    # minimum turning radius
TRUCK_MAX_V      = 10.0    # m/s

class State(Enum):
    INTACT      = 0
    BURNING     = 1
    EXTINGUISHED= 2
    BURNED      = 3

# --- Tetromino Generation Logic ---
_BASE_SHAPES = [
    [(0,0),(0,1),(0,2),(0,3)], [(0,0),(1,0),(1,1),(1,2)], [(0,0),(0,1),(0,2),(1,2)],
    [(0,0),(0,1),(1,0),(1,1)], [(0,1),(0,2),(1,0),(1,1)], [(0,0),(1,0),(1,1),(2,1)],
    [(0,0),(0,1),(0,2),(1,1)]
]

def _normalise(pts):
    a = np.array(pts)
    a -= a.min(axis=0)
    return tuple(sorted(map(tuple, a.tolist())))

def _rotations(shape):
    unique = set()
    pts = np.array(shape)
    for _ in range(4):
        pts = np.column_stack((-pts[:,1], pts[:,0]))
        pts -= pts.min(axis=0)
        unique.add(_normalise(pts.tolist()))
    return [list(s) for s in unique]

ALL_SHAPES = list(itertools.chain.from_iterable(_rotations(s) for s in _BASE_SHAPES))

def generate_grid(seed=None):
    """Generates the 10% coverage obstacle grid."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((GRID_N, GRID_N), dtype=bool)
    target = int(GRID_N * GRID_N * TARGET_COVERAGE)
    placed = set()
    for _ in range(1000):
        if len(placed) >= target: break
        shape = ALL_SHAPES[rng.integers(len(ALL_SHAPES))]
        r0, c0 = rng.integers(0, GRID_N, size=2)
        cells = [(r0+dr, c0+dc) for dr, dc in shape]
        if all(0 <= r < GRID_N and 0 <= c < GRID_N for r, c in cells):
            for r, c in cells:
                placed.add((r, c))
                grid[r, c] = True
    return grid, list(placed)

# --- Obstacle Manager ---
class ObstacleManager:
    def __init__(self, grid):
        self.grid = grid.copy()
        self.state = {(r, c): State.INTACT for r in range(GRID_N) 
                      for c in range(GRID_N) if grid[r, c]}
        self.burn_start = {}

    def cell_xy(self, r, c):
        return (c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2)

    def is_passable(self, r, c):
        if not (0 <= r < GRID_N and 0 <= c < GRID_N): return False
        if not self.grid[r, c]: return True
        return self.state.get((r, c)) in (State.EXTINGUISHED, State.BURNED)

    def ignite(self, r, c, t):
        if self.state.get((r, c)) == State.INTACT:
            self.state[(r, c)] = State.BURNING
            self.burn_start[(r, c)] = t
            return True
        return False

    def extinguish(self, r, c):
        if self.state.get((r, c)) == State.BURNING:
            self.state[(r, c)] = State.EXTINGUISHED
            return True
        return False

    def update(self, t):
        """Advances fire spread logic."""
        newly_burned, newly_ignited = [], []
        for (r, c), s in list(self.state.items()):
            if s != State.BURNING: continue
            if t - self.burn_start.get((r, c), t) >= FIRE_SPREAD_T:
                self.state[(r, c)] = State.BURNED
                newly_burned.append((r, c))
                cx, cy = self.cell_xy(r, c)
                for (r2, c2), s2 in list(self.state.items()):
                    if s2 == State.INTACT:
                        x2, y2 = self.cell_xy(r2, c2)
                        if math.hypot(cx - x2, cy - y2) <= FIRE_SPREAD_R:
                            if self.ignite(r2, c2, t): newly_ignited.append((r2, c2))
        return newly_burned, newly_ignited