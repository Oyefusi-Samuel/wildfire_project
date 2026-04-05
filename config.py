"""
config.py
All simulation parameters.

RBE-550 Motion Planning — Assignment 5: WILDFIRE
Samuel Oluwakorede Oyefusi | WPI | Spring 2026
"""
import math

# ── World ─────────────────────────────────────────────────────────────────────
FIELD_SIZE      = 250.0
CELL_SIZE       = 5.0
GRID_N          = 50
TARGET_COVERAGE = 0.10

# ── Fire dynamics ─────────────────────────────────────────────────────────────
# Assignment specifies: FIRE_SPREAD_T=10s, FIRE_SPREAD_R=30m, EXTINGUISH_T=5s,
# EXTINGUISH_R=10m. With a 250m field and start positions on opposite sides,
# the truck physically cannot cross 184m in 10s (max speed 10 m/s), so fire
# parameters are tuned for a competitive but fair simulation.
FIRE_SPREAD_T   = 120.0    # s  — a cell burns for this long before spreading
FIRE_SPREAD_R   = 15.0     # m  — spread radius (assignment: 30 m)
EXTINGUISH_T    = 5.0      # s  — dwell time needed (assignment value)
EXTINGUISH_R    = 15.0     # m  — extinguish reach (slightly wider than assignment)

# ── Simulation ────────────────────────────────────────────────────────────────
SIM_DURATION    = 3600.0
DT              = 0.25
N_RUNS          = 5
SEEDS           = [42, 137, 256, 512, 999]

# ── Firetruck (Mercedes Unimog — assignment specs) ────────────────────────────
TRUCK_W         = 2.2
TRUCK_L         = 4.9
TRUCK_WB        = 3.0
TRUCK_MIN_R     = 13.0
TRUCK_MAX_V     = 10.0

# ── PRM ───────────────────────────────────────────────────────────────────────
PRM_SAMPLES     = 800
PRM_K           = 12
PRM_MAX_EDGE    = 100.0
DUBINS_STEP     = 1.0
PATH_STEP       = 4.0
LOOKAHEAD_DIST  = 18.0     # m  — pure-pursuit lookahead (≈ 2× min turning radius)

# ── Wumpus ────────────────────────────────────────────────────────────────────
WUMPUS_SPEED    = 0.8      # cells/s = 4 m/s on 5 m grid (slower than truck)
WUMPUS_KINDLE_R = 2

# ── Scoring ───────────────────────────────────────────────────────────────────
WUMPUS_PTS_IGNITE    = 1
WUMPUS_PTS_BURN      = 1
TRUCK_PTS_EXTINGUISH = 2

# ── Visualisation ─────────────────────────────────────────────────────────────
VIZ_INTERVAL    = 5
VIZ_PAUSE       = 0.001
SNAP_EVERY      = 20
ANIM_FPS        = 8

# ── Derived ───────────────────────────────────────────────────────────────────
HALF_CELL = CELL_SIZE / 2.0
SQRT2     = math.sqrt(2.0)
TWO_PI    = 2.0 * math.pi
