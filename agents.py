"""
agents.py
Environment generation, obstacle state management, Wumpus and Firetruck agents.

Key improvements over baseline:
  - Firetruck follows dense Dubins curves (kinematically valid smooth motion)
  - Reachability filtering: truck only targets fires it can arrive at in time
  - Hunt mode: truck chases Wumpus when all fires are extinguished/burned
  - Wumpus flees when in hunt range

RBE-550 Motion Planning — Assignment 5: WILDFIRE
Samuel Oluwakorede Oyefusi | WPI | Spring 2026
"""
import math, itertools
from enum import Enum
import numpy as np

from config import (
    GRID_N, CELL_SIZE, HALF_CELL, TARGET_COVERAGE,
    FIRE_SPREAD_T, FIRE_SPREAD_R,
    EXTINGUISH_R, EXTINGUISH_T, TRUCK_PTS_EXTINGUISH,
    FIELD_SIZE, TRUCK_L, TRUCK_MIN_R, TRUCK_MAX_V,
    WUMPUS_SPEED, WUMPUS_KINDLE_R,
    WUMPUS_PTS_IGNITE, WUMPUS_PTS_BURN,
    PATH_STEP, LOOKAHEAD_DIST,
)
from discrete_planner  import AStarPlanner
from sampling_planner  import PRMPlanner, dubins_dense_path


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    INTACT       = 0
    BURNING      = 1
    EXTINGUISHED = 2
    BURNED       = 3


# ─────────────────────────────────────────────────────────────────────────────
# Tetromino grid generator
# ─────────────────────────────────────────────────────────────────────────────

_BASE = [
    [(0,0),(0,1),(0,2),(0,3)],
    [(0,0),(1,0),(1,1),(1,2)],
    [(0,0),(0,1),(0,2),(1,2)],
    [(0,0),(0,1),(1,0),(1,1)],
    [(0,1),(0,2),(1,0),(1,1)],
    [(0,0),(1,0),(1,1),(2,1)],
    [(0,0),(0,1),(0,2),(1,1)],
]

def _norm(pts):
    a=np.array(pts); a-=a.min(axis=0)
    return tuple(sorted(map(tuple,a.tolist())))

def _rots(s):
    seen,pts=set(),np.array(s)
    for _ in range(4):
        pts=np.column_stack((-pts[:,1],pts[:,0])); pts-=pts.min(axis=0)
        seen.add(_norm(pts.tolist()))
    return [list(x) for x in seen]

_SHAPES = list(itertools.chain.from_iterable(_rots(s) for s in _BASE))


def generate_grid(seed=None):
    rng=np.random.default_rng(seed)
    grid=np.zeros((GRID_N,GRID_N),dtype=bool)
    placed=set()
    target=int(GRID_N*GRID_N*TARGET_COVERAGE)
    for _ in range(200_000):
        if len(placed)>=target: break
        shape=_SHAPES[rng.integers(len(_SHAPES))]
        r0,c0=int(rng.integers(0,GRID_N)),int(rng.integers(0,GRID_N))
        cells=[(r0+dr,c0+dc) for dr,dc in shape]
        if all(0<=r<GRID_N and 0<=c<GRID_N for r,c in cells):
            for r,c in cells:
                placed.add((r,c)); grid[r,c]=True
    return grid, list(placed)


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle manager
# ─────────────────────────────────────────────────────────────────────────────

class ObstacleManager:
    def __init__(self, grid):
        self.grid       = grid.copy()
        self.state      = {(r,c):State.INTACT for r in range(GRID_N)
                           for c in range(GRID_N) if grid[r,c]}
        self.burn_start = {}
        self.flash      = {}    # (r,c) -> countdown for newly ignited flash

    def cell_xy(self, r, c):
        return (c*CELL_SIZE+HALF_CELL, r*CELL_SIZE+HALF_CELL)

    def is_passable(self, r, c):
        if not (0<=r<GRID_N and 0<=c<GRID_N): return False
        if not self.grid[r,c]: return True
        return self.state.get((r,c)) in (State.EXTINGUISHED, State.BURNED)

    def by_state(self, s):
        return [(r,c) for (r,c),st in self.state.items() if st==s]

    def ignite(self, r, c, sim_time):
        if self.state.get((r,c))==State.INTACT:
            self.state[(r,c)]=State.BURNING
            self.burn_start[(r,c)]=sim_time
            self.flash[(r,c)]=8   # flash for 8 frames
            return True
        return False

    def extinguish(self, r, c):
        if self.state.get((r,c))==State.BURNING:
            self.state[(r,c)]=State.EXTINGUISHED
            return True
        return False

    def burn_age(self, r, c, sim_time):
        """How many seconds this cell has been burning."""
        return sim_time - self.burn_start.get((r,c), sim_time)

    def time_to_spread(self, r, c, sim_time):
        """Seconds remaining before this burning cell spreads (negative = past)."""
        return FIRE_SPREAD_T - self.burn_age(r, c, sim_time)

    def update(self, sim_time):
        newly_burned, spread_targets = [], []
        for (r,c),s in list(self.state.items()):
            if s!=State.BURNING: continue
            if sim_time-self.burn_start.get((r,c),sim_time)<FIRE_SPREAD_T: continue
            cx,cy=self.cell_xy(r,c)
            for (r2,c2),s2 in self.state.items():
                if s2==State.INTACT:
                    x2,y2=self.cell_xy(r2,c2)
                    if math.hypot(cx-x2,cy-y2)<=FIRE_SPREAD_R:
                        spread_targets.append((r2,c2))
            self.state[(r,c)]=State.BURNED; newly_burned.append((r,c))

        newly_ignited=[cell for cell in spread_targets if self.ignite(*cell,sim_time)]

        # Tick down flash counters
        for k in list(self.flash):
            self.flash[k]-=1
            if self.flash[k]<=0: del self.flash[k]

        return newly_burned, newly_ignited


# ─────────────────────────────────────────────────────────────────────────────
# Wumpus
# ─────────────────────────────────────────────────────────────────────────────

_DIRS8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]


class Wumpus:
    """
    Grid-navigating arsonist.
    Goal: target the intact obstacle cluster with highest kindling potential.
    Arson: ignite all 8 adjacent obstacle cells each step.
    Flee: run away when truck enters hunt mode and gets close.
    """

    def __init__(self, obs, start):
        self.obs        = obs
        self.cell       = start
        self._planner   = AStarPlanner(obs)
        self._path      = []
        self._pi        = 0
        self._acc       = 0.0
        self.score      = 0
        self.plan_time  = 0.0
        self.n_plans    = 0
        self.caught     = False
        # Expose path for visualisation
        self.vis_path   = []

    def xy(self):
        return self.obs.cell_xy(*self.cell)

    def step(self, sim_time, dt):
        self._replan_if_needed()
        self._move(dt)
        return self._arson(sim_time)

    def award_burn_points(self, n):
        self.score += n * WUMPUS_PTS_BURN

    def _replan_if_needed(self):
        if self._path and self._pi < len(self._path): return
        goal = self._pick_goal()
        if goal is None: return
        path, t = self._planner.plan(self.cell, goal)
        self.plan_time += t; self.n_plans += 1
        self._path, self._pi = path, 0
        self.vis_path = list(path)

    def _move(self, dt):
        if not self._path or self._pi >= len(self._path): return
        self._acc += dt * WUMPUS_SPEED
        while self._acc >= 1.0 and self._pi < len(self._path):
            self.cell = self._path[self._pi]
            self._pi += 1; self._acc -= 1.0

    def _arson(self, sim_time):
        events = []
        r,c = self.cell
        for dr,dc in _DIRS8:
            if self.obs.ignite(r+dr, c+dc, sim_time):
                self.score += WUMPUS_PTS_IGNITE
                events.append(('ignite',(r+dr,c+dc)))
        return events

    def _pick_goal(self):
        intact = self.obs.by_state(State.INTACT)
        if not intact: return None
        r0,c0 = self.cell
        def _pri(cell):
            r,c=cell
            k=sum(1 for dr in range(-WUMPUS_KINDLE_R,WUMPUS_KINDLE_R+1)
                    for dc in range(-WUMPUS_KINDLE_R,WUMPUS_KINDLE_R+1)
                    if self.obs.state.get((r+dr,c+dc))==State.INTACT)
            return -k + 0.5*(abs(r-r0)+abs(c-c0))
        return min(intact,key=_pri)


# ─────────────────────────────────────────────────────────────────────────────
# Firetruck
# ─────────────────────────────────────────────────────────────────────────────

class Firetruck:
    """
    Ackermann-steered firetruck driven by PRM + 4-family Dubins paths.

    Improvements over baseline:
    - Follows kinematically valid dense Dubins curves (not raw PRM waypoints)
    - Reachability filter: only targets fires reachable before they spread
    - Hunt mode: navigates toward Wumpus when no fires remain
    """

    def __init__(self, obs, prm, start):
        self.obs         = obs
        self.prm         = prm
        self.pos         = list(start)   # [x, y, theta]
        self._path       = []            # dense (x,y,theta) waypoints
        self._pi         = 0
        self._goal_cell  = None
        self._wait       = 0.0
        self._wait_cell  = None
        self.score       = 0
        self.plan_time   = 0.0
        self.n_plans     = 0
        # Hunt mode
        self.hunt_mode   = False
        self._hunt_wumpus = None
        self._hunt_cooldown = 0.0
        # Extinguish progress (0-1) for visualisation
        self.ext_progress = 0.0
        self.ext_target   = None

    def xy(self):
        return self.pos[0], self.pos[1]

    def activate_hunt(self, wumpus):
        self.hunt_mode = True
        self._hunt_wumpus = wumpus
        self._path = []; self._goal_cell = None

    def step(self, dt, sim_time):
        if self.hunt_mode:
            return self._hunt_step(dt)
        events = self._try_extinguish(dt, sim_time)
        if self._wait_cell is not None:
            return events
        self._replan_if_needed(sim_time)
        self._follow_path(dt)
        return events

    # ── Extinguishing ─────────────────────────────────────────────────────────

    def _try_extinguish(self, dt, sim_time):
        x,y = self.pos[0], self.pos[1]
        for cell in self.obs.by_state(State.BURNING):
            cx,cy = self.obs.cell_xy(*cell)
            if math.hypot(cx-x, cy-y) > EXTINGUISH_R: continue
            if self._wait_cell != cell:
                self._wait_cell, self._wait = cell, 0.0
            self._wait += dt
            self.ext_progress = min(1.0, self._wait / EXTINGUISH_T)
            self.ext_target = (cx, cy)
            if self._wait >= EXTINGUISH_T:
                if self.obs.extinguish(*cell):
                    self.score += TRUCK_PTS_EXTINGUISH
                self._wait_cell, self._wait = None, 0.0
                self.ext_progress = 0.0
                self.ext_target = None
                self._goal_cell, self._path = None, []
                return [('extinguish', cell)]
            return []
        self._wait_cell, self._wait = None, 0.0
        self.ext_progress = 0.0
        self.ext_target = None
        return []

    # ── Goal selection ────────────────────────────────────────────────────────

    def _replan_if_needed(self, sim_time):
        burning = self.obs.by_state(State.BURNING)
        still_valid = (self._goal_cell in burning
                       and self._path and self._pi < len(self._path))
        if still_valid: return

        target = self._pick_goal(burning, sim_time)
        if target is None: return

        goal = self._approach_pt(target)
        wpts, elapsed = self.prm.query(tuple(self.pos), goal)
        self.plan_time += elapsed; self.n_plans += 1

        if wpts:
            dense = dubins_dense_path(wpts, TRUCK_MIN_R, PATH_STEP)
            self._path = dense if dense else wpts
            self._pi   = 0
            self._goal_cell = target

    def _pick_goal(self, burning, sim_time):
        """
        Pick closest burning cell with an urgency bonus for fires about to spread.
        Closest-first prevents the truck wasting time crossing the whole map when
        a nearer fire is available.
        """
        if not burning: return None
        x, y = self.pos[0], self.pos[1]

        def _score(cell):
            cx, cy = self.obs.cell_xy(*cell)
            dist = max(1.0, math.hypot(cx - x, cy - y))
            t_left = self.obs.time_to_spread(*cell, sim_time)
            urgency = max(0.0, 1.0 - t_left / FIRE_SPREAD_T)
            return dist - urgency * 30.0   # closer is better; urgency pulls priority up

        return min(burning, key=_score)

    def _approach_pt(self, cell):
        cx,cy = self.obs.cell_xy(*cell)
        x,y,th = self.pos
        dx,dy = cx-x, cy-y
        dist = math.hypot(dx,dy)
        if dist < 1e-3: return (x,y,th)
        scale = (dist - max(1.0, EXTINGUISH_R*0.6)) / dist
        gx = max(TRUCK_L, min(FIELD_SIZE-TRUCK_L, x+dx*scale))
        gy = max(TRUCK_L, min(FIELD_SIZE-TRUCK_L, y+dy*scale))
        return (gx, gy, math.atan2(dy,dx))

    # ── Ackermann path following ───────────────────────────────────────────────

    def _follow_path(self, dt):
        """Pure-pursuit path follower with lookahead distance.
        Looks ahead a fixed distance along the path instead of targeting
        the immediate next waypoint, preventing overshoot oscillation and
        keeping the truck at full speed through curves."""
        if not self._path or self._pi >= len(self._path): return
        x, y, theta = self.pos

        # Advance past any waypoints already within lookahead distance
        while self._pi < len(self._path) - 1:
            tx, ty, _ = self._path[self._pi]
            if math.hypot(tx - x, ty - y) > LOOKAHEAD_DIST:
                break
            self._pi += 1

        tx, ty, _ = self._path[self._pi]
        desired = math.atan2(ty - y, tx - x)
        diff = (desired - theta + math.pi) % (2 * math.pi) - math.pi

        max_dth = TRUCK_MAX_V / TRUCK_MIN_R * dt
        dtheta = max(-max_dth, min(max_dth, diff * 1.2))

        # Reduce speed only when turning tightly
        vel = TRUCK_MAX_V * max(0.4, 1.0 - abs(diff) / math.pi)

        self.pos[0] = max(0.0, min(FIELD_SIZE, x + vel * math.cos(theta) * dt))
        self.pos[1] = max(0.0, min(FIELD_SIZE, y + vel * math.sin(theta) * dt))
        self.pos[2] = theta + dtheta

    def _hunt_step(self, dt):
        wumpus = self._hunt_wumpus
        if wumpus is None or wumpus.caught: return []

        wx,wy = wumpus.xy()
        tx,ty = self.xy()
        dist  = math.hypot(wx-tx, wy-ty)

        if dist < 15.0:
            wumpus.caught = True
            return [('wumpus_caught', wumpus.cell)]

        # Replan toward wumpus every 3 seconds (use cooldown)
        self._hunt_cooldown -= dt
        if self._hunt_cooldown <= 0 or not self._path or self._pi >= len(self._path):
            goal = (wx, wy, math.atan2(wy-ty, wx-tx))
            wpts, elapsed = self.prm.query(tuple(self.pos), goal)
            self.plan_time += elapsed; self.n_plans += 1
            if wpts:
                dense = dubins_dense_path(wpts, TRUCK_MIN_R, PATH_STEP)
                self._path = dense if dense else wpts
                self._pi   = 0
            self._hunt_cooldown = 3.0

        self._follow_path(dt)
        return []
