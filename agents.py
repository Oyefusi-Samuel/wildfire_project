import math
from config import *
from discrete_planner import AStarPlanner

class Wumpus:
    def __init__(self, obs, start_cell):
        self.obs = obs
        self.cell = start_cell
        self.planner = AStarPlanner(obs)
        self.path = []

    def step(self, t, dt):
        # Target the closest INTACT obstacle
        if not self.path:
            intact = [pos for pos, s in self.obs.state.items() if s == State.INTACT]
            if intact:
                target = min(intact, key=lambda p: math.hypot(p[0]-self.cell[0], p[1]-self.cell[1]))
                self.path = self.planner.plan(self.cell, target)

        if self.path:
            self.cell = self.path.pop(0)
            # Try to ignite surrounding area
            r, c = self.cell
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    self.obs.ignite(r+dr, c+dc, t)

class Firetruck:
    def __init__(self, obs, prm, start_pos):
        self.obs = obs
        self.prm = prm
        self.pos = list(start_pos) # [x, y, theta]
        self.extinguish_timer = 0.0

    def step(self, t, dt):
        x, y, theta = self.pos
        # 1. Check if we are currently at a fire
        burning = [pos for pos, s in self.obs.state.items() if s == State.BURNING]
        
        for br, bc in burning:
            bx, by = self.obs.cell_xy(br, bc)
            if math.hypot(bx - x, by - y) <= EXTINGUISH_R:
                self.extinguish_timer += dt
                if self.extinguish_timer >= EXTINGUISH_T:
                    self.obs.extinguish(br, bc)
                    self.extinguish_timer = 0.0
                return # Stop to extinguish
        
        # 2. Simple movement (Advance toward nearest fire)
        if burning:
            bx, by = self.obs.cell_xy(*burning[0])
            angle_to_fire = math.atan2(by - y, bx - x)
            self.pos[2] = angle_to_fire # Instant turn for now
            self.pos[0] += TRUCK_MAX_V * math.cos(angle_to_fire) * dt
            self.pos[1] += TRUCK_MAX_V * math.sin(angle_to_fire) * dt