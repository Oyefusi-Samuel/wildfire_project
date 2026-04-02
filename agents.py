from config import *
import math

class Wumpus:
    def __init__(self, obs, start_cell):
        self.obs = obs
        self.cell = start_cell
        self.score = 0
        self.speed = 4.0 
        self.move_acc = 0.0

    def step(self, t, dt):
        # Ignite adjacent obstacles
        r, c = self.cell
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if self.obs.ignite(r + dr, c + dc, t):
                    self.score += 1 # Point for igniting
        return []

class Firetruck:
    def __init__(self, obs, prm, start_pos):
        self.obs = obs
        self.prm = prm
        self.pos = list(start_pos) # [x, y, theta]
        self.score = 0
        self.wait = 0.0

    def step(self, t, dt):
        x, y, theta = self.pos
        # Check if near a fire to extinguish
        burning = [(r, c) for (r, c), s in self.obs.state.items() if s == State.BURNING]
        for br, bc in burning:
            cx, cy = self.obs.cell_xy(br, bc)
            if math.hypot(cx - x, cy - y) <= EXTINGUISH_R:
                self.wait += dt
                if self.wait >= EXTINGUISH_T:
                    if self.obs.extinguish(br, bc):
                        self.score += 2 # Points for putting it out
                    self.wait = 0.0
                return [] # Stay still while extinguishing
        
        # Simple constant velocity forward if not at a fire
        self.pos[0] += TRUCK_MAX_V * math.cos(theta) * dt
        self.pos[1] += TRUCK_MAX_V * math.sin(theta) * dt
        return []