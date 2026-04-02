from config import *
from discrete_planner import AStarPlanner
from sampling_planner import PRMPlanner

class Wumpus:
    def __init__(self, obs, start_cell):
        self.obs = obs
        self.cell = start_cell
        self.planner = AStarPlanner(obs)
        self.score = 0

    def step(self, t, dt):
        # Logic to move toward nearest 'Intact' cluster and ignite it
        r, c = self.cell
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if self.obs.ignite(r+dr, c+dc, t):
                self.score += 1

class Firetruck:
    def __init__(self, obs, prm, start_pos):
        self.obs = obs
        self.prm = prm
        self.pos = list(start_pos) # [x, y, theta]
        self.score = 0

    def step(self, t, dt):
        # Logic to find highest spread-risk fire and navigate using PRM
        pass