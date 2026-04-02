import heapq
import math
from config import *

class AStarPlanner:
    def __init__(self, obs):
        self.obs = obs

    def plan(self, start, goal):
        if not self.obs.is_passable(*goal): return []
        
        open_list = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                return self._reconstruct(came_from, current)

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if self.obs.is_passable(*neighbor):
                    step_cost = math.sqrt(dr**2 + dc**2)
                    tentative_g = g_score[current] + step_cost
                    
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = math.hypot(neighbor[0]-goal[0], neighbor[1]-goal[1])
                        heapq.heappush(open_list, (tentative_g + h, neighbor))
        return []

    def _reconstruct(self, came_from, current):
        path = [current]; 
        while current in came_from:
            current = came_from[current]; path.append(current)
        return path[::-1]