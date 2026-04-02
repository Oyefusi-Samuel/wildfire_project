import heapq
import math
from collections import defaultdict

class AStarPlanner:
    def __init__(self, obs):
        self.obs = obs
        self.dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        self.costs = [1.0, 1.0, 1.0, 1.0, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]

    def plan(self, start, goal):
        open_list = [(0, start)]
        g_score = {start: 0}
        came_from = {}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                return self._reconstruct(came_from, current)

            for i, (dr, dc) in enumerate(self.dirs):
                neighbor = (current[0] + dr, current[1] + dc)
                if self.obs.is_passable(*neighbor):
                    tentative_g = g_score[current] + self.costs[i]
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + math.hypot(neighbor[0]-goal[0], neighbor[1]-goal[1])
                        heapq.heappush(open_list, (f_score, neighbor))
        return []

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]