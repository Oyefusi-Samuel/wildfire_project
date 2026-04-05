"""
discrete_planner.py
8-connected grid A* for the Wumpus agent.

Heuristic: Euclidean distance (admissible).
Edge costs: 1.0 cardinal, sqrt(2) diagonal.

Reference:
    Hart, P.E., Nilsson, N.J., & Raphael, B. (1968).
    IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.

RBE-550 Motion Planning — Assignment 5: WILDFIRE
Samuel Oluwakorede Oyefusi | WPI | Spring 2026
"""
import math, heapq, time
from collections import defaultdict
from config import SQRT2

_DIRS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1,-1,SQRT2), (-1, 1,SQRT2), (1,-1,SQRT2), (1, 1,SQRT2),
]


class AStarPlanner:
    """
    Grid A* planner for the Wumpus.
    plan(start, goal) -> (path: list[(r,c)], elapsed_s: float)
    """

    def __init__(self, obs):
        self.obs = obs

    def plan(self, start: tuple, goal: tuple) -> tuple:
        t0   = time.perf_counter()
        goal = self._nearest_passable(goal, start)
        if goal is None:
            return [], time.perf_counter() - t0

        g   = defaultdict(lambda: math.inf)
        g[start] = 0.0
        prev = {}
        heap = [(self._h(start, goal), start)]
        seen = {start}

        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                return self._path(prev, start, goal), time.perf_counter()-t0
            for nbr, cost in self._nbrs(cur):
                ng = g[cur] + cost
                if ng < g[nbr]:
                    g[nbr] = ng; prev[nbr] = cur
                    if nbr not in seen:
                        heapq.heappush(heap, (ng + self._h(nbr, goal), nbr))
                        seen.add(nbr)
        return [], time.perf_counter() - t0

    def _h(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _nbrs(self, cell):
        r, c = cell
        return [((r+dr, c+dc), cost)
                for dr, dc, cost in _DIRS
                if self.obs.is_passable(r+dr, c+dc)]

    def _nearest_passable(self, goal, hint):
        if self.obs.is_passable(*goal):
            return goal
        cands = [(goal[0]+dr, goal[1]+dc)
                 for dr, dc, _ in _DIRS
                 if self.obs.is_passable(goal[0]+dr, goal[1]+dc)]
        return min(cands, key=lambda n: abs(n[0]-hint[0])+abs(n[1]-hint[1])) \
               if cands else None

    def _path(self, prev, start, goal):
        path, cur = [], goal
        while cur != start:
            path.append(cur); cur = prev[cur]
        path.append(start); path.reverse()
        return path
