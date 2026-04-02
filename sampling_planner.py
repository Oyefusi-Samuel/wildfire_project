import math
import random
import networkx as nx
from scipy.spatial import KDTree
from config import *

def get_dubins_path(q1, q2, rmin):
    # Simplified LSL/RSR Dubins logic for PRM edges
    dist = math.hypot(q2[0]-q1[0], q2[1]-q1[1])
    return dist, None # In practice, returns path samples

class PRMPlanner:
    def __init__(self, obs):
        self.obs = obs
        self.nodes = []
        self.graph = nx.Graph()

    def build(self, n_samples=900):
        while len(self.nodes) < n_samples:
            x, y = random.uniform(0, FIELD_SIZE), random.uniform(0, FIELD_SIZE)
            if self.obs.is_passable(int(y/CELL_SIZE), int(x/CELL_SIZE)):
                self.nodes.append((x, y, random.uniform(-math.pi, math.pi)))
        
        coords = np.array([(n[0], n[1]) for n in self.nodes])
        tree = KDTree(coords)
        for i, q in enumerate(self.nodes):
            _, idxs = tree.query([q[0], q[1]], k=12)
            for j in idxs:
                if i != j:
                    dist = math.hypot(q[0]-self.nodes[j][0], q[1]-self.nodes[j][1])
                    if dist < 50.0: # Edge length limit
                        self.graph.add_edge(i, j, weight=dist)

    def query(self, start, goal):
        # Find nearest nodes in graph and run A* over roadmap
        try:
            # logic to connect start/goal to PRM nodes
            return [], 0.0 # Returns path and planning time
        except nx.NetworkXNoPath:
            return [], 0.0