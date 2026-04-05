"""
sampling_planner.py
Dubins path computation (RSR, LSL, RSL, LSR) and PRM planner
for the Ackermann-steered firetruck.

References:
    Dubins, L.E. (1957). American Journal of Mathematics, 79(3), 497-516.
    Kavraki et al. (1996). IEEE T-RA, 12(4), 566-580.

RBE-550 Motion Planning — Assignment 5: WILDFIRE
Samuel Oluwakorede Oyefusi | WPI | Spring 2026
"""
import math, time, random
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

from config import (FIELD_SIZE, CELL_SIZE, GRID_N,
                    TRUCK_L, TRUCK_MIN_R,
                    PRM_SAMPLES, PRM_K, PRM_MAX_EDGE, DUBINS_STEP, TWO_PI)


# ─────────────────────────────────────────────────────────────────────────────
# Dubins primitives
# ─────────────────────────────────────────────────────────────────────────────

def _mod2pi(a): return a % TWO_PI

def _dubins_rsr(q1, q2, r):
    x1,y1,t1=q1; x2,y2,t2=q2
    c1x=x1+r*math.sin(t1); c1y=y1-r*math.cos(t1)
    c2x=x2+r*math.sin(t2); c2y=y2-r*math.cos(t2)
    d=math.hypot(c2x-c1x,c2y-c1y)
    if d<1e-9: return None
    ang=math.atan2(c2y-c1y,c2x-c1x)
    a1=_mod2pi(t1-ang+math.pi/2); a2=_mod2pi(ang-t2+math.pi/2)
    L=r*a1+d+r*a2
    def _s(n=20):
        pts,k=[],max(1,n//3)
        for i in range(k):
            th=t1-(i/k)*a1; pts.append((c1x-r*math.sin(th),c1y+r*math.cos(th),th))
        tx1=c1x-r*math.sin(t1-a1); ty1=c1y+r*math.cos(t1-a1)
        tx2=c2x-r*math.sin(t2+a2); ty2=c2y+r*math.cos(t2+a2)
        for i in range(k):
            s=i/k; pts.append((tx1+s*(tx2-tx1),ty1+s*(ty2-ty1),t1-a1))
        for i in range(k):
            th=t2+a2*(1-i/k); pts.append((c2x-r*math.sin(th),c2y+r*math.cos(th),th))
        pts.append((x2,y2,t2)); return pts
    return L,_s

def _dubins_lsl(q1, q2, r):
    x1,y1,t1=q1; x2,y2,t2=q2
    c1x=x1-r*math.sin(t1); c1y=y1+r*math.cos(t1)
    c2x=x2-r*math.sin(t2); c2y=y2+r*math.cos(t2)
    d=math.hypot(c2x-c1x,c2y-c1y)
    if d<1e-9: return None
    ang=math.atan2(c2y-c1y,c2x-c1x)
    a1=_mod2pi(ang-t1+math.pi/2); a2=_mod2pi(t2-ang+math.pi/2)
    L=r*a1+d+r*a2
    def _s(n=20):
        pts,k=[],max(1,n//3)
        for i in range(k):
            th=t1+(i/k)*a1; pts.append((c1x+r*math.sin(th),c1y-r*math.cos(th),th))
        tx1=c1x+r*math.sin(t1+a1); ty1=c1y-r*math.cos(t1+a1)
        tx2=c2x+r*math.sin(t2-a2); ty2=c2y-r*math.cos(t2-a2)
        for i in range(k):
            s=i/k; pts.append((tx1+s*(tx2-tx1),ty1+s*(ty2-ty1),t1+a1))
        for i in range(k):
            th=t2-a2*(1-i/k); pts.append((c2x+r*math.sin(th),c2y-r*math.cos(th),th))
        pts.append((x2,y2,t2)); return pts
    return L,_s

def _dubins_rsl(q1, q2, r):
    """Right arc → Straight → Left arc (cross family)."""
    x1,y1,t1=q1; x2,y2,t2=q2
    c1x=x1+r*math.sin(t1); c1y=y1-r*math.cos(t1)   # right centre
    c2x=x2-r*math.sin(t2); c2y=y2+r*math.cos(t2)   # left centre
    d=math.hypot(c2x-c1x,c2y-c1y)
    if d<2*r: return None
    ang=math.atan2(c2y-c1y,c2x-c1x)
    h=math.acos(2*r/d)
    a1=_mod2pi(t1-ang+math.pi/2+h)
    a2=_mod2pi(math.pi-h+ang-t2-math.pi/2)
    s=math.sqrt(max(0,d*d-4*r*r))
    L=r*a1+s+r*a2
    def _sfn(n=20):
        pts,k=[],max(1,n//3)
        for i in range(k):
            th=t1-(i/k)*a1; pts.append((c1x-r*math.sin(th),c1y+r*math.cos(th),th))
        end1_x=c1x-r*math.sin(t1-a1); end1_y=c1y+r*math.cos(t1-a1)
        end2_x=c2x+r*math.sin(t2+a2); end2_y=c2y-r*math.cos(t2+a2)
        th_mid=t1-a1
        for i in range(k):
            f=i/k; pts.append((end1_x+f*(end2_x-end1_x),end1_y+f*(end2_y-end1_y),th_mid))
        for i in range(k):
            th=t2+a2*(1-i/k); pts.append((c2x+r*math.sin(th),c2y-r*math.cos(th),th))
        pts.append((x2,y2,t2)); return pts
    return L,_sfn

def _dubins_lsr(q1, q2, r):
    """Left arc → Straight → Right arc (cross family)."""
    x1,y1,t1=q1; x2,y2,t2=q2
    c1x=x1-r*math.sin(t1); c1y=y1+r*math.cos(t1)   # left centre
    c2x=x2+r*math.sin(t2); c2y=y2-r*math.cos(t2)   # right centre
    d=math.hypot(c2x-c1x,c2y-c1y)
    if d<2*r: return None
    ang=math.atan2(c2y-c1y,c2x-c1x)
    h=math.acos(2*r/d)
    a1=_mod2pi(ang-t1+math.pi/2+h)
    a2=_mod2pi(math.pi-h-ang+t2+math.pi/2)
    s=math.sqrt(max(0,d*d-4*r*r))
    L=r*a1+s+r*a2
    def _sfn(n=20):
        pts,k=[],max(1,n//3)
        for i in range(k):
            th=t1+(i/k)*a1; pts.append((c1x+r*math.sin(th),c1y-r*math.cos(th),th))
        end1_x=c1x+r*math.sin(t1+a1); end1_y=c1y-r*math.cos(t1+a1)
        end2_x=c2x-r*math.sin(t2-a2); end2_y=c2y+r*math.cos(t2-a2)
        th_mid=t1+a1
        for i in range(k):
            f=i/k; pts.append((end1_x+f*(end2_x-end1_x),end1_y+f*(end2_y-end1_y),th_mid))
        for i in range(k):
            th=t2-a2*(1-i/k); pts.append((c2x-r*math.sin(th),c2y+r*math.cos(th),th))
        pts.append((x2,y2,t2)); return pts
    return L,_sfn

def _fallback(q1, q2):
    d=math.hypot(q2[0]-q1[0],q2[1]-q1[1])
    def _s(n=10):
        n=max(2,n)
        return [(q1[0]+i/(n-1)*(q2[0]-q1[0]),
                 q1[1]+i/(n-1)*(q2[1]-q1[1]),q1[2]) for i in range(n)]
    return d,_s

def dubins_shortest(q1, q2, rmin):
    """
    Shortest Dubins path among RSR, LSL, RSL, LSR families.
    Returns (length, sample_fn) where sample_fn(n) yields n (x,y,theta) points.
    """
    cands=[c for c in (_dubins_rsr(q1,q2,rmin), _dubins_lsl(q1,q2,rmin),
                       _dubins_rsl(q1,q2,rmin), _dubins_lsr(q1,q2,rmin))
           if c is not None]
    return min(cands,key=lambda c:c[0]) if cands else _fallback(q1,q2)


def dubins_dense_path(waypoints, rmin, step):
    """
    Expand a list of (x,y,theta) PRM waypoints into a dense list of
    kinematically valid points spaced ~step metres apart.
    """
    if not waypoints:
        return []
    dense = []
    for i in range(len(waypoints)-1):
        q1, q2 = waypoints[i], waypoints[i+1]
        length, sfn = dubins_shortest(q1, q2, rmin)
        n = max(4, int(length / step))
        pts = sfn(n)
        dense.extend(pts[:-1])   # skip last to avoid duplicating start of next
    dense.append(waypoints[-1])
    return dense


# ─────────────────────────────────────────────────────────────────────────────
# PRM planner
# ─────────────────────────────────────────────────────────────────────────────

class PRMPlanner:
    """
    PRM with 4-family Dubins local planner for the Ackermann firetruck.

    build(seed) -> build_time (s)
    query(start, goal) -> (waypoints, query_time_s)
    """

    def __init__(self, obs):
        self.obs        = obs
        self.nodes      = []
        self.G          = nx.Graph()
        self._kd        = None
        self.build_time = 0.0

    def build(self, seed=None):
        t0  = time.perf_counter()
        rng = random.Random(seed)

        while len(self.nodes) < PRM_SAMPLES:
            x = rng.uniform(TRUCK_L, FIELD_SIZE-TRUCK_L)
            y = rng.uniform(TRUCK_L, FIELD_SIZE-TRUCK_L)
            if self._free(x, y):
                self.nodes.append((x, y, rng.uniform(-math.pi, math.pi)))

        xy       = np.array([(n[0],n[1]) for n in self.nodes])
        self._kd = KDTree(xy)
        for i in range(len(self.nodes)):
            self.G.add_node(i)

        for i, qi in enumerate(self.nodes):
            _, idxs = self._kd.query([qi[0],qi[1]], k=min(PRM_K+1,len(self.nodes)))
            for j in map(int, idxs):
                if j==i or self.G.has_edge(i,j): continue
                ok, L = self._edge_ok(qi, self.nodes[j])
                if ok:
                    self.G.add_edge(i, j, weight=L)

        self.build_time = time.perf_counter()-t0
        return self.build_time

    def query(self, start, goal):
        t0 = time.perf_counter()
        if not self.nodes or self._kd is None:
            return [], time.perf_counter()-t0

        _, si = self._kd.query([start[0],start[1]], k=min(8,len(self.nodes)))
        _, gi = self._kd.query([goal[0], goal[1]],  k=min(8,len(self.nodes)))

        best, best_L = None, math.inf
        for s in map(int,si):
            for g in map(int,gi):
                if s==g: continue
                try:
                    p=nx.astar_path(self.G,s,g,
                        heuristic=lambda a,b: math.hypot(
                            self.nodes[a][0]-self.nodes[b][0],
                            self.nodes[a][1]-self.nodes[b][1]),
                        weight='weight')
                    L=sum(self.G[p[k]][p[k+1]]['weight'] for k in range(len(p)-1))
                    if L<best_L: best_L,best=L,p
                except (nx.NetworkXNoPath,nx.NodeNotFound): continue

        if best:
            wpts = [start] + [self.nodes[i] for i in best] + [goal]
        else:
            wpts = []
        return wpts, time.perf_counter()-t0

    def _free(self, x, y):
        r0=int(y/CELL_SIZE); c0=int(x/CELL_SIZE)
        cr=int(math.ceil(TRUCK_L/2/CELL_SIZE))+1
        for dr in range(-cr,cr+1):
            for dc in range(-cr,cr+1):
                nr,nc=r0+dr,c0+dc
                if 0<=nr<GRID_N and 0<=nc<GRID_N and self.obs.grid[nr,nc]:
                    ox=nc*CELL_SIZE+CELL_SIZE/2; oy=nr*CELL_SIZE+CELL_SIZE/2
                    if math.hypot(x-ox,y-oy)<CELL_SIZE*0.9:
                        return False
        return True

    def _edge_ok(self, q1, q2):
        L,sfn=dubins_shortest(q1,q2,TRUCK_MIN_R)
        if L>PRM_MAX_EDGE: return False,math.inf
        for x,y,_ in sfn(max(8,int(L/DUBINS_STEP))):
            if not self._free(x,y): return False,math.inf
        return True,L
