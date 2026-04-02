import time
from config import *
from config import ObstacleManager
from agents import Wumpus, Firetruck
from sampling_planner import PRMPlanner

def run_simulation(seed):
    # Initialize grid and managers
    grid = np.zeros((GRID_N, GRID_N), dtype=bool) # (Populate with tetrominoes)
    obs = ObstacleManager(grid)
    prm = PRMPlanner(obs)
    prm.build()
    
    wumpus = Wumpus(obs, (2, 25))
    truck = Firetruck(obs, prm, (225.0, 125.0, 0.0))
    
    sim_time = 0.0
    while sim_time < SIM_DURATION:
        wumpus.step(sim_time, DT)
        truck.step(sim_time, DT)
        obs.update(sim_time)
        sim_time += DT
    
    return wumpus.score, truck.score

if __name__ == "__main__":
    seeds = [42, 137, 256, 512, 999]
    for s in seeds:
        w_score, t_score = run_simulation(s)
        print(f"Seed {s}: Wumpus {w_score} | Truck {t_score}")