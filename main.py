import matplotlib.pyplot as plt
from config import *
from agents import Wumpus, Firetruck
from sampling_planner import PRMPlanner
import numpy as np

def run_simulation(seed):
    # For now, let's create a dummy grid with a few obstacles so we see action
    grid = np.zeros((GRID_N, GRID_N), dtype=bool)
    grid[10:15, 10:15] = True # A small forest 
    
    obs = ObstacleManager(grid)
    prm = PRMPlanner(obs)
    # prm.build() # Skip build for quick test if PRM is empty
    
    wumpus = Wumpus(obs, (10, 10))
    truck = Firetruck(obs, prm, (50.0, 50.0, 0.5))
    
    # Setup Live Plotting
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots()
    
    sim_time = 0.0
    while sim_time < 200.0: # Shorter time for testing
        wumpus.step(sim_time, DT)
        truck.step(sim_time, DT)
        _, ignited = obs.update(sim_time)
        wumpus.score += len(ignited) # Spread points
        
        # Update Visuals
        ax.clear()
        ax.set_xlim(0, FIELD_SIZE)
        ax.set_ylim(0, FIELD_SIZE)
        ax.scatter(truck.pos[0], truck.pos[1], color='blue', label='Truck')
        wx, wy = obs.cell_xy(*wumpus.cell)
        ax.scatter(wx, wy, color='red', label='Wumpus')
        
        plt.pause(0.01)
        sim_time += DT
        
    plt.ioff()
    plt.show()
    return wumpus.score, truck.score

if __name__ == "__main__":
    run_simulation(42)