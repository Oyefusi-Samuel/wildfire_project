import matplotlib.pyplot as plt
import numpy as np
from config import *
from agents import Wumpus, Firetruck

def run_simulation(seed):
    grid = generate_grid(seed)
    obs = ObstacleManager(grid)
    viz = Visualizer(obs) # The advanced visualizer from previous step
    
    wumpus = Wumpus(obs, (2, 2))
    truck = Firetruck(obs, None, (225.0, 225.0, math.pi))
    
    sim_time = 0.0
    plt.ion() # Interaction ON
    
    while sim_time < SIM_DURATION:
        # 1. Update Agents
        wumpus.step(sim_time, DT)
        truck.step(sim_time, DT)
        
        # 2. Update Fire (Corrected unpacking)
        newly_ignited = obs.update(sim_time)
        
        # 3. Calculate scores for display
        w_score = sum(1 for s in obs.state.values() if s in [State.BURNING, State.BURNED])
        t_score = sum(1 for s in obs.state.values() if s == State.EXTINGUISHED)
        
        # 4. Render Advanced Visuals
        viz.render(truck.pos, wumpus.cell, sim_time, w_score, t_score)
        
        sim_time += DT
        if not plt.fignum_exists(viz.fig.number): break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation(42)