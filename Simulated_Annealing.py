import pandapower as pp
import pandapower.plotting as plot
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
from networkx.algorithms.bipartite.basic import color
from scipy.ndimage import label
from copy import deepcopy
from utils import run_time_series
from deap import base, creator, tools, algorithms
import multiprocessing
warnings.simplefilter(action="ignore", category=FutureWarning)

net = pp.create_empty_network()

# create buses
b1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
bus = []
for i in range(17):
    b = pp.create_bus(net, vn_kv=0.4, name="Bus {i}")
    bus.append(b)

pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")

tid = pp.create_transformer(
    net, hv_bus=b1, lv_bus=bus[0], std_type="0.4 MVA 20/0.4 kV", name="Trafo"
)
pp.create_load(net, bus=bus[0], p_mw=0.01, q_mvar=0.005, name="Load")
pp.create_sgen(net, bus=bus[0], p_mw=0.01, q_mvar=0.005, name="Load")

for y in range(1, 8):
    pp.create_line(
        net, from_bus=bus[y - 1], to_bus=bus[y], length_km=0.100, name="Line", std_type="NAYY 4x50 SE"
    )
    pp.create_load(net, bus=bus[y], p_mw=0.01, q_mvar=0.005, name="Load")
    pp.create_sgen(net, bus=bus[y], p_mw=0.01, q_mvar=0.005, name="Load")

pp.create_line(
    net, from_bus=bus[0], to_bus=bus[8], length_km=0.100, name="Line", std_type="NAYY 4x50 SE")
pp.create_load(net, bus=bus[8], p_mw=0.01, q_mvar=0.005, name="Load")
pp.create_sgen(net, bus=bus[8], p_mw=0.01, q_mvar=0.005, name="Load")

for j in range(9, 17):
    pp.create_line(
        net, from_bus=bus[j - 1], to_bus=bus[j], length_km=0.100, name="Line", std_type="NAYY 4x50 SE"
    )
    pp.create_load(net, bus=bus[j], p_mw=0.01, q_mvar=0.005, name="Load")
    pp.create_sgen(net, bus=bus[j], p_mw=0.01, q_mvar=0.005, name="Load")

Load_Profiles = pd.read_csv("LoadData_B.csv", index_col=0)
Gen_Profiles = pd.read_csv("generationData_B.csv", index_col=0)

Best_load_indices = [3, 0, 4, 13, 9, 7, 14, 8, 11, 1, 5, 15, 2, 6, 16, 12, 10]
Best_gen_indices = [7, 8, 4, 12, 1, 9, 13, 3, 10, 5, 15, 11, 2, 0, 16, 14, 6]

# The following function contains the logic for swapping the elements. The swapping is done in a set of two denoted by i and j. The user has the input for selecting how many such swaps should be done per iteration. It was observed that the results were better with low number for num_swaps (below 3). As the self developed algorithm provides a good starting point, minute changes are made to the generator and load indices order to avoid the complete distortion of order.
def generate_neighbor(solution, num_swaps=2):
    new_solution = solution.copy()
    for a in range(num_swaps):
        i, j = random.sample(range(len(solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

# Simulated Annealing works with the principle that it allows exploration at the higher temperature but fine tunes the optimisation once the temperature decreases. This behaviour is controller by the "accept_probability" function. At higher teperature, the accept probabiity is compared with a random value between 0 to 1 to keep the explorative nature intact. Once, the temperature becomes zero, only the better solutions are accepted. This is implemented from line 38.
def accept_probability(max_value, max_value_new, temperature):
    if max_value_new < max_value:
        return 1.0
    return np.exp(np.clip((max_value - max_value_new) / temperature, -709, 709))

#Simulated annealing with initial temperature, cooling rate and number of iterations is initiated here. The current load and generator assignments are the best values obtained from the self developed initializer.
def simulated_annealing(net, initial_temp=3000, cooling_rate=0.9, num_iterations=1000):
    current_load_assignment = Best_load_indices
    current_gen_assignment = Best_gen_indices
    network_copy1 = deepcopy(net)
    _,res_lines_s = run_time_series(Gen_Profiles, Load_Profiles, network_copy1, current_gen_assignment, current_load_assignment,"Iteration_SA")
    max_value = res_lines_s.max(axis=1).sum()
    best_load_assignment = current_load_assignment.copy()
    best_gen_assignment = current_gen_assignment.copy()
    best_energy = max_value
    temperature = initial_temp
    energy_history = [max_value]
    temp_hist = [temperature]


    for i in range(num_iterations):
        # Modify both load and generator assignments
        new_load_assignment = generate_neighbor(current_load_assignment)
        new_gen_assignment = generate_neighbor(current_gen_assignment)
        network_copy2 = deepcopy(net)
        _,res_lines_n = run_time_series(Gen_Profiles, Load_Profiles, network_copy2, new_gen_assignment, new_load_assignment,"New_SA")
        max_value_new = res_lines_n.max(axis=1).sum()

        if accept_probability(max_value, max_value_new, temperature) > random.random():
            current_load_assignment = new_load_assignment
            current_gen_assignment = new_gen_assignment
            max_value = max_value_new

        if max_value < best_energy:
            best_load_assignment = current_load_assignment.copy()
            best_gen_assignment = current_gen_assignment.copy()
            best_energy = max_value
        temperature *= cooling_rate
        energy_history.append(max_value)
        temp_hist.append(temperature)
        #print(f"Current Iteration results")
        #print(f"Iteration {i}")
        #print(f"Temperature: {temperature:.2f}, Sum of max loadings(current): {max_value}")
        #print(f"Current Gen order: {current_gen_assignment}\nCurrent Load Order: {current_load_assignment}")
        #print(f"\nBest found results")
        #print(f"Sum of max loadings(best): {best_energy}")
        #print(f"Best Gen order: {best_gen_assignment}\nBest Load Order: {best_load_assignment}")

    return best_gen_assignment, best_load_assignment, best_energy, energy_history, temp_hist


best_gen_assignment,best_load_assignment,best_energy, energy_history, temp_hist = simulated_annealing(net)

print("Optimization complete.")
print("Best load assignment:", best_load_assignment)
print("Best generator assignment:", best_gen_assignment)
print("Best sum of max loading:", best_energy)


fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(energy_history, linestyle="-", color="blue",label="Energy (Sum of max loading) history")
ax1.set_ylabel("Sum of max loading Convergence During Simulated Annealing")
ax1.set_xlabel("Iteration")
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(temp_hist, color='r', label='Temperature')
ax2.set_ylabel('Temperature', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.grid(True)
plt.title("Energy Convergence During Simulated Annealing with temperature")
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.show()

bin = [78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118]

fig = plt.figure(figsize=(10, 6))
plt.hist(energy_history, bins=bin ,color="blue",label="Sum of max loading spread with Simulated Annealing",alpha=0.5,edgecolor="blue",linewidth=2)
plt.title("Sum of Max loading Histogram")
plt.xlabel("Sum of max loading")
plt.ylabel("Frequency")
plt.grid(True)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.legend()
plt.show()