from BEM import get_steady_reference, steady_reference, calculate_case
import matplotlib.pyplot as plt
from data_handler import Rotor, Flow, Simulation, Results
import numpy as np
import pandas as pd


rotor = Rotor()
flow = Flow(rotor)
airfoil = {'DU 95-W-180': pd.read_excel("../data/input/DU95W180.xlsx", skiprows=3)}
simulation = Simulation(model="Steady", t_max=5.5)

# reference = steady_reference(rotor, airfoil, flow, simulation, ("pitch", np.deg2rad(np.arange(-10, 10))))
# plt.plot(reference["pitch"], reference["CT"])
# plt.plot(reference["pitch"], reference["CP"])
# plt.show()

# steady_results = get_steady_reference(rotor, flow, airfoil, simulation, change_param="CT_steady", change_type="sine",
#                                    change_values=(0.7, 0.3, 1))
# plt.plot(simulation.time, steady_results)
# plt.show()


simulation["dt"] = 0.11
results = calculate_case(rotor, flow, airfoil, simulation, change_param="U_inf", change_type="sine",
                         change_values=(10, 5, 1/5))
fig, ax = plt.subplots(2, 1)
for result in results:
	ax[0].plot(result.simulation.time, result.CT, label=result.simulation.model)
	ax[1].plot(result.simulation.time, np.mean(result.a, axis=1), label=result.simulation.model)
plt.legend()
plt.show()


