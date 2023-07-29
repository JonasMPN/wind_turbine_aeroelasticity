from BEM import calculate_case
from data_handler import Rotor, Flow, Simulation, Results, plot_case_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# change rotor parameters here
rotor = Rotor()
# change flow parameters here
flow = Flow(rotor)
# change airfoil here (Root airfoil: alpha, Cl, Cd, Cm)
airfoil = {'DU 95-W-180': pd.read_excel("../data/input/DU95W180.xlsx", skiprows=3)}
# change simulation parameters here
simulation = Simulation(model="Steady", t_max=30, dt=0.5)

calc_and_plot = {
		"case_A_1": True,
		"case_A_2": True,
		"case_B_1": True,
		"case_B_2": True
}

if calc_and_plot["case_A_1"]:
	results = calculate_case(rotor, flow, airfoil, simulation,
	                         change_param="CT_steady",
	                         change_type="step",
	                         change_values=(0.5, 0.9))
	fig, ax = plt.subplots(2, 1)
	for result in results:
		ax[0].plot(result.simulation.time, result.CT, label=result.simulation.model)
		ax[1].plot(result.simulation.time, np.mean(result.a, axis=1), label=result.simulation.model)
	plt.legend()
	plt.show()
	

if calc_and_plot["case_A_2"]:
	simulation.reset()
	results = calculate_case(rotor, flow, airfoil, simulation,
	                         change_param="CT_steady",
	                         change_type="sine",
	                         change_values=(0.5, 0.5, 0.3/(2*np.pi*rotor.R)*flow.V0))
	fig, ax = plt.subplots(2, 1)
	for result in results:
		ax[0].plot(result.simulation.time, result.CT, label=result.simulation.model)
		ax[1].plot(result.simulation.time, np.mean(result.a, axis=1), label=result.simulation.model)
	plt.legend()
	plt.show()

if calc_and_plot["case_B_1"]:
	simulation.reset()
	results = calculate_case(rotor, flow, airfoil, simulation,
	                         change_param="U_inf",
	                         change_type="step",
	                         change_values=(1, 1.5))
	fig, ax = plt.subplots(2, 1)
	for result in results:
		ax[0].plot(result.simulation.time, result.CT, label=result.simulation.model)
		ax[1].plot(result.simulation.time, np.mean(result.a, axis=1), label=result.simulation.model)
	plt.legend()
	plt.show()

if calc_and_plot["case_B_1"]:
	simulation.reset()
	results = calculate_case(rotor, flow, airfoil, simulation,
	                         change_param="CT_steady",
	                         change_type="sine",
	                         change_values=(1, 0.5, 0.3/(2*np.pi*rotor.R)*flow.V0))
	fig, ax = plt.subplots(2, 1)
	for result in results:
		ax[0].plot(result.simulation.time, result.CT, label=result.simulation.model)
		ax[1].plot(result.simulation.time, np.mean(result.a, axis=1), label=result.simulation.model)
	plt.legend()
	plt.show()
