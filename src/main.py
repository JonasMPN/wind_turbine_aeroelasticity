from BEM import calculate_case
from data_handler import Rotor, Flow, Simulation, Results, plot_case_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def load_or_calculate_and_save(file_path, rotor, flow, airfoil, simulation, change_param, change_type, change_values):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    else:
        results = calculate_case(rotor, flow, airfoil, simulation, change_param, change_type, change_values)
        with open(file_path, 'wb') as file:
            pickle.dump(results, file)

    return results

# change rotor parameters here
rotor = Rotor()
# change flow parameters here
flow = Flow(rotor)
# change airfoil here (Root airfoil: alpha, Cl, Cd, Cm)
airfoil = {'DU 95-W-180': pd.read_excel("../data/input/DU95W180.xlsx", skiprows=3)}


# turn off the print statements of the 'calculate_case' by setting Simulation(verbose=False).

calc_and_plot = {
		"case_A_1": True,
		"case_A_2": True,
		"case_B_1": True,
		"case_B_2": True
}

if calc_and_plot["case_A_1"]:
	print("Working on case A.1")
	simulation = Simulation(model="Steady", t_max=10, dt=0.05)
	change_param = "CT_steady"
	change_type = "step"
	change_values = (0.5, 0.9)
	file_path = "../data/results/pickles/case_A_1.pkl"
	results = load_or_calculate_and_save(file_path, rotor, flow, airfoil, simulation, change_param, change_type, change_values)
	plot_case_results(results, "../data/results", change_param, change_type, change_values)
	

if calc_and_plot["case_A_2"]:
	print("Working on case A.2")
	n_periods = 3
	frequency = 0.3/(2*np.pi*rotor.R)*flow.V0
	simulation = Simulation(model="Steady", t_max=n_periods/frequency, dt=0.4)
	change_param = "CT_steady"
	change_type = "sine"
	change_values = (0.5, 0.5, frequency)
	file_path = "../data/results/pickles/case_A_2.pkl"
	results = load_or_calculate_and_save(file_path, rotor, flow, airfoil, simulation, change_param, change_type, change_values)
	plot_case_results(results, "../data/results", change_param, change_type, change_values)

if calc_and_plot["case_B_1"]:
	print("Working on case B.1")
	simulation = Simulation(model="Steady", t_max=10, dt=0.05)
	change_param = "U_inf"
	change_type = "step"
	change_values = (10, 15)
	file_path = "../data/results/pickles/case_B_1.pkl"
	results = load_or_calculate_and_save(file_path, rotor, flow, airfoil, simulation, change_param, change_type, change_values)
	plot_case_results(results, "../data/results", change_param, change_type, change_values)

if calc_and_plot["case_B_2"]:
	print("Working on case B.2")
	n_periods = 3
	change_param = "U_inf"
	change_type = "sine"
	sine_mean = 20
	sine_amplitude = 10
	omega = 0.3/(2*np.pi*rotor.R)*sine_mean
	change_values = (sine_mean, sine_amplitude, omega)
	simulation = Simulation(model="Steady", t_max=n_periods/omega, dt=0.1)
	file_path = "../data/results/pickles/case_B_2.pkl"
	results = load_or_calculate_and_save(file_path, rotor, flow, airfoil, simulation, change_param, change_type, change_values)
	plot_case_results(results, "../data/results", change_param, change_type, change_values)




