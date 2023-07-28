from BEM import solve_condition, steady_reference
import matplotlib.pyplot as plt
from data_handler import Rotor, Flow, Simulation, Results
import numpy as np
import pandas as pd


rotor = Rotor()
flow = Flow(rotor)
airfoil = {'DU 95-W-180': pd.read_excel("../data/input/DU95W180.xlsx", skiprows=3)}
simulation = Simulation(model="Steady")

reference = steady_reference(rotor, airfoil, flow, simulation, ("pitch", np.deg2rad(np.arange(-10, 10))))
plt.plot(reference["pitch"], reference["CT"])
plt.plot(reference["pitch"], reference["CP"])
plt.show()






# # change rotor parameters here
# rotor = Rotor()
# # change flow parameters here
# flow = Flow(rotor)
# # change airfoil here (Root airfoil: alpha, Cl, Cd, Cm)
# airfoil = {'DU 95-W-180': pd.read_excel('DU95W180.xlsx', skiprows=3)}
# # change simulation parameters here
# simulation = Simulation(model="Steady")
# # result container does not need changes in the initialisation
# results = Results(rotor, simulation)
#
# CT0 = 0.3
# CT1 = 0.6




#
# # find twist
# reference = generate_reference_curve(rotor, airfoil, flow, simulation, results)
#
# f = interpolate.interp1d(reference['CT'], reference['beta'])
# beta0, beta1 = f([CT0, CT1])
#
# # beta0 = np.interp(CT0, reference['CT'], reference['beta'])
# # beta1 = np.interp(CT1, reference['CT'], reference['beta'])
#
# # plt.figure()
# # plt.plot( reference["CT"], reference["beta"])
# # plt.scatter([CT0, CT1],[beta0, beta1])
#
#
# beta = np.zeros(np.shape(simulation['time']))+beta0  # we initialize the array of thrust coefficient,
# # setting all initial values at Ct0
#
# beta[len(beta)//2:] = beta1
#
# models = ["Steady", "OYE", "LM", "PP"]
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# linestyles = ["dashed", "dotted", "dashdot", "solid"]
# for i, m in enumerate(models):
# 	CT_array = np.zeros(len(simulation['time']))
# 	a_array = np.zeros(len(simulation['time']))
# 	simulation['current_index'] = 0
# 	for j, b in enumerate(beta):
# 		simulation['model'] = m
# 		rotor['theta_pitch'] = b
# 		curr_res = function_BEM(rotor, airfoil, flow, simulation, results)
# 		for k, key in enumerate(results):
# 			results[key][j] = curr_res[k]
#
# 	ax1.plot(simulation['time'], results['CT'], color='g', ls=linestyles[i], label=m)
# 	# print(results['CT'])
# 	ax2.plot(simulation['time'], np.mean(results['a'], axis=1), color='b', ls=linestyles[i], label=m)
#
# ax1.set_xlabel('Time')
# ax1.set_ylabel('CT', color='g')
# ax2.set_ylabel('a', color='b')
# plt.legend()
# plt.show()
