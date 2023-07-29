CT0 = 0.3
CT1 = 0.6

# find twist
reference = get_steady_reference(rotor, flow, airfoil, simulation, "CT_steady", "step", [CT0, CT1])

# beta0 = np.interp(CT0, reference['CT'], reference['beta'])
# beta1 = np.interp(CT1, reference['CT'], reference['beta'])

# plt.figure()
# plt.plot( reference["CT"], reference["beta"])
# plt.scatter([CT0, CT1],[beta0, beta1])

beta = np.zeros(np.shape(simulation['time']))+reference[0]  # we initialize the array of thrust coefficient,
# setting all initial values at Ct0

beta[len(beta)//2:] = reference[1]

models = ["Steady", "OYE", "LM", "PP"]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
linestyles = ["dashed", "dotted", "dashdot", "solid"]
for i, m in enumerate(models):
	CT_array = np.zeros(len(simulation['time']))
	a_array = np.zeros(len(simulation['time']))
	simulation['current_index'] = 0
	for j, b in enumerate(beta):
		simulation['model'] = m
		rotor['pitch'] = b
		curr_res = BEM(rotor, airfoil, flow, simulation, results)
		for k, key in enumerate(results):
			results[key][j] = curr_res[k]

	ax1.plot(simulation['time'], results['CT'], color='g', ls=linestyles[i], label=m)
	# print(results['CT'])
	ax2.plot(simulation['time'], np.mean(results['a'], axis=1), color='b', ls=linestyles[i], label=m)

ax1.set_xlabel('Time')
ax1.set_ylabel('CT', color='g')
ax2.set_ylabel('a', color='b')
plt.legend()
plt.show()
