import numpy as np


def C_T_to_a(C_T):
	"""
	This function calculates the induction factor 'a' as a function of thrust coefficient C_T
	:param C_T: thrust coefficient
	:return:
	"""
	C_T_1 = 1.816
	C_T_2 = 2*np.sqrt(C_T_1)-C_T_1
	
	a = np.zeros(np.shape(C_T))
	a[C_T >= C_T_2] = 1+(C_T[C_T >= C_T_2]-C_T_1)/(4*(np.sqrt(C_T_1)-1))
	a[C_T < C_T_2] = 0.5-0.5*np.sqrt(1-C_T[C_T < C_T_2])
	return a


def a_to_C_T(a):
	"""
	This function calculates the thrust coefficient as a function of induction factor 'a'.
	:param a: induction factor
	:return:
	"""
	C_T = 4*a*(1-a)
	C_T_1 = 1.816
	a1 = 1-np.sqrt(C_T_1)/2
	C_T[a > a1] = C_T_1-4*(np.sqrt(C_T_1)-1)*(1-a[a > a1])
	return C_T


def pitt_peters(C_t, u_rotor, u_inf, R, dt):
	"""
	Applies the Pitt-Peters model.
	:param C_t: Thrust coefficient on the actuator, vind is the induced velocity,
	:param u_rotor: current velocity at the annulus
	:param u_inf: unperturbed velocity
	:param R: radius of the annulus
	:param dt: time step duration
	:return:
	"""
	a = -u_rotor/u_inf  # determine the induction coefficient for the time step {i-1}
	C_T_old = -a_to_C_T(a)  # calculate the thrust coefficient from the induction for the time step {i-1}
	
	dvind_dt = (C_t-C_T_old)/(16/(3*np.pi))*(u_inf ** 2/R)  # calculate the time derivative of the induction velocity
	vind_new = u_rotor+dvind_dt*dt  # calculate the induction at time {i} by time integration
	return vind_new, dvind_dt


def oye_dynamic_inflow(u_rotor, C_T_now, C_T_qs, vint, u_inf, R, r, dt,):
	# this function determines the time derivative of the induction at the annulli
	# using the Øye dynamic inflow model
	# Ct is the thrust coefficient on the actuator, vind is the induced velocity,
	# Uinf is the unperturbed velocity and R is the radial scale of the flow,
	# r is the radial position of the annulus. vqs is the quasi-steady value from BEM,
	# vint is an intermediate value and vz is the induced velocity
	
	# calculate quasi-steady induction velocity
	vqst1 = -C_T_to_a(-C_T_now)*u_inf
	
	# calculate current induction factor
	a = -u_rotor/u_inf
	
	# calculate time scales of the model
	t1 = 1.1/(1-1.3*a)*R/u_inf
	t2 = (0.39-0.26*(r/R) ** 2)*t1
	
	# calculate next-time-step quasi-steady induction velocity
	vqst2 = -C_T_to_a(-C_T_qs)*u_inf
	
	# calculate time derivative of intermediate velocity
	dvint_dt = (vqst1+(vqst2-vqst1)/dt*0.6*t1-vint)/t1
	
	# calculate new intermediate velocity
	vint2 = vint+dvint_dt*dt
	
	# calculate time derivaive of the induced velocity
	dvz_dt = ((vint+vint2)/2-u_rotor)/t2
	
	# calculate new induced velocity
	vz2 = u_rotor+dvz_dt*dt
	return vz2, vint2
