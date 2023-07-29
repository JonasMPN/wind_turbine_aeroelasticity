from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from helper_functions import Helper
helper = Helper()


@dataclass
class Rotor:
	"""
	settable parameters:
		n_r:                # number of blade elements
		R:	                # radius
		R_root:             # radius at which the innermost blade element is positioned
		B:	                # n blades
		theta_pitch:	    # pitch in radians
		airfoil:	        # name of the airfoil
	deduced parameters:
		D:  	            # diameter
		r:	                # radial blade element positions
		twist:	            # blade twist in radians (through pre-defined function)
		chord:	            # chord at blade element positions (through pre-defined function)
		sigma:	            # rotor solidity at blade element positions
	"""
	n_r: int = 50
	R: float = 50
	R_root: float = 0.2*50
	B: int = 3
	pitch: float = np.deg2rad(-2)
	airfoil: str = "DU 95-W-180"
	
	r: np.ndarray = field(init=False)
	twist: np.ndarray = field(init=False)
	chord: np.ndarray = field(init=False)
	sigma: np.ndarray = field(init=False)
	
	def __post_init__(self):
		self.r = np.linspace(self.R_root, self.R, self.n_r)
		self.mu = self.r/self.R
		self.twist = np.deg2rad(self._twist_distribution(self.mu))
		self.chord = self._chord_distribution(self.mu)
		self.sigma = self.chord*self.B/(2*np.pi*self.r)
		
	def settable(self):
		return copy([param for param in self.__dict__.keys() if not param.startswith("__") and not callable(param)])
	
	def __getitem__(self, item):
		return self.__dict__[item]
	
	def __setitem__(self, key, value):
		settable = [param for param in self.__dict__.keys() if not param.startswith("__") and not callable(param)]
		if key not in settable:
			raise ValueError(f"Parameter {key} does not exist for an object of dataclass Rotor")
		self.__dict__[key] = value
		has_dependencies = ["n_r", "R", "R_root", "B"]
		if key in has_dependencies:
			self.__post_init__()
	
	@staticmethod
	def _twist_distribution(mu: np.ndarray):
		return 14*(1-mu)
	
	@staticmethod
	def _chord_distribution(mu: np.ndarray):
		return 3*(1-mu)+1


@dataclass
class Flow:
	rotor: Rotor
	V0: float = 10
	rho: float = 1.225
	tsr: float = 10
	
	omega: float = field(init=False)
	
	def __post_init__(self):
		self.omega = self.tsr*self.V0/self.rotor.R
	
	def settable(self):
		return copy([param for param in self.__dict__.keys() if not param.startswith("__") and not callable(param)])
	
	def __getitem__(self, item):
		return self.__dict__[item]
	
	def __setitem__(self, key, value):
		settable = [param for param in self.__dict__.keys() if not param.startswith("__") and not callable(param)]
		if key not in settable:
			raise ValueError(f"Parameter {key} does not exist for an object of dataclass Flow")
		self.__dict__[key] = value


@dataclass
class Simulation:
	model: str
	error: float = 1e-4
	dt: float = 0.1
	current_index: int = 0
	t_max: float = 30
	
	time: np.ndarray = field(init=False)
	actual_t_max: np.float = field(init=False)
	
	def __post_init__(self):
		self.time = np.arange(0, self.t_max, self.dt)
		self.actual_t_max = self.time[-1]
	
	def reset(self):
		self.current_index = 0
	
	def __getitem__(self, item):
		return self.__dict__[item]
	
	def __setitem__(self, key, value):
		settable = [param for param in self.__dict__.keys() if not param.startswith("__") and not callable(param)]
		if key not in settable:
			raise ValueError(f"Parameter {key} does not exist for an object of dataclass Simulation")
		self.__dict__[key] = value
		has_dependencies = ["dt", "t_max"]
		if key in has_dependencies:
			self.__post_init__()
	

@dataclass
class Results:
	rotor: Rotor
	simulation: Simulation
	
	skip: tuple = field(init=False)
	P: np.ndarray = field(init=False)
	T: np.ndarray = field(init=False)
	CP: np.ndarray = field(init=False)
	CT: np.ndarray = field(init=False)
	a: np.ndarray = field(init=False)
	ap: np.ndarray = field(init=False)
	f: np.ndarray = field(init=False)
	Ct: np.ndarray = field(init=False)
	Cq: np.ndarray = field(init=False)
	v_int: np.ndarray = field(init=False)
	alpha: np.ndarray = field(init=False)
	phi: np.ndarray = field(init=False)
	
	def __post_init__(self):
		self.skip = ("skip", "rotor", "simulation")
		n_blade_elements = len(self.rotor["r"])
		n_time_steps = len(self.simulation["time"])
		
		self.P = np.zeros(n_time_steps)
		self.T = np.zeros(n_time_steps)
		self.CP = np.zeros(n_time_steps)
		self.CT = np.zeros(n_time_steps)
		self.a = np.zeros((n_time_steps, n_blade_elements))
		self.ap = np.zeros((n_time_steps, n_blade_elements))
		self.f = np.zeros((n_time_steps, n_blade_elements))
		self.Ct = np.zeros((n_time_steps, n_blade_elements))
		self.Cq = np.zeros((n_time_steps, n_blade_elements))
		self.v_int = np.zeros((n_time_steps, n_blade_elements))
		self.alpha = np.zeros((n_time_steps, n_blade_elements))
		self.phi = np.zeros((n_time_steps, n_blade_elements))
		
	def __getitem__(self, item):
		return self.__dict__[item]
	
	def __setitem__(self, key, value):
		settable = [param for param in self.__dict__.keys() if not param.startswith("__") and not callable(param)]
		if key not in settable:
			raise ValueError(f"Parameter {key} does not exist for an object of dataclass Results")
		self.__dict__[key] = value
		
	def __iter__(self):
		return iter(tuple([param for param in self.__dict__.keys() if param not in self.skip]))
	

def plot_case_results(results: list[Results], save_dir: str, change_param: str, change_type: str, change_values: tuple):
	helper.create_dir(save_dir)
	fig_Ct, ax_Ct = plt.subplots()
	fig_Cq, ax_Cq = plt.subplots()
	fig_a, ax_a = plt.subplots()
	fig_ap, ax_ap = plt.subplots()
	fig_alpha, ax_alpha = plt.subplots()
	fig_phi, ax_phi = plt.subplots()
	
	n_elements = results[0].rotor.n_r
	i_middle_element = np.floor(n_elements/2)
	i_elements = np.array([0, i_middle_element, n_elements])
	
	for result in results:
		for i_element in i_elements:
			ax_Ct.plot(result.simulation.time, result.Ct[:, i_element], label=result.simulation.model)
			ax_Cq.plot(result.simulation.time, result.Cq[:, i_element], label=result.simulation.model)
			ax_a.plot(result.simulation.time, result.a[:, i_element], label=result.simulation.model)
			ax_ap.plot(result.simulation.time, result.ap[:, i_element], label=result.simulation.model)
			ax_alpha.plot(result.simulation.time, result.alpha[:, i_element], label=result.simulation.model)
			ax_phi.plot(result.simulation.time, result.phi[:, i_element], label=result.simulation.model)
			
	axes = [ax_Ct, ax_Cq, ax_a, ax_ap, ax_alpha, ax_phi]
	figs = [fig_Ct, fig_Cq, fig_a, fig_ap, fig_alpha, fig_phi]
	params = ["Ct", "Cq", "a", "ap", "alpha", "phi"]
	file_name_base = change_param+"_"+change_type+"_"+str(change_values)+".png"
	x_label = "time (s)"
	y_labels = ["local thrust coefficient (-)",
	           "local torque coefficient (-)",
	           "local axial induction (-)",
	           "local tangential induction (-)",
	           "local angle of attack (°)",
	           "local inflow angle (°)"]
	
	for param, fig, ax, y_label in zip(params, figs, axes, y_labels):
		file = save_dir + f"/{param}_{file_name_base}"
		helper.handle_axis(ax, x_label=x_label, y_label=y_label)
		helper.handle_figure(fig, save_to=file)
	
	
	
	
	
	
