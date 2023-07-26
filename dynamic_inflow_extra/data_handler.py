from dataclasses import dataclass, field
import numpy as np


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
		beta:	            # blade twist in radians (through pre-defined function)
		chord:	            # chord at blade element positions (through pre-defined function)
		sigma:	            # rotor solidity at blade element positions
	"""
	n_r: int = 50
	R: float = 50
	R_root: float = 0.2*50
	B: int = 3
	theta_pitch: float = np.deg2rad(-2)
	airfoil: str = "DU 95-W-180"
	
	r: np.ndarray = field(init=False)
	beta: np.ndarray = field(init=False)
	chord: np.ndarray = field(init=False)
	sigma: np.ndarray = field(init=False)
	
	def __post_init__(self):
		self.r = np.linspace(self.R_root, self.R, self.n_r)
		self.mu = self.r/self.R
		self.beta = np.deg2rad(self._twist_distribution(self.mu))
		self.chord = self._chord_distribution(self.mu)
		self.sigma = self.chord*self.B/(2*np.pi*self.r)
		
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
	
	def __post_init__(self):
		self.time = np.arange(0, self.t_max, self.dt)
	
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
	v_int: np.ndarray = field(init=False)
	
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
		self.v_int = np.zeros((n_time_steps, n_blade_elements))
	
	def __getitem__(self, item):
		return self.__dict__[item]
	
	def __setitem__(self, key, value):
		settable = [param for param in self.__dict__.keys() if not param.startswith("__") and not callable(param)]
		if key not in settable:
			raise ValueError(f"Parameter {key} does not exist for an object of dataclass Results")
		self.__dict__[key] = value
		
	def __iter__(self):
		return iter(tuple([param for param in self.__dict__.keys() if param not in self.skip]))
	
