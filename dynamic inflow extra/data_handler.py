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
	
	D: float = field(init=False)
	r: np.ndarray = field(init=False)
	beta: np.ndarray = field(init=False)
	chord: np.ndarray = field(init=False)
	sigma: np.ndarray = field(init=False)
	
	def __post_init__(self):
		self.D = 2*self.R
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
	
	@staticmethod
	def _twist_distribution(mu: np.ndarray):
		return 14*(1-mu)
	
	@staticmethod
	def _chord_distribution(mu: np.ndarray):
		return 3*(1-mu)+1

