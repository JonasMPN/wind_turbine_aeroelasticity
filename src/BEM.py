import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from helper_functions import Helper
from data_handling import *
import pandas.errors as pd_error
import scipy
import timeit
helper = Helper()


class BEM:
	def __init__(self,
	             data_root: str,
	             file_airfoil: str):
		self.rotor_radius = None
		self.root_radius = None
		self.n_blades = None
		self.air_density = None
		self.root = data_root
		self.a_prime = 0
		try:
			self.df_results = pd.read_csv(data_root+"/BEM_results.dat")
		except FileNotFoundError or pd_error.EmptyDataError:
			self.df_results = pd.DataFrame()
		
		df_tmp = pd.read_excel(file_airfoil, skiprows=3)
		self.interp = {"c_l": interpolate.interp1d(df_tmp["alpha"], df_tmp["cl"]),
		               "c_d": interpolate.interp1d(df_tmp["alpha"], df_tmp["cd"])}
		self.blade_end_correction = 1
		self.constants = None
	
	def set_constants(self,
	                  rotor_radius: float,
	                  root_radius: float,
	                  n_blades: int,
	                  air_density: float) -> None:
		self.constants = {param: value for param, value in locals().items() if param != "self"}
		self._set(**self.constants)
		return None
	
	def get_results(self, tip_speed_ratio: float, wind_speed: float, pitch: float, rotor_radius = 50,
	                root_radius = 0.2*50, n_blades = 3, air_density = 1.225, tip_loss_correction = True,
	                root_loss_correction = True, resolution = 200, start_radius = 0.2*50):
		if self.df_results.empty:
			print("There are no results in the results file yet.")
		else:
			return self.df_results.query(order_to_query({param: value for param, value in locals().items() if
			                                             param != "self"}))
	
	def solve(self,
	          wind_speed: float,
	          tip_speed_ratio: float,
	          pitch: float or np.ndarray,
	          start_radius: float = None,
	          resolution: int = 200,
	          max_convergence_error: float = 1e-5,
	          max_iterations: int = 200,
	          tip_loss_correction: bool = True,
	          root_loss_correction: bool = True,
	          verbose: bool=False) -> pd.DataFrame:
		"""
		Solves BEM.
		:param wind_speed:
		:param tip_speed_ratio:
		:param pitch:
		:param start_radius:
		:param resolution:
		:param max_convergence_error:
		:param max_iterations:
		:param tip_loss_correction:
		:param root_loss_correction:
		:return:
		"""
		skip = ["skip", "self", "max_convergence_error", "max_iterations", "verbose"]
		start_radius = start_radius if start_radius is not None else self.root_radius
		# Set identifier for results
		use_as_identifier = {param: value for param, value in (self.constants | locals()).items() if param not in skip}
		identifier = {param: np.ones(resolution-1)*value for param, value in use_as_identifier.items()}
		try:
			if exists_already(self.df_results, **use_as_identifier):
				print(f"BEM already done for {use_as_identifier}, skipping solve().")
				return self.get_results(tip_speed_ratio, wind_speed, pitch, self.rotor_radius, self.root_radius,
				                        self.n_blades, self.air_density, tip_loss_correction, root_loss_correction,
				                        resolution, start_radius)
		except Exception as exc:
			if self.df_results.empty:
				pass
			elif exc == KeyError:
				print("You probably added a parameter that does not exist in 'BEM_results.dat'. Go talk to Jonas.")
		# Initialise the result containers. See what the parameters mean in _prepare_results_dict(self, **kwargs)
		params_to_save = ["r_centre", "r_inner", "r_outer", "a", "a_prime", "C_t", "C_n", "C_q", "f_n", "f_t", "C_T",
		                  "C_P", "alpha", "circulation", "end_correction", "C_l", "C_d", "phi", "inflow_speed"]
		self._prepare_results_dict(params_to_save)
		
		pitch = np.deg2rad(pitch)
		# Calculate the rotational speed
		omega = tip_speed_ratio*wind_speed/self.rotor_radius
		radii = np.linspace(start_radius, self.rotor_radius, resolution)
		# Loop along the span of the blade (blade sections)
		print(f"Doing BEM for v0={wind_speed}, tsr={tip_speed_ratio}, pitch={np.rad2deg(pitch)}")
		for r_inner, r_outer in zip(radii[:-1], radii[1:]):  # Take the left and right radius of every element
			if verbose:
				time_start = timeit.default_timer()
			r_centre = (r_inner+r_outer)/2  # representative radius in the middle of the section
			elem_length = r_outer-r_inner  # length of element
			# Get/Set values from the local section
			chord = self._get_chord(r_centre, self.rotor_radius)  # Get the chord
			twist = self._get_twist(r_centre, self.rotor_radius)  # Get the twist
			area_annulus = np.pi*(r_outer ** 2-r_inner ** 2)
			a, a_new, a_prime, converged = 1/3, 0, 0, False
			for i in range(max_iterations):
				# get inflow angle, and inflow speed for airfoil
				phi, inflow_speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega,
				                               radius=r_centre)
				# get combined lift and drag coefficient projected into the normal and tangential direction
				_, _, _, c_n, c_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch,
				                                             tip_seed_ratio=tip_speed_ratio, university="tud")
				# get thrust force (in N) of the whole turbine at the current radius
				thrust = self._aero_force(inflow_speed, chord, c_n)*self.n_blades*elem_length
				# calculate thrust coefficient that results from the blade element
				C_T = thrust/(1/2*self.air_density*wind_speed ** 2*area_annulus)
				# get Glauert corrected axial induction factor
				a_new = self._a(C_T=C_T)
				# get the combined (tip and root) correction factor
				end_correction = self._blade_end_correction(tip=tip_loss_correction, root=root_loss_correction,
				                                            radius=r_centre, tip_seed_ratio=tip_speed_ratio, a=a_new)
				# correct the Glauert corrected axial induction factor with the blade end losses
				a_new /= end_correction
				# update the axial induction factor for the next iteration
				a = 0.75*a+0.25*a_new
				# get the tangential force (in N/m) of the whole turbine at the current radius
				f_tangential = self._aero_force(inflow_speed, chord, c_t)*self.n_blades
				# get the tangential induction factor that corresponds to this force AND correct it for tip losses
				a_prime = self._a_prime(f_tangential, r_centre, wind_speed, a, tip_speed_ratio)/end_correction
				# check if the axial induction factor has converged. If it has, the tangential induction factor has too
				if np.abs(a-a_new) < max_convergence_error:
					converged = True
					break
			# notify user if loop did not converge, but was stopped by the maximum number of iterations
			if not converged:
				print(f"BEM did not converge for the blade element between {r_inner}m and {r_outer}m. Current "
				      f"change after {max_iterations}: {np.abs(a-a_new)}.")
			# Now that we have the converged axial induction factor, we can get the rest of the values
			phi, inflow_speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega,
			                               radius=r_centre)
			alpha, C_l, C_d, C_n, C_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch, radius=r_centre,
			                                                     tip_seed_ratio=tip_speed_ratio, university="tud")
			
			results_element = {param: value for param, value in locals().items() if param in params_to_save}
			results_element["C_q"] = self._C_q(a, a_prime, tip_speed_ratio, r_centre, self.rotor_radius)
			results_element["f_n"] = self._aero_force(inflow_velocity=inflow_speed, chord=chord, force_coefficient=C_n)
			results_element["f_t"] = self._aero_force(inflow_velocity=inflow_speed, chord=chord, force_coefficient=C_t)
			results_element["C_T"] = self._C_T(a)
			results_element["C_P"] = self._C_P(a)
			results_element["circulation"] = 1/2*inflow_speed*C_l*chord
			self._update_results(**results_element)
			if verbose:
				print(f"BEM for element at r={r_centre} took {np.round(timeit.default_timer()-time_start, 3)}s.")
			
		self.df_results = pd.concat([self.df_results, pd.DataFrame(identifier | self.results)])
		self.df_results.to_csv(self.root+"/BEM_results.dat", index=False)
		return pd.DataFrame(self.results)
	
	def _calculate_thrust(self, f_n, radial_positions):
		"""
			Calculate thrust from the normal forces. Account for f_t = 0 at the tip.
		f_n: normal forces
		radial_positions: radial position along the blade matching the positions of f_n
		n_blades:   number of blades
		radius:     max radius
		"""
		thrust = self.n_blades*scipy.integrate.simpson([*f_n, 0], [*radial_positions, self.rotor_radius])
		return thrust
	
	def _calculate_power(self, f_t, radial_positions, omega):
		"""
			Calculate power from the normal forces. Account for f_n = 0 at the tip.
		f_t: tangential forces
		radial_positions: radial position along the blade matching the positions of f_n
		n_blades:   number of blades
		radius:     max radius
		omega:      [rad/s] rotational speed
		"""
		power = omega*self.n_blades*scipy.integrate.simpson(
				np.multiply([*radial_positions, self.rotor_radius], [*f_t, 0]),
				[*radial_positions, self.rotor_radius])
		return power
	
	def _calc_ct(self, thrust, velocity):
		"""
			Calculate the thrust coefficient ct
		"""
		return thrust/(0.5*np.pi*(self.rotor_radius ** 2)*self.air_density*(velocity ** 2))
	
	def _calc_cp(self, power, velocity):
		"""
			Calculate the power coefficient ct
		"""
		return power/(0.5*np.pi*(self.rotor_radius ** 2)*self.air_density*(velocity ** 3))
	
	def _calc_ct_distribution(self, f_n, velocity):
		"""
		Calculate the distribution of the thrust coefficient along the blade
		f_n: normal forces along the blade
		radius: maximum radius of the Blade
		velocity: fluid velocity od V0
		"""
		f_n = np.array(f_n)
		return f_n/(0.5*np.pi*(self.rotor_radius ** 2)*self.air_density*(velocity ** 3))
	
	def _calc_cp_distribution(self, f_t, velocity):
		"""
		Calc the distribution of the power coeff. along the blade
		f_t: tangential forces along the blade
		radius: maximum radius of the Blade
		velocity: fluid velocity od V0
		"""
		f_t = np.array(f_t)
		return f_t/(0.5*np.pi*(self.rotor_radius ** 2)*self.air_density*(velocity ** 3))
	
	def _phi_to_aero_values(self, phi: float, twist: float or np.ndarray, pitch: float, tip_seed_ratio: float,
	                        university: str,
	                        radius: float = None, a: float = None, blade_end_correction_type: str = None,
	                        tip: bool = None,
	                        root: bool = None) -> tuple:
		alpha = np.rad2deg(phi-(twist+pitch))
		c_l = self.interp["c_l"](alpha)
		c_d = self.interp["c_d"](alpha)
		c_n = self._c_normal(phi, c_l, c_d)
		c_t = self._c_tangent(phi, c_l, c_d)
		if university == "tud":
			return alpha, c_l, c_d, c_n, c_t
		elif university == "dtu":
			return alpha, c_l, c_d, c_n, c_t, self._blade_end_correction(which=blade_end_correction_type, tip=tip,
			                                                             root=root, phi=phi, radius=radius, a=a,
			                                                             tip_seed_ratio=tip_seed_ratio)
	
	def _set(self, **kwargs) -> None:
		"""
		Sets parameters of the instance. Raises an error if a parameter is trying to be set that doesn't exist.
		:param kwargs:
		:return:
		"""
		existing_parameters = [*self.__dict__]
		for parameter, value in kwargs.items():
			if parameter not in existing_parameters:
				raise ValueError(f"Parameter {parameter} cannot be set. Settable parameters are {existing_parameters}.")
			self.__dict__[parameter] = value
		return None
	
	def _assert_values(self):
		not_set = list()
		for variable, value in vars(self).items():
			if value is None:
				not_set.append(variable)
		if len(not_set) != 0:
			raise ValueError(f"Variable(s) {not_set} not set. Set all variables before use.")
	
	def _update_a_prime(self, local_solidity: float, c_tangential: float, blade_end_correction: float,
	                    inflow_angle: float) -> None:
		self.a_prime = local_solidity*c_tangential*(1+self.a_prime)/(4*blade_end_correction*np.sin(inflow_angle)*
		                                                             np.cos(inflow_angle))
	
	def _aero_force(self, inflow_velocity: float, chord: float, force_coefficient: float):
		"""
		Calculates the tangential force per unit span.
		:param inflow_velocity:
		:param chord:
		:param c_normal:
		:return:
		"""
		return 1/2*self.air_density*inflow_velocity ** 2*chord*force_coefficient
	
	def _a_prime(self, F_tangential: float, radius: float, wind_speed: float, a: float, tip_speed_ratio: float):
		return F_tangential/(
				4*self.air_density*np.pi*radius ** 2/self.rotor_radius*wind_speed ** 2*(1-a)*tip_speed_ratio)
	
	def _blade_end_correction(self, radius: float, tip_seed_ratio: float, a: float, phi: float = None,
	                          tip: bool = True, root: bool = True) -> float:
		"""
		Different Prandtl correction methods.
		:param which:
		:param tip:
		:param root:
		:param phi:
		:param radius:
		:param tip_seed_ratio:
		:param a:
		:return:
		"""
		F = 1
		if tip:
			d = 2*np.pi/self.n_blades*(1-a)/(np.sqrt(tip_seed_ratio ** 2+(1-a) ** 2))
			F = 2/np.pi*np.arccos(np.exp(-np.pi*((self.rotor_radius-radius)/self.rotor_radius)/d))
		if root:
			d = 2*np.pi/self.n_blades*(1-a)/(np.sqrt(tip_seed_ratio ** 2+(1-a) ** 2))
			F *= 2/np.pi*np.arccos(np.exp(-np.pi*((radius-self.root_radius)/self.rotor_radius)/d))
		return F
	
	def _prepare_results_dict(self, parameters: list):
		"""
		"r_centre"      : radius used for the calculations
		"r_inner"       : inner radius of the blade element
		"r_outer"       : outer radius of the blade element
		"a"             : Axial Induction factor
		"a_prime"       : Tangential induction factor
		"C_t"           : tangential aerodynamic coefficient
		"C_n"           : normal aerodynamic coefficient
		"C_q"           : torque coefficient
		"f_n"           : Forces normal to the rotor plane in N/m
		"f_t"           : Forces tangential in the rotor plane in N/m
		"C_T"           : thrust coefficient
		"C_P"           : power coefficient
		"alpha"         : angle of attack
		"circulation"   : magnitude of the circulation using Kutta-Joukowski
		"end_correction": blade end correction (depending on 'tip' and 'root'),
		"c_l"           : lift coefficient at radial position
		"c_d"           : drag coefficient at radial position
		"phi"           : inflow angle
		"inflow_speed"  : inflow speed for airfoil
		:param parameters:
		:return:
		"""
		self.results = {parameter: list() for parameter in parameters}
		return None
	
	def _update_results(self, **kwargs):
		for key, value in kwargs.items():
			self.results[key].append(value)
		return None
	
	@staticmethod
	def _a(C_T: float) -> float:
		C_T1 = 1.816
		CT_2 = 2*np.sqrt(C_T1)-C_T1
		if C_T < CT_2:
			return 1/2-np.sqrt(1-C_T)/2
		else:
			return 1+(C_T-C_T1)/(4*np.sqrt(C_T1)-4)
	
	@staticmethod
	def _C_T(a: float):
		C_T1 = 1.816
		if a < 1-np.sqrt(C_T1)/2:
			return 4*a*(1-a)
		else:
			return C_T1-4*(np.sqrt(C_T1)-1)*(1-a)
	
	@staticmethod
	def _C_P(a: float):
		return 4*a*(1-a) ** 2
	
	@staticmethod
	def _c_normal(phi: float, c_lift: float, c_drag: float) -> float:
		"""
		Calculates an aerodynamic "lift" coefficient according to a coordinate transformation with phi
		:param phi: angle between flow and rotational direction in rad
		:param c_lift: lift coefficient old coordinate system
		:param c_drag: lift coefficient old coordinate system
		:return: Normal force in Newton
		"""
		return c_lift*np.cos(phi)+c_drag*np.sin(phi)
	
	@staticmethod
	def _c_tangent(phi: float, c_lift: float, c_drag: float) -> float:
		"""
		Calculates an aerodynamic "drag" coefficient according to a coordinate transformation with phi
		:param phi: angle between flow and rotational direction in rad
		:param c_lift: lift coefficient old coordinate system
		:param c_drag: lift coefficient old coordinate system
		:return: Normal force in Newton
		"""
		return c_lift*np.sin(phi)-c_drag*np.cos(phi)
	
	@staticmethod
	def _C_q(a: float, a_prime: float, tsr: float, radial_position: float, rotor_radius: float):
		return 4*a_prime*(1-a)*tsr*radial_position/rotor_radius
	
	@staticmethod
	def _local_solidity(chord: float, radius: float, n_blades: int) -> float:
		"""
		Calculates the local solidity
		:param chord: in m
		:param radius: distance from rotor axis to blade element in m
		:param n_blades: number of blades
		:return: local solidity
		"""
		return n_blades*chord/(2*np.pi*radius)
	
	@staticmethod
	def _flow(a: float, a_prime: float, wind_speed: float, rotational_speed: float, radius: float) -> [float, float]:
		"""
		Function to calculate the inflow angle based on the two induction factors, the inflow velocity, radius and
		angular_velocity
		:param a:
		:param a_prime:
		:return:
		"""
		u_axial = wind_speed*(1-a)
		u_tangential = rotational_speed*radius*(1+a_prime)
		return np.tan(u_axial/u_tangential), np.sqrt(u_axial ** 2+u_tangential ** 2)
	
	@staticmethod
	def _get_twist(r, r_max):
		"""
			function to get the twist along the blade in radians
			r: radial position along the blade in [m]
			r_max: maximum radius of the blade in [m]
			out: twist in radians
		"""
		return np.radians(14*(1-r/r_max))
	
	@staticmethod
	def _get_chord(r, r_max):
		"""
			function to calculate chord along the span in m
			r: radial position along the blade in [m]
			r_max: maximum radius of the blade in [m]
		"""
		return 3*(1-r/r_max)+1
	
	@staticmethod
	def _tangential_induction_factor(phi: float, local_solidity: float, c_tangent: float, tip_loss_correction: float) \
			-> float:
		return 1/((4*tip_loss_correction*np.sin(phi)*np.cos(phi))/(local_solidity*c_tangent)-1)
