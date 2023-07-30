import numpy as np
from scipy import optimize
from data_handler import Rotor, Flow, Simulation, Results
from copy import copy
import timeit


def CTfunction(a, glauert = False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    CT = np.zeros(np.shape(a))
    CT = 4*a*(1-a)
    if glauert:
        CT1=1.816
        a1=1-np.sqrt(CT1)/2
        CT[a>a1] = CT1-4*(np.sqrt(CT1)-1)*(1-a[a>a1])
    return CT


def ainduction(CT,glauert=False):
    """
    This function calculates the induction factor 'a' as a function of thrust coefficient CT 
    including Glauert's correction. Notice that here we are using the wind turbine notation of 'a' 
    """
    if glauert:
        CT1=1.816
        CT2=2*np.sqrt(CT1)-CT1
    else:
        CT1=0
        CT2=100
    
    a=np.zeros(np.shape(CT))
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a


def pitt_peters(Ct,a,Uinf,R,dt,glauert=True ):
    
    # this function determines the time derivative of the induction at the annulli
    # Ct is the thrust coefficient on the actuator, vind is the induced velocity,
    # Uinf is the unperturbed velocity and R is the radial scale of the flow, nd dt is the time step
    # it returns the new value of induction vindnew and the time derivative dvind_dt
    # glauert defines if glauert's high load correction is applied
    
    # a=-vind/Uinf # determine the induction coefficient for the time step {i-1}
    # Ctn= -CTfunction(a, glauert) # calculate the thrust coefficient from the induction for the time step {i-1}

    # dvind_dt =  (Ct-Ctn )/(16/(3*np.pi))*(Uinf**2/R) # calculate the time derivative of the induction velocity
    # a_new = vind + dvind_dt*dt # calculate the induction at time {i} by time integration
    Ctn= CTfunction(a, glauert)
    dvdt = 3*np.pi*Uinf**2/16/R*(Ct - Ctn)
    v = -a*Uinf - dvdt*dt
    a_new = -v/Uinf
    return a_new


def larsen_madsen(CT, a,  V0, r, dt, glauert=True):
    # this function determines the time derivative of the induction at the annulli
    # using the Larsen-Madsen dynamic inflow model
    # Ct2 is the thrust coefficient on the actuator at the next time step,
    # vind is the induced velocity,
    # Uinf is the unperturbed velocity and R is the radial scale of the flow,
    # R is the radius. vqst2 is the quasi-steady value from momentum theory,
    
    vz = -a * V0 #calculate induced velocity from a(i)
    tau = 0.5*r/(V0 + vz)
    a_qs = ainduction(CT, glauert) #calculate a_qs from CT(i+1)
    a_new = a*np.exp(-dt/tau) + a_qs*(1-np.exp(-dt/tau))
    
    return a_new


def oye(a, Ct1, Ct2, vint, V0, R, r,dt,glauert=False):
    # this function determines the time derivative of the induction at the annulli
    # using the Øye dynamic inflow model
    # Ct is the thrust coefficient on the actuator, vind is the induced velocity,
    # Uinf is the unperturbed velocity and R is the radial scale of the flow,
    # r is the radial position of the annulus. vqs is the quasi-steady value from BEM,
    #vint is an intermediate value and vz is the induced velocity
    
    # calculate  quasi-steady induction velocity
    vqst1=-ainduction(Ct1, glauert=True)*V0

    # calculate current induced velocity
    vz = -a * V0

    # calculate time scales of the model
    t1 = 1.1/(1-1.3*a)*R/V0
    t2 = (0.39-0.26*(r/R)**2)*t1

    # calculate next-time-step quasi-steady induction velocity
    vqst2=-ainduction(Ct2, glauert=True)*V0
    
    #calculate time derivative of intermediate velocity
    dvint_dt= (vqst1+ (vqst2-vqst1)/dt*0.6*t1 - vint)/t1

    # calculate new intermediate velocity
    vint2 = vint +dvint_dt*dt
    
    #calculate time derivaive of the induced velocity
    dvz_dt = ((vint+vint2)/2-vz)/t2
    
    #calculate new induced velocity
    vz2 = vz +dvz_dt*dt
    
    # calculate new induction factor
    a_new = -vz2 / V0
    return a_new, vint2


def BEM(rotor: Rotor, airfoil: dict, flow: Flow, simulation: Simulation, results: Results, BEM_max_iterations: int=200):
    # Pre-allocate variables
    a_new = np.full(len(rotor['r']), 0.3)
    ap_new = np.zeros(len(rotor['r']))
    a_old = np.zeros(len(rotor['r']))
    ap_old = np.zeros(len(rotor['r']))
    v_int = 0
    BEM_iteration = 0
    while np.sum(np.abs(a_new-a_old)) > simulation['error'] or np.sum(np.abs(ap_new-ap_old)) > simulation['error']:
        BEM_iteration += 1
        if BEM_iteration > BEM_max_iterations:
            exc = f"BEM did not converge (current errors: a {np.sum(np.abs(a_new-a_old))}, ap " \
                  f"{np.sum(np.abs(ap_new-ap_old))} after {BEM_max_iterations} iterations.)"
            raise Exception(exc)

        # Update induction
        a_old = a_new
        ap_old = ap_new

        # Velocity component
        Vn = (1-a_new)*flow['V0']
        Vt = (1+ap_new)*flow['omega']*rotor['r']
        Vrel = np.sqrt(Vn**2+Vt**2)

        # Inflow angles
        Phi = np.arctan(Vn/Vt)
        Theta = rotor['pitch']+rotor['twist']
        Alfa = np.rad2deg(Phi-Theta)

        # Lift and drag coefficient
        Cl = np.zeros(len(rotor['r']))
        Cd = np.zeros(len(rotor['r']))
        for i in range(len(rotor['r'])):
            Cl[i] = np.interp(Alfa[i], airfoil[rotor['airfoil']]["Alfa"], airfoil[rotor['airfoil']]["Cl"])
            Cd[i] = np.interp(Alfa[i], airfoil[rotor['airfoil']]["Alfa"], airfoil[rotor['airfoil']]["Cd"])

        # Normal and tangential coefficient (to rotor plane)
        cn = Cl*np.cos(Phi)+Cd*np.sin(Phi)
        ct = Cl*np.sin(Phi)-Cd*np.cos(Phi)

        # Local thrust and torque coefficient
        Ct = Vrel**2/flow['V0']**2*rotor['sigma']*cn
        Cq = Vrel**2/flow['V0']**2*rotor['sigma']*ct
        
        if (simulation['current_index'] == 0) or (simulation['model'] == 'Steady'):
            a_n = ainduction(Ct, glauert=True)
        else:
            if simulation['model'] == 'PP':
                a_o = results["a"][simulation['current_index']-1]
                f_o = results["f"][simulation['current_index']-1]
                a_n = pitt_peters(Ct, a_o*f_o, flow['V0'], rotor['r'], simulation['dt'], glauert=True)
            
            elif simulation['model'] == 'LM':
                a_o = results["a"][simulation['current_index']-1]
                f_o = results["f"][simulation['current_index']-1]
                Ct_o = results["Ct"][simulation['current_index']-1]
                a_n = larsen_madsen(Ct, a_o*f_o, flow['V0'], rotor['r'], simulation['dt'], glauert=True)
                
            elif simulation['model'] == 'OYE':
                if simulation['current_index'] == 1:
                    Ct_o = results["Ct"][simulation['current_index']-1]
                    v_int = -ainduction(Ct_o, glauert=True)*flow["V0"]
                else:
                    v_int = results["v_int"][simulation['current_index']-1]
                Ct_o = results["Ct"][simulation['current_index']-1]
                a_o = results["a"][simulation['current_index']-1]
                f_o = results["f"][simulation['current_index']-1]
                
                a_n, v_int = oye(a_o*f_o, Ct_o, Ct, v_int, flow['V0'], rotor['R'], rotor['r'], simulation['dt'],
                                 glauert=True)
            else:
                raise NotImplemented(f"Unsteady model {simulation['model']} not implemented.")
        
        exp = np.exp(-rotor['B']/2 * ((1-rotor['mu'])/rotor['mu']) * np.sqrt(1+flow['tsr']**2 *
                                                                             rotor['mu']**2/(1-a_n)**2))
        f_tip = 2/np.pi * np.arccos(exp)
        #Root correction
        exp = np.exp(-rotor['B']/2 * ((rotor['mu']-rotor['mu'][0])/rotor['mu']) * np.sqrt(1+flow['tsr']**2 *
                                                                                          rotor['mu']**2/(1-a_n)**2))
        f_root = 2/np.pi * np.arccos(exp)
        #Combined correction
        f = f_tip*f_root
        f[f < 0.1] = 0.1
        a_n = a_n/f
        # a_n[a_n > 0.95] = 0.95
        ap_n = ct*rotor['B']/(2*flow['rho']*2*np.pi*rotor['mu']*rotor['r']*flow['V0']**2*(1-a_n)*flow['tsr'] *
                              rotor['mu']*f)
        
        a_new = 0.2*a_n+0.8*a_old
        ap_new = 0.2*ap_n+0.8*ap_old

    # Blade forces
    pn = 1/2*flow['rho']*Vrel**2*rotor['chord']*cn
    pt = 1/2*flow['rho']*Vrel**2*rotor['chord']*ct

    # rotor performance
    M = rotor['B']*np.trapz(rotor['r']*pt, rotor['r'])
    P = M*flow['omega']
    CP = P/(1/2*flow['rho']*flow['V0']**3*np.pi*rotor['R']**2)
    T = rotor['B']*np.trapz(pn, rotor['r'])
    CT = T/(1/2*flow['rho']*flow['V0']**2*np.pi*rotor['R']**2)
    simulation['current_index'] += 1
    return {"P": P, "T": T, "CP": CP, "CT": CT, "a": a_new, "ap": ap_new, "f": f, "Ct": Ct, "Cq": Cq, "v_int": v_int,
            "alpha": Alfa, "phi": Phi}


def steady_reference(rotor, airfoil, flow, simulation, change: tuple[str, list or np.ndarray]):
    if len(change) != 2:
        raise ValueError(f"'steady_reference' must receive a tuple 'change' with two entries; first the name of the "
                         f"parameter that is to be changed and second the values thereof.")
    rotor_settable, flow_settable = rotor.settable(), flow.settable()
    param, values = change[0], change[1]
    if param in rotor_settable:
        setting_for = "rotor"
    elif param in flow_settable:
        setting_for = "flow"
    else:
        raise ValueError(f"Parameter {param} cannot be set for either the rotor (settable: {rotor_settable}) or the "
                         f"flow (settable: {flow_settable}).")
       
    # Solve steady BEM
    CP_array = np.zeros(len(values))
    CT_array = np.zeros(len(values))
    simulation['model'] = 'Steady'
    for i, value in enumerate(values):
        if setting_for == "rotor":
            rotor[param] = value
        elif setting_for == "flow":
            flow[param] = value
        result = BEM(rotor, airfoil, flow, simulation, None)
        CT_array[i] = result["CT"]
        CP_array[i] = result["CP"]
    return {param: values, "CT": CT_array, "CP": CP_array}


def get_steady_reference(rotor: Rotor, flow: Flow, airfoil: dict, simulation: Simulation,
                         change_param: str, change_type: str, change_values: tuple, min_root_error: float=1e-3):
    """
    Calculates certain steady values to match the wanted 'change_param' to achieve the 'change_values'.
    
    :param rotor:
    :param flow:
    :param airfoil:
    :param simulation:
    :param change_param: The parameter that has a predefined change. Implemented are "CT_steady" and "U_inf"
    :param change_type: The way the parameter changes. Implemented are "step" and "sine"
    :param change_values: Has a different format dependent on "change_type":
        - "change_type" == "step": (value_pre_step, value_after_step). The step will be at half the time given by
        'simulation'
        - "change_type" == "sine": (mean, amplitude, frequency).
    :return: Changes dependent on "change_param":
        - "CT_steady": returns list of needed pitch angles (rad)
        - "U_inf": returns list of needed tip speed ratios
    """
    implemented_change_param = ["CT_steady", "U_inf"]
    implemented_change_type = ["step", "sine"]
    time = simulation["time"]
    if change_param == "CT_steady":
        #  pitches the blades to the right angle to yield the wanted steady CT
        CT = {"wanted": None}
        
        def residual(pitch: float):
            tmp_result = steady_reference(rotor, airfoil, flow, simulation, change=("pitch", [pitch]))
            return tmp_result["CT"][0]-CT["wanted"]
        
        if change_type == "step":
            wanted_values = change_values
        elif change_type == "sine":
            mean, amplitude, freq = change_values
            wanted_values = mean+(amplitude*np.sin(2*np.pi*freq*time))
        else:
            raise NotImplemented(f"Calculation for 'change_type'=={change_type} not implemented. Implemented are "
                             f"{implemented_change_type}.")
        
        steady_results = np.zeros(len(wanted_values))  # the pitch
        for i, wanted_value in enumerate(wanted_values):
            CT["wanted"] = wanted_value
            steady_results[i] = optimize.newton(residual, steady_results[i-1], tol=min_root_error)
    elif change_param == "U_inf":
        #  changes the TSR to keep the rotational speed the same
        if change_type == "step":
            wanted_values = change_values
        elif change_type == "sine":
            mean, amplitude, freq = change_values
            wanted_values = mean+(amplitude*np.sin(2*np.pi*freq*time))
        else:
            raise NotImplemented(f"Calculation for 'change_type'=={change_type} not implemented. Implemented are "
                             f"{implemented_change_type}.")
        steady_results = flow["omega"]*rotor.R/np.asarray(wanted_values)  # the TSR
    else:
        raise NotImplemented(f"Calculation for 'change_param'=={change_param} not implemented. Implemented are "
                             f"{implemented_change_param}.")
    return steady_results, wanted_values


def calculate_case(rotor: Rotor, flow: Flow, airfoil: dict, simulation: Simulation, change_param: str,
                   change_type: str, change_values: tuple, models: tuple = ("OYE", "LM", "PP")):
    verbose = simulation.verbose
    steady_sim = copy(simulation)
    fewer_than_one_period = True
    if change_type == "sine":
        frequency = change_values[2]
        if frequency*steady_sim.actual_t_max < 1:  # not even one period resolved
            pass
        elif 1/(steady_sim.dt*frequency) < 3:
            raise ValueError(f"Sine waves must be resolved with at least 3 points but the current dt={steady_sim.dt} "
                             f"only resolves {np.round(1/(steady_sim.dt*frequency), 3)} points.")
        elif not (frequency/steady_sim.dt).is_integer():  # this will speed up calculations later
            fewer_than_one_period = False
            old_dt = steady_sim.dt
            old_res_per_period = np.floor(1/(old_dt*frequency))
            new_res_per_period = old_res_per_period+1
            new_dt = 1/(new_res_per_period*frequency)
            steady_sim["dt"] = new_dt
            steady_sim["t_max"] = 1/frequency
            simulation["dt"] = new_dt
            print(f"'dt' has been changed from {old_dt} to {new_dt} to resolve the zero-crossings of the sine.")
            
    if verbose:
        print("Calculating steady reference values")
        start_steady_reference = timeit.default_timer()
    steady_results, wanted_values = get_steady_reference(rotor, flow, airfoil, steady_sim, change_param, change_type,
                                                         change_values)
    if verbose:
        print(f"Finished calculating steady reference values ("
              f"{np.round(timeit.default_timer()-start_steady_reference, 3)}s).")
        
    if change_type == "step":
        time = simulation.time
        time_under1s, time_rest = time[time<1], time[time>=1]
        steady_results = np.hstack([np.full(time_under1s.size, steady_results[0]),
                                    np.full(time_rest.size, steady_results[1])])
        
        wanted_values = np.hstack([np.full(time_under1s.size, wanted_values[0]),
                                    np.full(time_rest.size, wanted_values[1])])
    elif not fewer_than_one_period:  # change_type == "sine
        n_full_periods, remaining_time_steps = np.divmod(len(simulation.time), 1/(simulation.dt*change_values[2]))
        n_full_periods, remaining_time_steps = int(n_full_periods), int(remaining_time_steps)
        steady_results = np.tile(steady_results, n_full_periods)
        steady_results = np.hstack([steady_results, steady_results[1:remaining_time_steps]])
        
        wanted_values = np.tile(wanted_values, n_full_periods)
        wanted_values = np.hstack([wanted_values, wanted_values[1:remaining_time_steps]])
        
    results = [Results(rotor, simulation)]
    simulations = [simulation]
    for model in models:
        if model != simulation.model:  # skip if input sim does already specify one of the models
            sim = copy(simulation)
            sim["model"] = model
            simulations.append(sim)
            results.append(Results(rotor, sim))
            
    for specific_simulation, result in zip(simulations, results):
        if verbose:
            print(f"Starting calculations on model: {result.simulation.model}")
            start_model = timeit.default_timer()
        for (t_i, steady_result), wanted_value in zip(enumerate(steady_results), wanted_values):
            if change_param == "CT_steady":
                rotor["pitch"] = steady_result
            else:
                flow["tsr"] = steady_result
                flow["V0"] = wanted_value
            tmp_result = BEM(rotor, airfoil, flow, specific_simulation, result)
            for param, value in tmp_result.items():
                result[param][t_i] = value
        if verbose:
            print(f"Finished calculations on model: {result.simulation.model} ("
                  f"{np.round(timeit.default_timer()-start_model, 3)}s).")
    return results
    
    


