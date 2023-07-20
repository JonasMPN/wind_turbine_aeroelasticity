import numpy as np
from scipy.interpolate import interp1d
from math import pi, acos, exp, atan, cos, sin
import pandas as pd
import matplotlib.pyplot as plt


def function_BEM(ROTOR, AIRFOIL, FLOW, SIMULATION):
    # Pre-allocate variables
    a_new = np.full(len(ROTOR['r']), 0.3)
    ap_new = np.zeros(len(ROTOR['r']))
    a_old = np.zeros(len(ROTOR['r']))
    ap_old = np.zeros(len(ROTOR['r']))

    iteration = 0

    while np.sum(np.abs(a_new-a_old)) > SIMULATION['error'] or np.sum(np.abs(ap_new-ap_old)) > SIMULATION['error']:
        iteration += 1

        # Update induction
        a_old = a_new
        ap_old = ap_new

        # Velocity component
        Vn = (1-a_new)*FLOW['V0']
        Vt = (1+ap_new)*FLOW['omega']*ROTOR['r']
        Vrel = np.sqrt(Vn**2+Vt**2)

        # Inflow angles
        Phi = np.arctan(Vn/Vt)
        Theta = ROTOR['theta_pitch']+ROTOR['beta']
        Alfa = np.rad2deg(Phi-Theta)

        # Lift and drag coefficient
        Cl = np.zeros(len(ROTOR['r']))
        Cd = np.zeros(len(ROTOR['r']))
        for i in range(len(ROTOR['r'])):
            Cl[i] = np.interp(Alfa[i], AIRFOIL[ROTOR['airfoil']]["Alfa"], AIRFOIL[ROTOR['airfoil']]["Cl"])
            Cd[i] = np.interp(Alfa[i], AIRFOIL[ROTOR['airfoil']]["Alfa"], AIRFOIL[ROTOR['airfoil']]["Cd"])

        # Normal and tangential coefficient (to rotor plane)
        cn = Cl*np.cos(Phi)+Cd*np.sin(Phi)
        ct = Cl*np.sin(Phi)-Cd*np.cos(Phi)

        # Thrust and torque coefficient
        CT = Vrel**2/FLOW['V0']**2*ROTOR['sigma']*cn
        CQ = Vrel**2/FLOW['V0']**2*ROTOR['sigma']*ct

        # Axial and tangential induction
        f = ROTOR['B']*(ROTOR['R']-ROTOR['r'])/(2*ROTOR['r']*np.sin(Phi))
        # for i in range(len(f)):
        #     if f[i] < 0.01:
        #         f[i] = 0.01
        F = 2/pi*np.arccos(np.exp(-f))
        Ctb = CT/F
        k = [0.00 ,0.251163 ,0.0544955 ,0.0892074]
        a_n = (k[3]*Ctb**3+k[2]*Ctb**2+k[1]*Ctb+k[0])
        ap_n = CQ/(4*F*(1-a_n)*FLOW['omega']*ROTOR['r']/FLOW['V0'])
        a_new = (0.2*a_n+0.8*a_old)
        ap_new = 0.2*ap_n+0.8*ap_old

    # Blade forces
    pn = 1/2*FLOW['rho']*Vrel**2*ROTOR['chord']*cn
    pt = 1/2*FLOW['rho']*Vrel**2*ROTOR['chord']*ct

    # Rotor performance
    M = ROTOR['B']*np.trapz(ROTOR['r']*pt,ROTOR['r'])
    P = M*FLOW['omega']
    CP = P/(1/2*FLOW['rho']*FLOW['V0']**3*pi*ROTOR['R']**2)
    T = ROTOR['B']*np.trapz(pn, ROTOR['r'])
    CT = T/(1/2*FLOW['rho']*FLOW['V0']**2*pi*ROTOR['R']**2)

    return P, T, CP, CT, a_new, ap_new


# Input Parameters
# Flow
FLOW = {}
FLOW['V0'] = 10                                 # Inflow velocity [m/s]   
FLOW['rho'] = 1.225                             # Air density [kg/m2]
# FLOW['omega'] = 9*2*np.pi/60  

# Rotor
ROTOR = {}

ROTOR['R'] = 50                                 # Radius [m]
ROTOR['D'] = 2*ROTOR['R']                      # Diameter [m]  
# ROTOR['H'] = 90                                # Hub height [m]
ROTOR['B'] = 3                                 # Number of blades [-]
ROTOR['theta_pitch'] = np.deg2rad(-2)                       # Blade (collective) pitch angle [deg]

ROTOR['r']= np.linspace(0.2*ROTOR['R'] , ROTOR['R']-0.1, 100) # Radial positions [-]
ROTOR['beta']= np.deg2rad(14*(1-ROTOR['r']/ROTOR['R']))  # Blade twist [deg] 
ROTOR['chord']= (3*(1-ROTOR['r']/ROTOR['R'])+1) # Blade chord [m]
ROTOR['airfoil'] = 'DU 95-W-180'
ROTOR['sigma'] = ROTOR['chord']*ROTOR['B']/(2*np.pi*ROTOR['r']) # Rotor solidity [-]   

# Airfoil  
AIRFOIL = {}
AIRFOIL['DU 95-W-180'] = pd.read_excel('DU95W180.xlsx',skiprows=3) # Blade Airfoils [-] # Root airfoil: alpha, Cl, Cd, Cm

# Simulation options  
SIMULATION = {}
SIMULATION['error'] = 0.01                      # Convergence criteria BEM
# SIMULATION['dt'] = 0.1                          # Time step [s]
# SIMULATION['time'] = np.arange(0, 160, SIMULATION['dt']) # Time series [s]  
# SIMULATION['taustar_nw'] = 0.5                  # Constants for dynamic inflow model
# SIMULATION['taustar_fw'] = 2                    # Constants for dynamic inflow model

# Solve (steady) BEM
CP_array = []
CT_array = []
lambda_array = [10] 


for i in range(len(lambda_array)):

    lambda_val = lambda_array[i]    
    FLOW['omega'] = lambda_val*FLOW['V0']/ROTOR['R'] # Rotational speed [rad/s]
    P,T,CP,CT,a_new,ap_new = function_BEM(ROTOR,AIRFOIL,FLOW,SIMULATION)

    CP_array.append(CP)
    CT_array.append(CT)

# plt.figure(1)
# plt.plot(lambda_array,CP_array)

plt.figure(2)
plt.plot(ROTOR["r"]/ROTOR["R"],a_new)
