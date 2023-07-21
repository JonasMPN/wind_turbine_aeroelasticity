import numpy as np
from scipy.interpolate import interp1d
from math import pi, acos, exp, atan, cos, sin
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

def CTfunction(a, glauert = False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    CT = np.zeros(np.shape(a))
    CT = 4*a*(1-a)  
    if glauert:
        CT1=1.816;
        a1=1-np.sqrt(CT1)/2;
        CT[a>a1] = CT1-4*(np.sqrt(CT1)-1)*(1-a[a>a1])
    
    return CT


def pitt_peters(Ct,vind,Uinf,R,dt,glauert=False ):
    
    # this function determines the time derivative of the induction at the annulli 
    # Ct is the thrust coefficient on the actuator, vind is the induced velocity, 
    # Uinf is the unperturbed velocity and R is the radial scale of the flow, nd dt is the time step
    # it returns the new value of induction vindnew and the time derivative dvind_dt
    # glauert defines if glauert's high load correction is applied
    
    # a=-vind/Uinf # determine the induction coefficient for the time step {i-1}
    # Ctn= -CTfunction(a, glauert) # calculate the thrust coefficient from the induction for the time step {i-1}

    # dvind_dt =  (Ct-Ctn )/(16/(3*np.pi))*(Uinf**2/R) # calculate the time derivative of the induction velocity
    # a_new = vind + dvind_dt*dt # calculate the induction at time {i} by time integration
    Ctn= CTfunction(vind, glauert)
    dvdt = 3*np.pi*Uinf**2/16/R*(Ct - Ctn)
    v = -vind*Uinf - dvdt*dt
    a_new = -v/Uinf
    return a_new

def function_BEM(ROTOR, AIRFOIL, FLOW, SIMULATION, RESULTS):
    # Pre-allocate variables
    a_new = np.full(len(ROTOR['r']), 0.3)
    ap_new = np.zeros(len(ROTOR['r']))
    a_old = np.zeros(len(ROTOR['r']))
    ap_old = np.zeros(len(ROTOR['r']))

    iteration = 0

    while np.sum(np.abs(a_new-a_old)) > SIMULATION['error'] or np.sum(np.abs(ap_new-ap_old)) > SIMULATION['error']:
        iteration += 1
        if iteration > 1000:
            print(SIMULATION['current_index'])
            raise Exception("Too many iterations")

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
        f[f < 0.1] = 0.1
        F = 2/pi*np.arccos(np.exp(-f))
        Ctb = CT/F
        k = [0.00 ,0.251163 ,0.0544955 ,0.0892074]
        a_n = (k[3]*Ctb**3+k[2]*Ctb**2+k[1]*Ctb+k[0])
        ap_n = CQ/(4*F*(1-a_n)*FLOW['omega']*ROTOR['r']/FLOW['V0'])
        
        if (SIMULATION['model'] == 'PP') & (SIMULATION['current_index'] > 0):
            a_o = RESULTS["a"][SIMULATION['current_index']-1]
            a_n = pitt_peters(CT,a_o,FLOW['V0'],ROTOR['r'],SIMULATION['dt'] ,glauert=True )
        
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
    

    SIMULATION['current_index'] += 1
    return P, T, CP, CT, a_new, ap_new


# Input Parameters
# Rotor
ROTOR = {}
ROTOR['R'] = 50                                 # Radius [m]
ROTOR['D'] = 2*ROTOR['R']                      # Diameter [m]  
# ROTOR['H'] = 90                                # Hub height [m]
ROTOR['B'] = 3                                 # Number of blades [-]
ROTOR['theta_pitch'] = np.deg2rad(-2)                       # Blade (collective) pitch angle [deg]

ROTOR['r']= np.linspace(0.2*ROTOR['R'] , ROTOR['R'], 20) # Radial positions [-]
ROTOR['beta']= np.deg2rad(14*(1-ROTOR['r']/ROTOR['R']))  # Blade twist [deg] 
ROTOR['chord']= (3*(1-ROTOR['r']/ROTOR['R'])+1) # Blade chord [m]
ROTOR['airfoil'] = 'DU 95-W-180'
ROTOR['sigma'] = ROTOR['chord']*ROTOR['B']/(2*np.pi*ROTOR['r']) # Rotor solidity [-]   

# Flow
FLOW = {}
FLOW['V0'] = 10                                 # Inflow velocity [m/s]   
FLOW['rho'] = 1.225                             # Air density [kg/m2]
FLOW['lambda'] = 10
FLOW['omega'] = FLOW['lambda']*FLOW['V0']/ROTOR['R'] # Rotational speed [rad/s]

# Airfoil  
AIRFOIL = {}
AIRFOIL['DU 95-W-180'] = pd.read_excel('DU95W180.xlsx',skiprows=3) # Blade Airfoils [-] # Root airfoil: alpha, Cl, Cd, Cm

# Simulation options  
SIMULATION = {}
SIMULATION['error'] = 0.001                      # Convergence criteria BEM
SIMULATION['dt'] = 0.05                         # Time step [s]
SIMULATION['time'] = np.arange(0, 9, SIMULATION['dt']) # Time series [s]
SIMULATION['current_index'] = 0
SIMULATION['model'] = 'Steady'
# SIMULATION['taustar_nw'] = 0.5                  # Constants for dynamic inflow model
# SIMULATION['taustar_fw'] = 2                    # Constants for dynamic inflow model

RESULTS = {}
RESULTS['P'] = np.zeros(len(SIMULATION['time']))
RESULTS['T'] = np.zeros(len(SIMULATION['time']))
RESULTS['CP'] = np.zeros(len(SIMULATION['time']))
RESULTS['CT'] = np.zeros(len(SIMULATION['time']))
RESULTS['a'] = np.zeros([len(SIMULATION['time']), len(ROTOR['r'])])
RESULTS['ap'] = np.zeros([len(SIMULATION['time']), len(ROTOR['r'])])

CT0 = 1.1
CT1 = 0.4

def generate_reference_curve(ROTOR, AIRFOIL, FLOW, SIMULATION, RESULTS):
    # Solve (steady) BEM
    CP_array = np.zeros(len(SIMULATION['time']))
    CT_array = np.zeros(len(SIMULATION['time']))
    beta_array = np.linspace(-0.2, 0.2, len(SIMULATION['time']))
    
    SIMULATION['model'] = 'Steady'
    for i, t in enumerate(SIMULATION['time']):
        ROTOR['theta_pitch'] = beta_array[i]
        P, T, CP, CT, a_new, ap_new = function_BEM(ROTOR,AIRFOIL,FLOW,SIMULATION, RESULTS)
        CT_array[i] = CT
        
    REFERENCE = {}
    REFERENCE['beta'] = beta_array
    REFERENCE['CT'] = CT_array
    return REFERENCE
    



# find twist
REFERENCE = generate_reference_curve(ROTOR, AIRFOIL, FLOW, SIMULATION, RESULTS)

f = interpolate.interp1d(REFERENCE['CT'], REFERENCE['beta'])
beta0, beta1 = f([CT0, CT1])

# beta0 = np.interp(CT0, REFERENCE['CT'], REFERENCE['beta'])
# beta1 = np.interp(CT1, REFERENCE['CT'], REFERENCE['beta'])

# plt.figure()
# plt.plot( REFERENCE["CT"], REFERENCE["beta"])
# plt.scatter([CT0, CT1],[beta0, beta1])

#%%

beta = np.zeros(np.shape(SIMULATION['time']))+beta0 # we initialize the array of thrust coefficient, setting all initial values at Ct0

beta[1:] = beta1 

CT_array = np.zeros(len(SIMULATION['time']))
a_array = np.zeros(len(SIMULATION['time']))
SIMULATION['current_index'] = 0 
for i, b in enumerate(beta):
    SIMULATION['model'] = 'PP'
    ROTOR['theta_pitch'] = b
    curr_res = function_BEM(ROTOR,AIRFOIL,FLOW,SIMULATION, RESULTS)
    for j, key in enumerate(RESULTS):
        RESULTS[key][i] = curr_res[j]
    
    
    
#%%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(SIMULATION['time'], RESULTS['CT'], color = 'g')
ax2.plot(SIMULATION['time'], np.mean(RESULTS['a'],axis=1), color='b')


ax1.set_xlabel('Time')
ax1.set_ylabel('CT', color='g')
ax2.set_ylabel('a', color='b')







    # new_results = {
    # 'P': P,
    # 'T': T,
    # 'CP': CP,
    # 'CT': CT,
    # 'a': a_new,
    # 'ap': ap_new
    # }
    
    # for key, value in new_results.items():
    #     RESULTS[key][SIMULATION['current_index']] = value
