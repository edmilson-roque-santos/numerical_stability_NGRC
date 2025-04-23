"""
Lorenz 96 simulation. 

Created on Tue Mar 25 13:09:35 2025

@author: Edmilson Roque dos Santos
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import os
from scipy.integrate import solve_ivp
import sympy as spy

from main.base_polynomial import poly_library as polb

max_time_total = 1000
#=======Vector field - Function========##========================#
def H(x):

    coupling = np.zeros(x.shape[0])
    for id_node in range(x.shape[0]):
    
        if (id_node > 1) & (id_node < x.shape[0] - 1):
            coupling[id_node] = x[id_node - 1]*(x[id_node + 1] - x[id_node - 2])
            
        if id_node == 0:
            coupling[id_node] = x[-1]*(x[1] - x[-2])
            
        if id_node == 1:
            coupling[id_node] = x[0]*(x[2] - x[-1])
            
        if id_node == x.shape[0]-1:
            coupling[id_node] = x[-2]*(x[0] - x[x.shape[0]-1-2])
            
    return coupling

def Lorenz_96(t, state, F):
    '''
    Vector field of the Lorenz system for parameters (sigma, beta, rho).

    Parameters
    ----------
    t : float
        Time to evaluate the vector field.
    state : numpy array
        State at which the vector field is evaluated.
    sigma : float
    beta : float
    rho : float

    Returns
    -------
    list
        Vector field at the state and time t.

    '''
    
    return - state + F + H(state)

def euler_method(initial_condition, t_eval, dt, F):
    
    
    sol = np.zeros((t_eval.shape[0]+1, initial_condition.shape[0]))   
    sol[0, :] = initial_condition 
    for i, t in enumerate(t_eval):
        # Euler's method
        sol[i + 1, :] = sol[i, :] + np.array(Lorenz_96(t, sol[i, :], F))*dt

    return sol

def parametric_Lorenz96(params, time_total = max_time_total, 
                        dt = 0.001, 
                        initial_condition = None, 
                        random_seed = None, plot_figure = False,
                        save_data = False, 
                        filename = None,
                        method = 'RK45'):
    '''
    Lorenz 96 system with parameters given by the user.

    Parameters
    ----------
    time_total : float
        Total time for integration of the ODE.
    dt : time step
        Time step of the time series sampling.
    initial_condition : numpy array, optional
        Initial condition for the ODE. The default is [0.21, 0.1, 0.1] + 1e-3*rng.random(3).
    random_seed : int, optional
        Seed for the pseudo random generator. The default is None.
    plot_figure : Boolean, optional
        Check if the time series will be displayed. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    #======= Classical Lorenz parameters (chaotic)========##========================#
    
    N = params[0]  
    F = params[1]
    
    #=======RK4th_Method_Parameters=======##========================#
    t_eval = np.arange(0.0, time_total, dt)
    #========================##========================#
    #=======Variables and Initial conditions ========##========================#
    if initial_condition is None:
        if random_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(random_seed)    
        
        initial_condition = rng.uniform(low = -1, high = 1, size = (N))
        if filename is None:
            folder = 'data/input_data/'
            out_direc = os.path.join('', folder)
            
            if os.path.isdir(out_direc) == False:
                os.makedirs(out_direc)
            filename = 'Lorenz96_ts_{}_{}_{}.txt'.format(time_total, dt, random_seed)
            output = os.path.join(out_direc, filename)
        else:
            output = filename
    else:
        if initial_condition.shape[0] != N:
            raise ValueError("Initial condition must match the dimension of \
                             the dynamical system")
        if filename is None:
            folder = 'data/input_data/'
            out_direc = os.path.join('', folder)
            
            if os.path.isdir(out_direc) == False:
                os.makedirs(out_direc)
            
            filename = 'Lorenz96_ts_{}_{}_ic_{}.txt'.format(time_total, dt, initial_condition)
            output = os.path.join(out_direc, filename)
        else:
            output = filename   
            
    initial_condition = np.array(initial_condition)
    
    if method != 'Euler':
        sol = solve_ivp(Lorenz_96, [0, time_total], initial_condition,
                        method= method,
                        args=(F, ), t_eval = t_eval, 
                        first_step=0.001, 
                        max_step = 0.001,
                        rtol = 1e-5,
                        atol = 1e-8)
        X_t = sol.y.T
    else:
        X_t = euler_method(initial_condition, t_eval, dt, F)
               
    if save_data:
        np.savetxt(output, X_t)
                    
    return X_t