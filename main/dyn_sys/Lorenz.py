"""
Create time series of a parametric Lorenz system.

Created on Tue Mar 14 10:10:22 2023

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
def Lorenz_system(t, state, sigma, beta, rho):
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
    x, y, z = state
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
 
    return [dx, dy, dz]

def euler_method(initial_condition, t_eval, dt, sigma, beta, rho):
    
    
    sol = np.zeros((t_eval.shape[0]+1, initial_condition.shape[0]))   
    sol[0, :] = initial_condition 
    for i, t in enumerate(t_eval):
        # Euler's method
        sol[i + 1, :] = sol[i, :] + np.array(Lorenz_system(t, sol[i, :], sigma, beta, rho))*dt

    return sol

def parametric_Lorenz(params, time_total = max_time_total, 
                      dt = 0.001, 
                      initial_condition = None, 
                      random_seed = None, plot_figure = False,
                      save_data = False, 
                      filename = None,
                      method = 'RK45'):
    '''
    Lorenz system with parameters given by the user.

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
    
    sigma = params[0]  
    beta = params[1]
    rho = params[2]    
    
    #=======RK4th_Method_Parameters=======##========================#
    t_eval = np.arange(0.0, time_total, dt)
    #========================##========================#
    #=======Variables and Initial conditions ========##========================#
    if filename is not None:
        folder = 'data/input_data/'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        output = os.path.join(out_direc, filename)
    
    if initial_condition is None:
        if random_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(random_seed)    
        
        initial_condition = [0.21, 0.1, 0.1] + 1e-3*rng.random(3)
        if filename is None:
            folder = 'data/input_data/'
            out_direc = os.path.join('', folder)
            
            if os.path.isdir(out_direc) == False:
                os.makedirs(out_direc)
            filename = 'Lorenz_ts_{}_{}_{}.txt'.format(time_total, dt, random_seed)
            output = os.path.join(out_direc, filename)
        else:
            output = filename
    else:
        if initial_condition.shape[0] != 3:
            raise ValueError("Initial condition must match the dimension of \
                             the dynamical system")
        if filename is None:
            folder = 'data/input_data/'
            out_direc = os.path.join('', folder)
            
            if os.path.isdir(out_direc) == False:
                os.makedirs(out_direc)
            
            filename = 'Lorenz_ts_{}_{}_ic_{}.txt'.format(time_total, dt, initial_condition)
            output = os.path.join(out_direc, filename)
        else:
            output = filename   
            
    initial_condition = np.array(initial_condition)
    
    if method != 'Euler':
        sol = solve_ivp(Lorenz_system, [0, time_total], initial_condition,
                        method= method,
                        args=(sigma, beta, rho), t_eval = t_eval, 
                        first_step=0.001, 
                        max_step = 0.001,
                        rtol = 1e-5,
                        atol = 1e-8)
        x_t, y_t, z_t = sol.y
    else:
        sol = euler_method(initial_condition, t_eval, dt, sigma, beta, rho)
        x_t, y_t, z_t = sol[:, 0], sol[:, 1], sol[:, 2]
    
    
    X_t = np.array([x_t, y_t, z_t])
    
    if save_data:
        np.savetxt(output, X_t.T)
        
    if plot_figure:
        fig = plt.figure(dpi = 300)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_t, y_t, z_t)
        ax.set_xlabel(r'x')
        ax.set_ylabel(r"y")
        ax.set_zlabel(r"z")
        plt.show()
        
    return X_t.T
    
def spy_Lorenz(state, sigma, beta, rho):

    '''
    Symbolic Lorenz dynamics.

    Parameters
    ----------
    x_t : list
        Symbolic variables for state.
    betaa : float, optional
        Rulkov parameter. The default is 4.4.
    sigma : float, optional
        Rulkov parameter. The default is 0.001.
    beta : float, optional
        Rulkov parameter. The default is 0.001.

    Returns
    -------
    f_isolated : sympy Matrix
        Symbolic isolated map.

    '''    
    x = state[0]
    y = state[1]
    z = state[2]
    
    f_dict = dict()
    
    f_dict[0] = sigma * (y - x)
    f_dict[1] = x * (rho - z) - y
    f_dict[2] = x * y - beta * z
        
    return f_dict    
    
def get_true_coeff_Lorenz(params):
    '''
    Obtain the true coefficient matrix for the network dynamics.

    Parameters
    ----------
    params : dict
        
    Returns
    -------
    c_matrix_true
    '''
    
    # Canonical parameter values for Lorenz:
    sigma, rho, beta = [10.0, 8.0/3.0, 28]
    
    dict_can_bf = polb.dict_canonical_basis(params)

    L, N, delay_dimension = params['L'], params['number_of_vertices'], params['delay_dimension']
    
    c_matrix_true = np.zeros((L, 3))        
    indexes = np.arange(0, N, delay_dimension, dtype = int)
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    state = [x_t[indexes[0]],x_t[indexes[1]],x_t[indexes[2]]]
    F = spy_Lorenz(state, sigma, rho, beta)
    
    for i, id_node in enumerate(indexes):
        c_matrix_true[:, i] = polb.get_coeff_matrix_wrt_basis(F[i].expand(), 
                                                                    dict_can_bf)
    
    return c_matrix_true

    
    
    
    
    
    
    
    
    

