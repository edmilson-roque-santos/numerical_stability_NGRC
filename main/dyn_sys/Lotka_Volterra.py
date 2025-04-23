"""
Lotka-Volterra system of equations

Created on Wed Nov 20 09:27:25 2024

@author: Edmilson Roque dos Santos
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from numpy.random import default_rng


#=======Vector field - Function========##========================#
def Lotka_volterra(t, state, alpha, beta, gamma, delta):
    '''
    Vector field of the Lotka-Volterra system for parameters (alpha, beta, gamma, delta).

    Parameters
    ----------
    t : float
        Time to evaluate the vector field.
    state : numpy array
        State at which the vector field is evaluated.
    alpha : float
    beta : float
    gamma : float
    delta : float

    Returns
    -------
    list
        Vector field at the state and time t.

    '''
    x, y = state
    
    dx = alpha*x - beta*x*y
    dy = - gamma*y + delta*x*y
     
    return [dx, dy]

def parametric_LV(params, time_total, dt, initial_condition = None, 
                  random_seed = None, plot_figure = False):
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
    
    alpha = params[0]  
    beta = params[1]
    gamma = params[2]    
    delta = params[3]    
    
    #=======RK4th_Method_Parameters=======##========================#
    t_eval = np.arange(0.0, time_total, dt)
    #========================##========================#
    #=======Variables and Initial conditions ========##========================#
    if initial_condition is None:
        if random_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(random_seed)    
        
        initial_condition = [4, 2] + 1e-3*rng.random(2)
    
    else:
        if initial_condition.shape[0] != 2:
            raise ValueError("Initial condition must match the dimension of \
                             the dynamical system")
    
    initial_condition = np.array(initial_condition)
    
    sol = solve_ivp(Lotka_volterra, [0, time_total], initial_condition, 
                    method = 'RK45',
                    args=(alpha, beta, gamma, delta), t_eval = t_eval, first_step= 0.01, 
                    max_step = 0.01)
    
    x_t, y_t = sol.y
    
    X_t = np.array([x_t, y_t])
    
    if plot_figure:
        fig = plt.figure(dpi = 300)
        ax = fig.add_subplot()
        ax.scatter(X_t[0, :], X_t[1, :])
        ax.set_xlabel(r'x')
        ax.set_ylabel(r"y")
        plt.show()
        
    return X_t.T

def LV_gen_params(num_components, params_0 = np.array([1, 1, 1, 1]), 
                      step = 0.1):
    
    params_components = dict()
    for id_k in range(num_components):
    
        params_components[id_k] = params_0.copy()
        params_components[id_k][2] = params_components[id_k][2]*(1+step*id_k)
    
    return params_components
    
    
    
    
    
    
    
    
    
    
    
    
    

