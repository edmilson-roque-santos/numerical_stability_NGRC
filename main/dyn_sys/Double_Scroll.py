"""
Create time series of a parametric Double Scroll system.
Introduced in: 
A. Chang, J. C. Bienfang, G. M. Hall, J. R. Gardner,
and D. J. Gauthier, Chaos: An Interdisciplinary Journal
of Nonlinear Science 8, 782 (1998)

Parameters from:
D. J. Gauthier, E. Bollt, A. Griffith, and W. A. S. Barbosa, Nat Commun 12, 5564 (2021)

Created on Mon Aug  4 19:08:28 2025

@author: Edmilson Roque dos Santos
"""


import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import os
from scipy.integrate import solve_ivp


max_time_total = 1000
#=======Vector field - Function========##========================#
def Double_Scroll_system(t, state, r1, r2, r4, alpha, ir):
    '''
    Vector field of the Double Scroll electronic circuit for parameters (r1, r2, r4, alpha, ir).

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
    v1, v2, i = state
    
    dv = v1 - v2
    g = (dv/r2) + 2*ir*np.sinh(alpha*dv)
    
    dv1 = (v1/r1) - g    
    dv2 = g - i
    di = v2 - r4*i

    return [dv1, dv2, di]

def euler_method(initial_condition, t_eval, dt, r1, r2, r4, alpha, ir):
    
    
    sol = np.zeros((t_eval.shape[0]+1, initial_condition.shape[0]))   
    sol[0, :] = initial_condition 
    for i, t in enumerate(t_eval):
        # Euler's method
        sol[i + 1, :] = sol[i, :] + np.array(Double_Scroll_system(t, sol[i, :], r1, r2, r4, alpha, ir))*dt

    return sol

def parametric_DoubleScroll(params, time_total = max_time_total, 
                      dt = 0.001, 
                      initial_condition = None, 
                      random_seed = None, plot_figure = False,
                      save_data = False, 
                      filename = None,
                      method = 'RK45'):
    '''
    Double scroll system with parameters given by the user.

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
    #======= Classical Double_Scroll parameters (chaotic)========##========================#
    
    r1 = params[0]
    r2 = params[1]
    r4 = params[2]
    alpha = params[3]
    ir = params[4]
    
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
            filename = 'Double_Scroll_ts_{}_{}_{}.txt'.format(time_total, dt, random_seed)
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
            
            filename = 'Double_Scroll_ts_{}_{}_ic_{}.txt'.format(time_total, dt, initial_condition)
            output = os.path.join(out_direc, filename)
        else:
            output = filename   
            
    initial_condition = np.array(initial_condition)
    
    if method != 'Euler':
        sol = solve_ivp(Double_Scroll_system, [0, time_total], initial_condition,
                        method= method,
                        args=(r1, r2, r4, alpha, ir), t_eval = t_eval, 
                        first_step = 0.001, 
                        max_step = 0.001,
                        rtol = 1e-5,
                        atol = 1e-8)
        x_t, y_t, z_t = sol.y
    else:
        sol = euler_method(initial_condition, t_eval, dt, r1, r2, r4, alpha, ir)
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
    