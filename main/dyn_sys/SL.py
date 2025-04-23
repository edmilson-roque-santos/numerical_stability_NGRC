"""
Create time series of a parametric Stuart-Landau system.

Created on Wed Feb  7 13:10:28 2024

@author: Edmilson Roque dos Santos
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from numpy.random import default_rng


def stuart_landau(t, z, gamma, w, beta):
    return (gamma + 1j*w + beta*z*np.conj(z)) * z

def parametric_SL(params, time_total, dt, initial_condition = None, 
                      random_seed = None, plot_figure = False):
    '''
    Stuart-Landau system with parameters given by the user.

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
    gamma = params[0]  
    w = params[1]
    beta = params[2]    
    
    #=======RK4th_Method_Parameters=======##========================#
    t_eval = np.arange(0.0, time_total, dt)
    #========================##========================#
    #=======Variables and Initial conditions ========##========================#
    if initial_condition is None:
        if random_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(random_seed)    
        
        initial_condition = 1.2 + 1e-3*rng.random(1)
    else:
        if initial_condition.shape[0] != 2:
            raise ValueError("Initial condition must match the dimension of \
                             the dynamical system")
    
    initial_condition = np.array(initial_condition, dtype = np.cdouble)
    
    sol = solve_ivp(stuart_landau, [0, time_total], initial_condition, 
                    args=(gamma, w, beta), t_eval = t_eval, first_step=0.01, 
                    max_step = 0.01)
    
    z_t = sol.y[0, :]
    
    X_t = np.empty((z_t.shape[0], 2))
    X_t[:, 0], X_t[:, 1] = np.real(z_t), np.imag(z_t)
    
    if plot_figure:
        fig = plt.figure(dpi = 300)
        ax = fig.add_subplot()
        ax.scatter(X_t[:, 0], X_t[:, 1])
        ax.set_xlabel(r'x')
        ax.set_ylabel(r"y")
        plt.show()
        
    return X_t

def SL_gen_params(num_components, params_0 = np.array([1, 1, -1])):
    
    params_components = dict()
    for id_k in range(num_components):
    
        params_components[id_k] = params_0.copy()
        params_components[id_k][1] = params_components[id_k][1]*(1+1.*id_k)
    
    return params_components

#=======Network dynamics vector field ========##========================#
def net_dynamics(t, x, Lambda, f_isolated, h_coupling):
    '''
    Iterates the network dynamics at one time step.    

    Parameters
    ----------
    x : numpy array - shape (N,)
        State at time step t.
    args : dict
    Dictionary with network dynamics information content.
    Keys: 
        'coupling' : float
            coupling strength
        'max_degree' : int
            maximum degree of the network
        'f' : function
            isolated map
        'h' : function
            coupling function
    Returns
    -------
    numpy array
        Next state at time step t + 1.

    '''
    
    return f_isolated(x) + h_coupling(x)*Lambda

def net_SL(params, G, time_total, dt, initial_condition = None, 
           random_seed = None, plot_figure = False):
    '''
    It generates a trajectory of the network dynamics with length given by
    "number_of_iterations". The initial condition is given by a uniform
    random distribution in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    number_of_iterations : TYPE
        DESCRIPTION.
    args : dict
        Dictionary with network dynamics information content.
        Keys: 
            'random_seed' : int
                Seed for the pseudo random generator.
            'adj_matrix' : numpy array
                Adjacency matrix 
            'eps' : float
                noise magnitude
            'coupling' : float
                coupling strength
            'max_degree' : int
                maximum degree of the network
            'f' : function
                isolated map
            'h' : function
                coupling function
    use_noise : boolean
        Add dynamical noise to the network dynamics.
    Returns
    -------
    time_series : numpy array
        Multivariate time series of the network dynamics.

    '''
    L = nx.laplacian_matrix(G).toarray()
    L = np.asarray(L)
    N = L.shape[0]
    if initial_condition is None:
        if random_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(random_seed)    
        
        initial_condition = 1.2 + 1e-3*rng.random(N)
    
    initial_condition = np.array(initial_condition, dtype = np.cdouble) 
    
    f_isolated = lambda x: (params['gamma'] + 1j*params['w'] +  params['beta']*x*np.conj(x))*x
    h_coupling = lambda x: - L @ x
    
    #=======RK4th_Method_Parameters=======##========================#
    t_eval = np.arange(0.0, time_total, dt)

    ic = np.array(initial_condition, dtype = np.cdouble)
    sol = solve_ivp(net_dynamics, [0, time_total], ic, 
                    args=(params['Lambda'], f_isolated, h_coupling), 
                    t_eval = t_eval, first_step=0.01, 
                    max_step = 0.01)
        
    z_t = sol.y.T
    
    X_t = np.empty((z_t.shape[0], 2*N))
    
    if plot_figure:
        fig = plt.figure(dpi = 300)
        ax = fig.add_subplot()
       
    id_vec = np.arange(0, 2*N - 1, 2)    
    for i, id_ in enumerate(id_vec):
        X_t[:, id_], X_t[:, id_+1] = np.real(z_t[:, i]), np.imag(z_t[:, i])
    
        if plot_figure:
            ax.scatter(X_t[:, id_], X_t[:, id_+1])
            ax.set_xlabel(r'x')
            ax.set_ylabel(r"y")
    
    if plot_figure:
        plt.show()
        
    return X_t    
'''
G = nx.path_graph(3)
params = {'gamma': 1.0, 'beta' : -1.0, 'w' : np.array([1.0, 2.32, 5]), 'Lambda':0.01}
time_total= 100
dt = 0.01

X_t = net_SL(params, G, time_total, dt, initial_condition = None, 
             random_seed = None, plot_figure = True)

'''









