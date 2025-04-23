"""
Script for plotting results from different experiments. 

Created on Fri Apr 11 10:43:33 2025

@author: Edmilson Roque dos Santos
"""


import numpy as np

from main.simulation import simulation as sim
from main.lab import lab
import main.tools as tls


# %%
def fig_delays_time_lags():
    # Script for plotting Figure of cond number for different experiments
    
    #This script the following experiment:
    '''
    Panel a) Histogram of the condition number for different delays
    
    Panel b) Box plot the condition number versus different time lags
    
    Ntrain  = 5000
    and the step size $\Delta t = 0.01$.
    
    '''    
    # Dictionaries for saving the results.
    res_dicts = dict()
    labels_dict = dict()
    titles_dict = dict()
    plot_dict = dict()
    # Extract the experiment of delay dimension
    
    experiment = dict()
    
    vary_params = dict()
    vary_params['delay_dimension'] = np.arange(1, 5, 1, dtype = int)
    vary_params['ttrain'] = np.array([50])
    vary_params['max_deg_monomials'] = [2]
    
    experiment['exp_name'] = 'Norm_ill_cond_delay_max_deg_Ntrain'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment['testing'] = False
    
    fixed_params = {'reg_params' : 0,
                    'ttest' : 100,
                    'Nseeds' : 100,
                    'normalize_ts' : False,
                    'normalize_cols' : True,
        }
    
    L = lab(experiment, vary_params, fixed_params)
       
    subdict = dict()
    subdict['delay_dimension'] = np.arange(1, 5, 1, dtype = int)
    subdict['ttrain'] = np.array([50])
    subdict['max_deg_monomials'] = [2]
    
    x_keys = 'delay_dimension'
    res_dicts[x_keys] = L.results_sigvals(subdict, x_keys,
                                          filename = None)
    
    labels = []
    for id_key, key in enumerate(res_dicts[x_keys].keys()):
        labels.append(r'$k = {}$'.format(key))
    labels_dict[x_keys] = labels
    titles_dict[x_keys] = r' '
    
    # Extract the experiment of maximum degree of polynomials
    
    experiment = dict()
    
    vary_params = dict()
    vary_params['time_skip'] = np.arange(1, 72, 2, dtype = int)
    
    experiment['exp_name'] = 'Time_lag_1-70_ic_50'
    experiment['network_name'] = 'Lorenz_63'
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    experiment['dependencies'] = False
    # Specify if we should perform testing
    experiment['testing'] = False
    
    fixed_params = {'delay_dimension' : 2,
                    'ttrain': 50,
                    'ttest': 1,
                    'Nseeds': 25,
                    'reg_params' : 0,
                    'normalize_ts' : True,
                    'normalize_cols' : True,
        }
    
    
    L_1 = lab(experiment, vary_params, fixed_params)
    x_keys = 'time_skip'
    
    list_metrics = ['sigvals']
    res_dicts[x_keys] = L_1.metrics_time_skip(list_metrics, 
                                              x_keys = x_keys)
    
    experiment = dict()
    
    vary_params = dict()
    vary_params['time_skip'] = np.arange(1, 72, 2, dtype = int)
    
    experiment['exp_name'] = 'x-coord_Time_lag_1-70_ic_50'
    experiment['network_name'] = 'Lorenz_63'
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method_index'
    experiment['dependencies'] = False
    # Specify if we should perform testing
    experiment['testing'] = False
    
    fixed_params = {'delay_dimension' : 2,
                    'max_deg_monomials' : 1,
                    'ttrain': 50,
                    'ttest': 1,
                    'Nseeds': 25,
                    'reg_params' : 0,
                    'normalize_ts' : True,
                    'normalize_cols' : True,
                    'index' : np.array([0])
        }
    
    L2 = lab(experiment, vary_params, fixed_params)
    x_keys = 'time_skip'
    
    list_metrics = ['sigvals']
    
    res_dicts['x_coord'] = L2.metrics_time_skip(list_metrics, x_keys = x_keys)
    
    labels_dict[x_keys] = labels
    titles_dict[x_keys] = r'Delay dimension $k = 1$'
    plot_dict[x_keys] = {'x_axis': vary_params[x_keys]}
    plot_figure = True
    if plot_figure:
        tls.fig_cond_number_delay(res_dicts, 
                                  labels_dict, 
                                  titles_dict,
                                  plot_dict,
                                  filename = None) #'fig_delays_time_lags'
# %%
def fig_cond_number_Ntrain():
    
    experiment = dict()
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method_lgth_train'
    # Specify if we should perform testing
    experiment['testing'] = False
    
    
    #Fix the number of delay dimensions to k = 1.
    vary_params1 = dict()
    
    vary_params1['Ntrain'] = np.unique(np.concatenate((np.linspace(10, 300, 50, dtype = int),
                                                       np.linspace(10, 50, 20, dtype = int)))
                                       )
    
    vary_params1['dt'] = np.array([0.01, 0.05, 0.1, 0.15, 0.25])
    
    experiment1 = experiment.copy()
    experiment1['exp_name'] = 'k_1_Ntrain_10-300_num_of_points_50'
    
    fixed_params1 = {'delay_dimension': 1,
                    'time_skip': 1,
                    'Nseeds': 10,
                    'normalize_ts' : True,
                    'normalize_cols' : True,
                    'Ntest' : 1
        }
    
    #Fix the number of delay dimensions to k = 2, time skip = 50.
    vary_params2 = dict()
    vary_params2['Ntrain'] = np.unique(np.concatenate((np.geomspace(50, 5000, num=50, endpoint=True, dtype = int),
                                                       np.linspace(50, 100, 20, dtype = int)))
                                       )
    
    
    vary_params2['dt'] = np.array([0.01, 0.05, 0.1, 0.15, 0.25])
    
    experiment2 = experiment.copy()
    experiment2['exp_name'] = 'Ntrain_50-700_num_of_points_50'
    fixed_params2 = {'delay_dimension': 2,
                    'time_skip': 50,
                    'Nseeds': 10,
                    'normalize_ts' : True,
                    'normalize_cols' : True,
                    'Ntest' : 1
        }
    
    L1 = lab(experiment1, vary_params1, fixed_params1)
    x_keys = 'Ntrain'
    list_metrics = ['sigvals']
    
    res_dict = dict()
    plot_dict = dict()
    x_dict = dict()
    plot_dict['exp_1'] = {'title' : r'$k = 1$',
                          'x_lim': [10, 300]}
    res_dict['exp_1'] = L1.metrics_lgth_train(list_metrics, x_keys = 'Ntrain', 
                                             curve_keys = ['dt'])
    x_dict['exp_1'] = vary_params1['Ntrain']
    
    
    L2 = lab(experiment2, vary_params2, fixed_params2)
    plot_dict['exp_2'] = {'title' : r'$k = 2, \tau = 50$',
                          'x_lim': [50, 5000]}
    res_dict['exp_2'] = L2.metrics_lgth_train(list_metrics, x_keys = 'Ntrain', 
                                             curve_keys = ['dt'])#
    
    x_dict['exp_2'] = vary_params2['Ntrain']
    tls.fig_metrics_lgth_train(res_dict, x_dict,
                               plot_dict,
                               filename = 'sigvals_lgth_train')
    
    return res_dict
# %%
def fig_cond_num_poly_deg():
    #This script test the following experiment:
    '''
    Consider 25 different initial conditions. 
    Vary the maximum degree of polynomial
    Fix the Ntrain = [5000]
    
    time_skip = 1
    Ntest = 100
    and the step size $\Delta t = 0.01$.    
    '''    
    
    experiment = dict()
    
    vary_params = dict()
    
    vary_params['max_deg_monomials'] = np.arange(2, 11, 1, dtype = int)
    
    experiment['exp_name'] = 'ill_cond_max_deg'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment['testing'] = False
    
    fixed_params = {'reg_params' : 0,
                    'ttrain' : 50,
                    'ttest' : 1,
                    'Nseeds' : 25,
                    'normalize_ts' : True,
                    'normalize_cols' : True,
        }
    
    L = lab(experiment, vary_params, fixed_params)
    exps_dict = L.exp_dict
    res = L.metrics_poly_deg(['sigvals'], curve_keys=None)
    x_axis = L.vary_params['max_deg_monomials']
    tls.plot_cond_num_poly_deg(res['sigvals'], 
                               x_axis, 
                               plot_dict = {'label':''},
                               filename = experiment['exp_name'])
# %%
def fig_reg_solvers():
    #This script test the following experiment:
    '''
    Consider 50 different initial conditions. 
    Vary the solvers: Cholesky SVD LU 
    Fix the Ntrain = [5000]
    
    delay_dimension = 2
    time_skip = 1
    Ntest = 100
    and the step size $\Delta t = 0.01$.    
    '''    
    res_dict = dict()
    list_metrics = ['VPT_test', 'zmax_test', 'abs_psd_test']
    
    reg_params = np.array([0, 1e-15, 1e-13, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    #=========================================================================#
    # Compute the solution using Cholesky factorization in Normal equation
    #=========================================================================#
    experiment = dict()
    
    vary_params = dict()
    
    vary_params['reg_params'] = reg_params.copy()
    
    experiment['exp_name'] = 'CHO_reg_0_1_normalized'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment['testing'] = True
    
    fixed_params = {'delay_dimension': 2,
                    'time_skip': 1,
                    'ttrain': 50,
                    'ttest': 100,
                    'Nseeds': 50,
                    'normalize_ts' : True,
                    'normalize_cols' : False,
                    'solver_ridge' : 'cholesky'
        }
    
    L = lab(experiment, vary_params, fixed_params)
    res_dict['Cholesky'] = dict()
    
    for id_list, key_metric in enumerate(list_metrics):    
        
        res_dict['Cholesky'][key_metric] = L.results_metric_reg(key_metric,
                                                                x_keys = 'reg_params',
                                                                curve_keys= None)
    
    #=========================================================================#
    # Compute the solution using SVD for least square
    #=========================================================================#
    
    experiment1 = dict()
    
    vary_params1 = dict()
    
    vary_params1['reg_params'] = reg_params.copy()
    
    experiment1['exp_name'] = 'SVD_reg_0_1_normalized'
    experiment1['network_name'] = 'Lorenz_63'
    experiment1['dependencies'] = False
    # Specify the script to use for the test.
    experiment1['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment1['testing'] = True
    
    fixed_params1 = {'delay_dimension': 2,
                    'time_skip': 1,
                    'ttrain': 50,
                    'ttest': 100,
                    'Nseeds': 50,
                    'normalize_ts' : True,
                    'normalize_cols' : False,
                    'solver_ridge' : 'SVD'
        }

    L1 = lab(experiment1, vary_params1, fixed_params1)
    res_dict['SVD'] = dict()
    
    for id_list, key_metric in enumerate(list_metrics):    
        
        res_dict['SVD'][key_metric] = L1.results_metric_reg(key_metric,
                                                            x_keys = 'reg_params',
                                                            curve_keys= None)
    #=========================================================================#
    # Compute the solution using inverse formula - LU factorization
    #=========================================================================#
    
    experiment2 = dict()
    
    vary_params2 = dict()
    
    vary_params2['reg_params'] = reg_params.copy()
    
    experiment2['exp_name'] = 'LU_reg_0_1_normalized'
    experiment2['network_name'] = 'Lorenz_63'
    experiment2['dependencies'] = False
    # Specify the script to use for the test.
    experiment2['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment2['testing'] = True
    
    fixed_params2 = {'delay_dimension': 2,
                    'time_skip': 1,
                    'ttrain': 50,
                    'ttest': 100,
                    'Nseeds': 50,
                    'normalize_ts' : True,
                    'normalize_cols' : False,
                    'solver_ridge' : 'LU'
        }
    
        
    L2 = lab(experiment2, vary_params2, fixed_params2)
    
    res_dict['LU'] = dict()
    
    for id_list, key_metric in enumerate(list_metrics):    
        
        res_dict['LU'][key_metric] = L2.results_metric_reg(key_metric,
                                                           x_keys = 'reg_params',
                                                           curve_keys= None)
    
    tls.fig_reg_solver(res_dict, x_axis = reg_params, filename = 'solvers_compare')
# %%
def fig_x_coord_reconstr_time_lags():
    # Script for plotting Figure of reconstr x_coord for increasing time lags
    
    #This script the following experiment:
    
    '''
    Select only the x-coordinate
    Consider ten different initial conditions. 
    Fix the number of data points to Ntrain = 5000 
    Ntest = 1
    and the step size $\Delta t = 0.01$.
    Fix the number of delay dimensions to k = 2.
    Fix the maximum degree of polynomial
    Fix the x-variable only
    regularizer parameter = 0
    Let us consider two variations:
        different time skips
        
    '''     
    # Dictionaries for saving the results.
    res_dicts = dict()
    experiment = dict()
    
    vary_params = dict()
    vary_params['time_skip'] = np.arange(1, 72, 2, dtype = int)
    
    experiment['exp_name'] = 'x-coord_reconstr_Time_lag_1-70_ic_50'
    experiment['network_name'] = 'Lorenz_63'
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method_index'
    experiment['dependencies'] = False
    # Specify if we should perform testing
    experiment['testing'] = True
    
    fixed_params = {'delay_dimension' : 3,
                    'max_deg_monomials' : 5,
                    'ttrain': 100,
                    'ttest': 100,
                    'Nseeds': 25,
                    'reg_params' : 0,
                    'normalize_ts' : True,
                    'normalize_cols' : False,
                    'index' : np.array([0])
        }
    
    
    L2 = lab(experiment, vary_params, fixed_params)
    x_keys = 'time_skip'
    
    list_metrics = ['sigvals', 'VPT_test', 'abs_psd_test']
    
    res_dicts = L2.metrics_time_skip(list_metrics, x_keys = x_keys)
    
    tls.fig_x_coord_time_skip(res_dicts, 
                              vary_params['time_skip'],
                              plot_dict = None,
                              filename = experiment['exp_name'] 
                              )
    
    
# %%
def fig_cond_num_chebyshev_poly_deg():
    #This script test the following experiment:
    '''
    Consider 25 different initial conditions. 
    Vary the maximum degree of Chebyshev polynomial
    Fix the Ntrain = [5000]
    
    time_skip = 1
    Ntest = 100
    and the step size $\Delta t = 0.01$.    
    '''    
    
    experiment = dict()
    
    vary_params = dict()
    
    vary_params['max_deg_monomials'] = np.arange(2, 11, 1, dtype = int)
    
    experiment['exp_name'] = 'ill_cond_max_deg_chebyshev'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment['testing'] = False
    
    fixed_params = {'reg_params' : 0,
                    'ttrain' : 50,
                    'ttest' : 1,
                    'Nseeds' : 25,
                    'normalize_ts' : True,
                    'normalize_cols' : True,
                    'use_chebyshev' : True,
                    'use_canonical' : False,
        }
    
    
    L = lab(experiment, vary_params, fixed_params)
    exps_dict = L.exp_dict
    res = L.metrics_poly_deg(['sigvals'], curve_keys=None)
    x_axis = L.vary_params['max_deg_monomials']
    tls.plot_cond_num_poly_deg(res['sigvals'], 
                               x_axis, 
                               plot_dict = {'label':''},
                               filename = experiment['exp_name'])
    
    