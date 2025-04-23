"""
Script for running a few test varying parameters

Created on Wed Jan 22 10:34:11 2025

@author: Edmilson Roque dos Santos
"""


import numpy as np

from main.simulation import simulation as sim
from main.lab import lab
import main.tools as tls

# %%
def run_test(exp = False):
    experiment = dict()
    
    vary_params = dict()
    vary_params['dt'] = np.array([0.01, 0.1])
    
    experiment['exp_name'] = 'test_0'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    
    #fixed_params = { }
    
    if exp:
        S = sim(experiment, vary_params)
        exps_dict = S.run()
    
    return exps_dict 
# %%
def t_deg_reg(exp = False):
    
    #This script test the following experiment:
    '''
    Consider ten different initial conditions. 
    Fix the number of data points to Ntrain = 10000 
    Ntest = 10000
    and the step size $\Delta t = 0.01$.
    Fix the number of delay dimensions to k = 1.
    Let us consider two variations:
        different maximum degree of polynomials.
        different regularizer parameters.
    
    '''    
    
    experiment = dict()
    
    vary_params = dict()
    vary_params['max_deg_monomials'] = np.arange(1, 6, 1, dtype = int)
    vary_params['reg_params'] = np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 
                                          1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    
    experiment['exp_name'] = 'deg_1-5-reg_-12_1e3'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    
    fixed_params = {'ttrain': 100,
                    'ttest': 100,
                    'Nseeds': 10
        }
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
    else:
        
        L = lab(experiment, vary_params, fixed_params)
        x_keys = 'reg_params'
        panel_keys = ['max_deg_monomials']
        
        
        list_metrics = ['VPT_test', 'zmax_test', 'abs_psd_test']
        L.plot_fig_metrics_reg(list_metrics, x_keys = 'reg_params', 
                               curve_keys = ['max_deg_monomials'],
                               filename = experiment['exp_name']+'metric_reg')#
        
        isbounded_dict = L.get_isbounded()
        
        subdict = dict()
        subdict['max_deg_monomials'] = np.arange(2, 6, 1, dtype = int)
        subdict['reg_params'] = [1e-12]
        res = L.results_sigvals(subdict, x_keys = 'max_deg_monomials',
                                filename = experiment['exp_name']+'_sigvals')    
        
        exps_dict = L.exp_dict
    return exps_dict 
# %%
def euler_ill_cond(exp = False):
    
    #This script test the following experiment:
    '''
    Consider 100 different initial conditions. 
    Vary the delay dimension
    Vary the maximum degree of polynomial
    Vary the Ntrain = [5000]
    
    time_skip = 1
    Ntest = 10000
    and the step size $\Delta t = 0.01$.
    
    '''    
    
    experiment = dict()
    
    vary_params = dict()
    vary_params['delay_dimension'] = np.arange(1, 5, 1, dtype = int)
    vary_params['ttrain'] = np.array([50])
    vary_params['max_deg_monomials'] = np.arange(2, 5, 1, dtype = int)
    
    experiment['exp_name'] = 'Norm_ill_cond_delay_max_deg_Ntrain'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment['testing'] = False
    
    fixed_params = {'reg_params' : 0,
                    'ttest' : 1,
                    'Nseeds' : 100,
                    'normalize_ts' : True,
                    'normalize_cols' : True,
        }
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
    else:
        
        L = lab(experiment, vary_params, fixed_params)
        exps_dict = L.exp_dict
        
        # Script for plotting Figure of cond number for different experiments
        res_dicts = dict()
        labels_dict = dict()
        titles_dict = dict()
        # Extract the experiment of delay dimension
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
        titles_dict[x_keys] = r'Max degree $p = 2$'
        
        # Extract the experiment of maximum degree of polynomials
        subdict = dict()
        subdict['delay_dimension'] = [1]
        subdict['ttrain'] = np.array([50])
        subdict['max_deg_monomials'] = np.arange(2, 5, 1, dtype = int)
        
        x_keys = 'max_deg_monomials'
        res_dicts[x_keys] = L.results_sigvals(subdict, x_keys,
                                              filename = None)
        labels = []
        for id_key, key in enumerate(res_dicts[x_keys].keys()):
            labels.append(r'$p = {}$'.format(key))
        
        labels_dict[x_keys] = labels
        titles_dict[x_keys] = r'Delay dimension $k = 1$'
        
        plot_figure = True
        if plot_figure:
            tls.fig_cond_number_delay_poly(res_dicts, labels_dict, titles_dict,
                                           filename = experiment['exp_name']+'_{}'.format(subdict['ttrain'][0]))
            
        # Checking if the delay dimension larger than 1 is already ill conditioned.
        # The answer is yes! The condition number if concentrated at 10^16
        # Extract the experiment of delay dimension
        subdict = dict()
        subdict['delay_dimension'] = [2]
        subdict['ttrain'] = np.array([50])
        subdict['max_deg_monomials'] = [2]
        
        x_keys = 'delay_dimension'
        res = L.results_sigvals(subdict, x_keys,
                                filename = None,
                                plot_results = True)
        
        
    return exps_dict 
# %%
def euler_ill_cond_poly_degree(exp = False):
    
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
    
    
    S = sim(experiment, vary_params, fixed_params)
    exps_dict = S.run()
    
    return exps_dict 
# %%
def euler_ill_cond_chebyshev_poly_degree(exp = False):
    
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
    
    
    S = sim(experiment, vary_params, fixed_params)
    exps_dict = S.run()
    
    return exps_dict 
# %%
def euler_ill_cond_time_skip(exp = False):
    
    #This script test the following experiment:
    '''
    Consider 100 different initial conditions. 
    Vary the delay dimension
    Vary the maximum degree of polynomial
        
    time_skip = 10
    Ntrain = 5000
    Ntest = 10000
    and the step size $\Delta t = 0.01$.
        
    '''    
    
    experiment = dict()
    
    vary_params = dict()
    vary_params['delay_dimension'] = np.arange(1, 5, 1, dtype = int)
    vary_params['max_deg_monomials'] = np.arange(2, 5, 1, dtype = int)
    
    experiment['exp_name'] = 'ill_cond_delay_max_deg_time_skip'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment['testing'] = False
    
    fixed_params = {'reg_params' : 0,
                    'time_skip' : 10,
                    'ttrain' : 50,
                    'ttest' : 100,
                    'Nseeds' : 100
        }
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
    else:
        
        L = lab(experiment, vary_params, fixed_params)
        exps_dict = L.exp_dict
        
        # Script for plotting Figure of cond number for different experiments
        res_dicts = dict()
        labels_dict = dict()
        titles_dict = dict()
        # Extract the experiment of delay dimension
        subdict = dict()
        subdict['delay_dimension'] = np.arange(1, 5, 1, dtype = int)
        subdict['max_deg_monomials'] = [2]
        
        x_keys = 'delay_dimension'
        res_dicts[x_keys] = L.results_sigvals(subdict, x_keys,
                                              filename = None)
        
        labels = []
        for id_key, key in enumerate(res_dicts[x_keys].keys()):
            labels.append(r'$k = {}$'.format(key))
        labels_dict[x_keys] = labels
        titles_dict[x_keys] = r'Max degree $p = 2$'
        
        # Extract the experiment of maximum degree of polynomials
        subdict = dict()
        subdict['delay_dimension'] = [1]
        subdict['max_deg_monomials'] = np.arange(2, 5, 1, dtype = int)
        
        x_keys = 'max_deg_monomials'
        res_dicts[x_keys] = L.results_sigvals(subdict, x_keys,
                                              filename = None)
        labels = []
        for id_key, key in enumerate(res_dicts[x_keys].keys()):
            labels.append(r'$p = {}$'.format(key))
        
        labels_dict[x_keys] = labels
        titles_dict[x_keys] = r'Delay dimension $k = 1$'
        
        plot_figure = True
        if plot_figure:
            tls.fig_cond_number(res_dicts, labels_dict, titles_dict,
                                filename = experiment['exp_name']+'_{}'.format(fixed_params['time_skip']))
        
    return exps_dict 
# %%
def skip_measures(exp = False):
    
    #This script test the following experiment:
    '''
    Consider ten different initial conditions. 
    Fix the number of data points to Ntrain = 5000 
    Ntest = 10000
    and the step size $\Delta t = 0.01$.
    Fix the number of delay dimensions to k = 2.
    Fix the maximum degree of polynomial
    regularizer parameter = 0
    Let us consider two variations:
        different time skips
        
    '''    
    
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
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
    else:
        L = lab(experiment, vary_params, fixed_params)
        x_keys = 'time_skip'
        
        list_metrics = ['sigvals']#, 'VPT_test', 'zmax_test', 'abs_psd_test'
        
        res_dict = L.metrics_time_skip(list_metrics, x_keys = x_keys)
        
        x_axis = vary_params[x_keys]
        #tls.plot_fig_metrics_time_skip(res_dict, x_axis, 
        #                               filename = experiment['exp_name']+'metric_time_skip',
        #                               reference_value = True)
        
        list_metrics = ['sigvals']
        L.plot_fig_metrics_time_skip(list_metrics, x_keys = x_keys,
                                     filename = None)        
        
        exps_dict = L.exp_dict
    return exps_dict 
# %%
def x_coord_cond_decay_time_lag(exp = False):
    
    #This script test the following experiment:
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
                    'normalize_cols' : False,
                    'index' : np.array([0])
        }
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
    else:
        L = lab(experiment, vary_params, fixed_params)
        x_keys = 'time_skip'
        
        list_metrics = ['sigvals']#, 'VPT_test', 'zmax_test', 'abs_psd_test'
        
        res_dict = L.metrics_time_skip(list_metrics, x_keys = x_keys)
        
        x_axis = vary_params[x_keys]
        #tls.plot_fig_metrics_time_skip(res_dict, x_axis, 
        #                               filename = experiment['exp_name']+'metric_time_skip',
        #                               reference_value = True)
        
        list_metrics = ['sigvals']
        L.plot_fig_metrics_time_skip(list_metrics, x_keys = x_keys,
                                     filename = None)        
        
        exps_dict = L.exp_dict
    return exps_dict 
# %%
def x_coord_metrics_time_lag(exp = False):
    
    #This script test the following experiment:
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
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
    else:
        L = lab(experiment, vary_params, fixed_params)
        x_keys = 'time_skip'
        
        list_metrics = ['sigvals']#, 'VPT_test', 'zmax_test', 'abs_psd_test'
        
        res_dict = L.metrics_time_skip(list_metrics, x_keys = x_keys)
        
        x_axis = vary_params[x_keys]
        #tls.plot_fig_metrics_time_skip(res_dict, x_axis, 
        #                               filename = experiment['exp_name']+'metric_time_skip',
        #                               reference_value = True)
        
        list_metrics = ['sigvals']
        L.plot_fig_metrics_time_skip(list_metrics, x_keys = x_keys,
                                     filename = None)        
        
        exps_dict = L.exp_dict
    return exps_dict 
# %%

def metrics_k_1(exp = False):
    
    #This script test the following experiment:
    '''
    Consider 50 different initial conditions. 
    Fix the number of data points to Ntrain = 5000 
    Ntest = 10000
    and the step size $\Delta t = 0.01$.
    Fix the number of delay dimensions to k = 1.
    Let us consider two variations:
        Maximum degree = 2.
        Regparam = 0.
    
    '''    
    
    experiment = dict()
    
    vary_params = dict()
    vary_params['max_deg_monomials'] = np.array([2])
    
    experiment['exp_name'] = 'ref_k_1_normalized'
    experiment['network_name'] = 'Lorenz_63'
    experiment['dependencies'] = False
    # Specify the script to use for the test.
    experiment['script'] = 'ngrc_euler_method'
    # Specify if we should perform testing
    experiment['testing'] = True
    
    fixed_params = {'ttrain': 50,
                    'ttest': 100,
                    'Nseeds': 50,
                    'reg_params' : 0,
                    'normalize_ts' : False,
                    'normalize_cols' : True
        }
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
    else:
        
        L = lab(experiment, vary_params, fixed_params)
                
        list_metrics = ['sigvals', 'VPT_test', 'zmax_test', 'abs_psd_test']
        for metric in list_metrics:
            L.get_stats(metric, True)
        
        exps_dict = L.exp_dict
    return exps_dict 
# %%
def reg_test(exp = False):
    
    #This script test the following experiment:
    '''
    Consider 50 different initial conditions. 
    Fix the number of data points to Ntrain = 5000 
    Ntest = 10000
    and the step size $\Delta t = 0.01$.
    Fix the number of delay dimensions to k = 2.
    Fix the time lag s = 15.
    
        different regularizer parameters.
    
    '''    
    
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
    
    
    if exp:
        S = sim(experiment, vary_params, fixed_params)
        exps_dict = S.run()
        
        S1 = sim(experiment1, vary_params1, fixed_params1)
        exps_dict1 = S1.run()
        
        S2 = sim(experiment2, vary_params2, fixed_params2)
        exps_dict2 = S2.run()

        return exps_dict, exps_dict1, exps_dict2
    
    else:
        
        L = lab(experiment, vary_params, fixed_params)
        x_keys = 'reg_params'
        panel_keys = ['max_deg_monomials']
        
        list_metrics = ['VPT_test', 'zmax_test', 'abs_psd_test']
        L.plot_fig_metrics_reg(list_metrics, x_keys = 'reg_params', 
                               curve_keys = None,
                               filename = experiment['exp_name']+'metric_reg')#
        
        
        exps_dict = L.exp_dict
     
        return exps_dict   
     
# %%
def sigvals_lgth_train(exp = False):
    
    #This script test the following experiment:
    '''
    Compute the condition number for different experiments
    Consider ten different initial conditions. 
    Vary the number of training data points
    Vary the step size $\Delta t$
    Ntest = 1
       
    '''    
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
    
    exps_dict = dict()
    if exp:
        S1 = sim(experiment1, vary_params1, fixed_params1)
        exps_dict['exp_1'] = S1.run()
        
        S2 = sim(experiment2, vary_params2, fixed_params2)
        exps_dict['exp_2'] = S2.run()
        
        return exps_dict
    
#%% 











