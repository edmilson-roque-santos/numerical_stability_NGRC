"""
Comparison among algorithms to compute NGRC reconstruction.
Numerical scheme is the explicit forward Euler method. 

Created on Tue Jul 22 09:52:36 2025

@author: Edmilson Roque dos Santos
"""

import numpy as np
from scipy import linalg as LA
import os 
from prettytable import PrettyTable
import sympy as spy

from main.ng_reservoir import ng_reservoir as ng
import main.tools as tls
from main.model_selection import ridge, OMP, loss, NRMSE, valid_prediction_time, l2_zmax_map, kl_div_psd
from main.signal import signal as sgn
from main.dyn_sys.Lorenz import parametric_Lorenz, spy_Lorenz, Lorenz_system, get_true_coeff_Lorenz
from main.base_polynomial import pre_settings as pre_set 
from main.base_polynomial import poly_library as polb
#============================##============================##============================#
#Time step - sampling
dt = 0.025
#Delayed coordinates and time skip
delay_dimension = 1
#Time skip between time points
time_skip = 1
#Warm up of the NGRC
warmup = (delay_dimension - 1)*time_skip
#Training and testing data
ttrain = 50
ttest = 100
seed = 1
#============================##============================##============================#
#Generate synthetic data
ts_sgn = sgn(dt, ttrain, ttest, 
             delay_dimension, 
             time_skip,                                  
             trans_t = 100, 
             normalize = True,
             seed = seed,
             method = 'RK45',
             dt_fine = 0.01)

folder = 'data/input_data/'
ts_filename = folder+'Lorenz_ts_RK_{}_{}_{}.txt'.format(ttrain+ttest+warmup*dt, 0.01, seed)
ts_sgn.generate_signal(parametric_Lorenz, 
                       np.array([10.0, 8.0/3.0, 28]),
                       ts_filename,
                       subsampling=True)

X_t_train, X_t_test = ts_sgn.X_t_train, ts_sgn.X_t_test
u_t_train, s_t_train = ts_sgn.u_t_train, ts_sgn.s_t_train
t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
#============================##============================##============================#
############# Construct the parameters dictionary ##############
parameters = dict()

degree = 9
parameters['exp_name'] = 'compare_alg'
parameters['network_name'] = 'Lorenz63'
parameters['Nseeds'] = 1
parameters['random_seed'] = 1
parameters['max_deg_monomials'] = degree
parameters['expansion_crossed_terms'] = True

parameters['use_lebesgue'] = False
parameters['use_kernel'] = True
parameters['noisy_measurement'] = False
parameters['use_canonical'] = True
parameters['use_chebyshev'] = False
parameters['normalize_cols'] = False
parameters['use_orthonormal'] = False
parameters['single_density'] = True
parameters['cluster_density'] = False

if parameters['cluster_density']:
    # For Lorenz system: construct the cluster indices for computing the estimated prob. measure
    # In d dimension, definition of the clusters in d dimensions
    d = X_t_train.shape[1]
    cluster_list = [np.array([0, 1]), np.array([2])]
    
    # Extension of the cluster information for the embedded coordinates.
    parameters['cluster_list'] = []
    indices = np.arange(0, d*delay_dimension, 1, dtype = int)
    for cluster in cluster_list:
        cluster_ = []
        for id_node in cluster:
            for k in range(delay_dimension):
                cluster_.append(indices[id_node + k*d])
            
        parameters['cluster_list'].append(np.array(cluster_))

parameters['use_qr'] = False
parameters['use_OMP'] = False
# Input data to compute orthonormal basis functions. The input reference
# data is the training data

parameters['lower_bound'] = np.min(X_t_train)
parameters['upper_bound'] = np.max(X_t_train)
parameters['X_time_series_data'] = X_t_train
parameters['length_of_time_series'] = X_t_train.shape[0]
parameters['delay_dimension'] = delay_dimension
parameters['time_skip'] = time_skip
parameters['number_of_vertices'] = X_t_train.shape[1]*delay_dimension

params = parameters.copy()

if params['use_orthonormal']:
    out_dir_ortho_folder = 'data/orth_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(parameters['exp_name'],
                                                                    parameters['random_seed'],
                                                                    parameters['max_deg_monomials'],
                                                                    dt,
                                                                    delay_dimension,
                                                                    time_skip,
                                                                    ttrain,
                                                                    ttest)
    
    output_orthnormfunc_filename = out_dir_ortho_folder

    if not os.path.isfile(output_orthnormfunc_filename):
        params['orthnorm_func_filename'] = output_orthnormfunc_filename
        params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(params, save_orthnormfunc = True)
    if os.path.isfile(output_orthnormfunc_filename):
        params['orthnorm_func_filename'] = output_orthnormfunc_filename
              
    params['build_from_reduced_basis'] = False

## Training phase
RC = ng(params, 
        delay = delay_dimension, 
        time_skip = time_skip,
        ind_term = True)
R = RC.run(X_t_train.T)
params = RC.params

S = R @ R.T 
s = LA.svd(R.T, lapack_driver='gesvd', compute_uv=False)
cond_number = s.max()/s.min()

#Identify the solvers to be used during the simulation
# list to encode dictionary to save training and testing time series
solvers = ['SVD', 'cholesky', 'LU']

#Regularizer parameter
reg_param = 0

#Create the dictionary indexed by solver
res_dict = dict()

# Compute theta - closeness of fit
computing_thetas = True
thetas = PrettyTable(['Method', 'theta_x', 'theta_y', 'theta_z'])
cos_thetas = PrettyTable(['Method', 'ctheta_x', 'ctheta_y', 'ctheta_z'])
tan_thetas = PrettyTable(['Method', 'ttheta_x', 'ttheta_y', 'ttheta_z'])

for solver in solvers:
    
    #Readout matrix calculation
    # This calculation relies on computing finite difference    
    W_out = ridge(s_t_train.T - u_t_train.T, R, 
                      reg_param = reg_param, solver = solver)
    
    if params['normalize_cols']:
        W_out = W_out/params['norm_column']
    
    v_t_train = u_t_train.T + np.sqrt(R.shape[1])*W_out @ R
    
    
    if computing_thetas:  
        y = s_t_train.T - u_t_train.T
        theta = np.arcsin(LA.norm(y - np.sqrt(R.shape[1])*W_out @ R, axis = 1)/LA.norm(y, axis = 1))
        thetas.add_row([solver, theta[0], theta[1], theta[2]])
        cos_thetas.add_row([solver, np.cos(theta[0]), np.cos(theta[1]), np.cos(theta[2])])
        tan_thetas.add_row([solver, np.tan(theta[0]), np.tan(theta[1]), np.tan(theta[2])])
    
    #============================##============================##============================#
    ## Testing phase
    hist = X_t_train[-(warmup + 1):, :].copy()
    
    v_t_test = RC.gen_autonomous_state(W_out, hist.T, t_test)
    s_t_test, v_t_test, t_test = tls.select_bounded(X_t_test.T, v_t_test, t_test)
    
    res_dict[solver] = {'W_out': W_out,
                        'v_t_train': v_t_train,
                        'v_t_test': v_t_test,
                        's_t_test': s_t_test,
                        't_test': t_test[:int(50/(0.9056*dt))]}

tls.plot_solvers_traj(res_dict, scale = 1/0.9056, filename = None)
tls.plot_solvers_diff_traj(res_dict, scale = 1/0.9056, filename = None)
tls.plot_solvers_Wout(res_dict, filename = None)

diff_pairs = tls.diff_traj_solvers(res_dict)

if computing_thetas:  
    print(thetas)
    print(cos_thetas)
    print(tan_thetas)
