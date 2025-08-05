"""
Double Scroll
Comparison approximate vector field and NGRC reconstruction.
Numerical scheme is the Runge Kutta 4(5) method.

Created on Tue Aug  5 08:44:17 2025

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
from main.dyn_sys.Double_Scroll import parametric_DoubleScroll
from main.base_polynomial import pre_settings as pre_set 
from main.base_polynomial import poly_library as polb
#============================##============================##============================#
#Time step - sampling
dt = 0.25
#Delayed coordinates and time skip
delay_dimension = 1
#Time skip between time points
time_skip = 1
#Warm up of the NGRC
warmup = (delay_dimension - 1)*time_skip
#Training and testing data
ttrain = 100
ttest = 2500
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
ts_filename = folder+'DoubleScroll_ts_RK45_{}_{}_{}.txt'.format(ttrain+ttest+warmup*dt, 0.01, seed)
ts_sgn.generate_signal(parametric_DoubleScroll, 
                       np.array([1.2, 3.44, 0.193, 11.6, 2.25*1e-5]),
                       ts_filename,
                       subsampling=True)

X_t_train, X_t_test = ts_sgn.X_t_train, ts_sgn.X_t_test
u_t_train, s_t_train = ts_sgn.u_t_train, ts_sgn.s_t_train
t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
#============================##============================##============================#
############# Construct the parameters dictionary ##############
parameters = dict()

degree = 3
parameters['exp_name'] = 'computing thetas '
parameters['network_name'] = 'DoubleScroll'
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

if not parameters['use_qr']:
    #Readout matrix calculation
    # This calculation relies on computing finite difference
    if parameters['use_OMP']:
        #Calculates the coefficients using Orthogonal Matching Pursuit
        W_out = OMP(s_t_train - u_t_train, R.T, tol = 1e-8)
    else:
        reg_param = 0
        W_out = ridge(s_t_train.T - u_t_train.T, R, 
                      reg_param = reg_param, solver = 'SVD')
    
    if params['normalize_cols']:
        W_out = W_out/params['norm_column']
    
    v_t_train = u_t_train.T + np.sqrt(R.shape[1])*W_out @ R
 
if parameters['use_qr']:
   
    # Compute the qr decomposition 
    Q, r = LA.qr(np.sqrt(R.shape[1])*R.T)
    u_q, s_q, v_q = LA.svd(Q)
    u_r, s_r, v_r = LA.svd(R)
    W_out = np.zeros((X_t_train.shape[1], R.shape[0]))
    #Readout matrix calculation
    for id_node  in range(X_t_train.shape[1]):
        # This calculation relies on computing finite difference
        y = s_t_train[:, id_node] - u_t_train[:, id_node]
        W_out[id_node, :] = Q.T @ y 
        W_out[id_node, :] = (LA.inv(r) @ W_out[id_node, :].T)
        
    v_t_train = u_t_train.T + W_out @ R
    
tls.plot_training(s_t_train.T, v_t_train, t_train, scale = 1/0.9056)

#============================##============================##============================#
## Testing phase
hist = X_t_train[-(warmup + 1):, :].copy()

# This a small exercise to check how the model fits the testing data 
# Predicting one time forward in time.
predict_one_time = False
if predict_one_time:
    test_data = np.vstack((hist, X_t_test))
    R_test = RC.run(test_data.T)
    v_t_test_ = X_t_test.T + W_out @ R_test
    s_t_test_, v_t_test_, t_test = tls.select_bounded(X_t_test.T, v_t_test_, t_test)
    tls.plot_testing(s_t_test_, v_t_test_, t_test, 
                     transient_plot = -1, 
                     scale = 7.8125, 
                     fig = None)

v_t_test = RC.gen_autonomous_state(W_out, hist.T, t_test)
s_t_test, v_t_test, t_test = tls.select_bounded(X_t_test.T, v_t_test, t_test)
tls.plot_testing(s_t_test, v_t_test, t_test, 
                 transient_plot = -1, 
                 scale = 7.8125, 
                 fig = None)

if v_t_test.shape[0] == 3:
    tls.plot_2d_all_combinations(s_t_test, v_t_test)
    filename = params['exp_name']
    
    tls.fig_top_stat(s_t_test, v_t_test, dt, nperseg=int(1/dt)*50, filename = None) #filename+'_top_stats'
    tls.fig_compare(s_t_train.T, v_t_train, t_train[:int(7.8125*25/(dt))], 
                    s_t_test, v_t_test, t_test,
                    scale = 7.8125,
                    transient_plot = int(7.8125*15/(dt)), filename = None) #filename+'_compare'
    
if parameters['use_orthonormal']:
    W_out_t = RC.params['R'] @ W_out.T/dt        
else:
    W_out_t = W_out.T/dt        


computing_thetas = True

if computing_thetas:    
    thetas = PrettyTable(['Method', 'theta_x', 'theta_y', 'theta_z'])
    cos_thetas = PrettyTable(['Method', 'ctheta_x', 'ctheta_y', 'ctheta_z'])
    tan_thetas = PrettyTable(['Method', 'ttheta_x', 'ttheta_y', 'ttheta_z'])
    
    y = s_t_train.T - u_t_train.T
    
    # Cholesky
    W_out_cho = ridge(s_t_train.T - u_t_train.T, R, 
                      reg_param = reg_param, solver = 'cholesky')
    
    theta_cho = np.arcsin(LA.norm(y - np.sqrt(R.shape[1])*W_out_cho @ R, axis = 1)/LA.norm(y, axis = 1))
    thetas.add_row(["cholesky", theta_cho[0], theta_cho[1], theta_cho[2]])
    cos_thetas.add_row(["cholesky", np.cos(theta_cho[0]), np.cos(theta_cho[1]), np.cos(theta_cho[2])])
    tan_thetas.add_row(["cholesky", np.tan(theta_cho[0]), np.tan(theta_cho[1]), np.tan(theta_cho[2])])
    
    # SVD
    W_out_svd = ridge(s_t_train.T - u_t_train.T, R, 
                      reg_param = reg_param, solver = 'SVD')

    theta_svd = np.arcsin(LA.norm(y - np.sqrt(R.shape[1])*W_out_svd @ R, axis = 1)/LA.norm(y, axis = 1))
    thetas.add_row(["svd",   theta_svd[0], theta_svd[1], theta_svd[2]])
    cos_thetas.add_row(["svd", np.cos(theta_svd[0]), np.cos(theta_svd[1]), np.cos(theta_svd[2])])
    tan_thetas.add_row(["svd", np.tan(theta_svd[0]), np.tan(theta_svd[1]), np.tan(theta_svd[2])])
    
    
    # LU
    W_out_lu = ridge(s_t_train.T - u_t_train.T, R, 
                      reg_param = reg_param, solver = 'LU')

    theta_lu = np.arcsin(LA.norm(y - np.sqrt(R.shape[1])*W_out_lu @ R, axis = 1)/LA.norm(y, axis = 1))
    thetas.add_row(["lu",   theta_lu[0], theta_lu[1], theta_lu[2]])
    cos_thetas.add_row(["lu", np.cos(theta_lu[0]), np.cos(theta_lu[1]), np.cos(theta_lu[2])])
    tan_thetas.add_row(["lu", np.tan(theta_lu[0]), np.tan(theta_lu[1]), np.tan(theta_lu[2])])
    
    
    print(thetas)
    print(cos_thetas)
    print(tan_thetas)

    diff1 = LA.norm(W_out_svd - W_out_cho, axis = 1) 
    diff2 = LA.norm(W_out_svd - W_out_lu, axis = 1)
    diff3 = LA.norm(W_out_cho - W_out_lu, axis = 1)
    np.max([diff1, diff2, diff3])