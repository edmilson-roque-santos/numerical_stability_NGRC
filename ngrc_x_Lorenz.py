"""
Reconstruction of x-coordinate of the Lorenz system.
Numerical scheme is the explicit forward Euler method. 

Created on Thu Apr 10 12:31:57 2025

@author: Edmilson Roque dos Santos
"""

import numpy as np
from scipy import linalg as LA
import os 

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
dt = 0.01
#Delayed coordinates and time skip
delay_dimension = 3
#Time skip between time points
time_skip = 15
#Warm up of the NGRC
warmup = (delay_dimension - 1)*time_skip
#Training and testing data
ttrain = 100
ttest = 25
seed = 1
#============================##============================##============================#
#Generate synthetic data
ts_sgn = sgn(dt, ttrain, ttest, 
             delay_dimension, 
             time_skip,                                  
             trans_t = 100, 
             normalize = True,
             seed = seed,
             method = 'Euler',
             dt_fine = 0.01)

folder = 'data/input_data/'
ts_filename = folder+'Lorenz_ts_Euler_{}_{}_{}.txt'.format(ttrain+ttest+warmup*dt, 0.01, seed)
ts_sgn.generate_signal(parametric_Lorenz, 
                       np.array([10.0, 8.0/3.0, 28]),
                       ts_filename,
                       subsampling=True)

index = np.array([0])
X_t_train, X_t_test = ts_sgn.X_t_train[:, index], ts_sgn.X_t_test[:, index]
u_t_train, s_t_train = ts_sgn.u_t_train[:, index], ts_sgn.s_t_train[:, index]
t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
#============================##============================##============================#
############# Construct the parameters dictionary ##############
parameters = dict()

degree = 5
parameters['exp_name'] = 'chebyshev_x_coord_reconstr'
parameters['network_name'] = 'Lorenz63'
parameters['Nseeds'] = 1
parameters['random_seed'] = 1
parameters['max_deg_monomials'] = degree
parameters['expansion_crossed_terms'] = True

parameters['use_lebesgue'] = False
parameters['use_kernel'] = True
parameters['noisy_measurement'] = False
parameters['use_canonical'] = False
parameters['use_chebyshev'] = True
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

if not parameters['use_qr']:
    #Readout matrix calculation
    # This calculation relies on computing finite difference
    if parameters['use_OMP']:
        #Calculates the coefficients using Orthogonal Matching Pursuit
        W_out = OMP(s_t_train - u_t_train, R.T, tol = 1e-8)
    else:
        reg_param = 0
        W_out = ridge(s_t_train.T - u_t_train.T, R, 
                      reg_param = reg_param,
                      solver = 'SVD')
    
    if params['normalize_cols']:
        W_out = W_out/params['norm_column']
    
    v_t_train = u_t_train.T + W_out @ R
 
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

v_t_test = RC.gen_autonomous_state(W_out, hist.T, t_test)
s_t_test, v_t_test, t_test = tls.select_bounded(X_t_test.T, v_t_test, t_test)
tls.plot_testing(s_t_test, v_t_test, t_test, 
                 transient_plot = -1, 
                 scale = 1/0.9056, 
                 fig = None)

tls.fig_x_coord_reconstr(s_t_test, v_t_test, t_test, dt, scale = 1/0.9056,
                         transient_plot = int(15/(0.9056*dt)), nperseg=int(1/dt)*5,
                         filename = parameters['exp_name'])

# Compute comparison wrt to the original vector
#c_matrix_true = get_true_coeff_Lorenz(params)

if parameters['use_orthonormal']:
    W_out_t = RC.params['R'] @ W_out.T/dt        
else:
    W_out_t = W_out.T/dt        
#error = np.absolute(W_out.T/dt - c_matrix_true)

# Dictionary associated with one specific experiment
exp_dict = dict()
exp_dict['W_out'] = W_out

# Import the time series generated by the NGRC
exp_dict['v_t_train'] = v_t_train
exp_dict['v_out_test'] = v_t_test
exp_dict['is_bounded'] = RC.is_bounded 
# Compute different error measures.

#During training 
exp_dict['NRMSE_train'] = NRMSE(s_t_train.T, exp_dict['v_t_train'])

exp_dict['VPT_train'] = valid_prediction_time(s_t_train.T, exp_dict['v_t_train'], dt)

tau_lyap = 1/0.9056
t_loss_train = np.linspace(0, t_train[-1]-tau_lyap, 50)
exp_dict['Loss_train'] = loss(s_t_train.T, exp_dict['v_t_train'], t_vec = t_loss_train, 
                              tau = tau_lyap, 
                              dt = dt)

exp_dict['abs_psd_train'] = kl_div_psd(s_t_train.T, 
                                          exp_dict['v_t_train'], 
                                          dt,
                                          nperseg=int(1/dt)*5)

#During testing 
exp_dict['NRMSE_test'] = NRMSE(s_t_test, exp_dict['v_out_test'])

exp_dict['VPT_test'] = valid_prediction_time(s_t_test, exp_dict['v_out_test'], dt)

t_loss_test = np.linspace(0, t_test[-1]-tau_lyap, 50)
exp_dict['Loss_test'] = loss(s_t_test, exp_dict['v_out_test'], t_vec = t_loss_test, 
                             tau = tau_lyap, 
                             dt = dt)

exp_dict['abs_psd_test'] = kl_div_psd(s_t_test, 
                                      exp_dict['v_out_test'], 
                                      dt,
                                      nperseg=int(1/dt)*5)

# Import the singular values of the matrix $R$
exp_dict['sigvals'] = s