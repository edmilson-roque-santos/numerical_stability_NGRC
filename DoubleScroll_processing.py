"""
Processing the Double Scroll system 
analysis of the time series for plotting a few results.

Created on Tue Aug  5 08:31:26 2025

@author: Edmilson Roque dos Santos
"""

import matplotlib.pyplot as plt
import numpy as np
import os 

from scipy import spatial as ss
from scipy import special
from scipy.linalg import subspace_angles
from scipy.signal import argrelextrema

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
ttrain = 300
ttest = 1
seed = 1
#Method of numerical integrating the differential equation
method = 'RK45'
# Fine sampling of the time series
dt_fine = 0.01
#============================##============================##============================#
#Generate synthetic data
ts_sgn = sgn(dt, ttrain, ttest, 
             delay_dimension, 
             time_skip,                                  
             trans_t = 5, 
             normalize = True,
             seed = seed,
             method = method,
             dt_fine = dt_fine)

folder = 'data/input_data/'

# Filename to save the time series. 
if method == 'Euler':
    filename = 'DoubleScroll_ts_Euler_{}_{}_{}.txt'.format(ttrain+ttest, 0.01, seed)
if method == 'RK45':
    filename = 'DoubleScroll_ts_RK45_{}_{}_{}.txt'.format(ttrain+ttest, dt_fine, seed)

ts_filename = folder+filename
ts_sgn.generate_signal(parametric_DoubleScroll, 
                       np.array([1.2, 3.44, 0.193, 11.6, 2.25*1e-5]),
                       ts_filename,
                       subsampling=True)

X_t_train, X_t_test = ts_sgn.X_t_train, ts_sgn.X_t_test
u_t_train, s_t_train = ts_sgn.u_t_train, ts_sgn.s_t_train
t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
#============================##============================##============================#
#Plot the auto correlation function - non normalized. In this way, 
#it looks the inner products that appear on the library matrix.
plot_auto_correlation = True
if plot_auto_correlation:
    index = [0, 1, 2]
    tls.plot_corr(X_t_train[:, index], index = index, normalize = True,
                  id_xlim=50,
                  fit = True)

#%%
#============================##============================##============================#
#Mutual information for the input time series     
    
def ksg(data, neig, center=True, borders=True):
    """
    Mutual Information using KSG estimators in method 2 dim I^(2) (X,Y)
    Based on the code of Dr. Ozge Canli.
    
    Args:
        data:
        neig: number of neighbors
        center: including center point or not
        borders: including border point or not

    Returns:

    """
    x = data[:, [0]]
    y = data[:, [1]]
    tree = ss.cKDTree(data)  # 2dim-tree
    tree_x = ss.cKDTree(x)  # 1dim-tree
    tree_y = ss.cKDTree(y)  # 1dim-tree
    n, p = data.shape  # number of points, p is the dim of point
    dist_2d, ind_2d = tree.query(data, neig + 1, p=float('inf'))
    Neigh_sum = 0.
    for i in range(n):
        e_x, e_y = np.max(np.fabs(np.tile(data[i, :], (neig + 1, 1)) - data[ind_2d[i, :]]), 0)

        if borders:

            nx = tree_x.query_ball_point([data[i, 0]], e_x, p=float('inf'))
            ny = tree_y.query_ball_point([data[i, 1]], e_y, p=float('inf'))
        else:
            nx = tree_x.query_ball_point([data[i, 0]], e_x - 1e-15, p=float('inf'))
            ny = tree_y.query_ball_point([data[i, 1]], e_y - 1e-15, p=float('inf'))

        if center:
            Neigh_sum += (special.digamma(len(nx)) + special.digamma(len(ny))) / n  # including center point
        else:
            Neigh_sum += special.digamma(len(nx) - 1) + special.digamma(len(ny) - 1) / n  # not including center point

    return special.digamma(neig) - (1 / neig) - Neigh_sum + special.digamma(n)

def mutual_info(X_t, index, id_xlim = 250):
    k = 10 # number of neighbors
    
    Td_array = np.arange(1, id_xlim, 2)
    I_array = np.zeros([len(Td_array), len(index)])
    
    for id_ in index:
        data = X_t[:, id_]
        
        for counter, Td in enumerate(Td_array):
            first_signal = data[:-Td].reshape(-1, 1)
            second_signal = data[Td:].reshape(-1, 1)
            s = np.hstack((first_signal, second_signal))
        
            I = ksg(s, k, borders=False)
            I_array[counter, id_] = I    
    
    return I_array, Td_array
    
def plot_MI(X_t, index, id_xlim = 250):
        
    I_array, Td_array = mutual_info(X_t, index, id_xlim)
    
    fig, ax = plt.subplots(len(index), 1, sharex = True, dpi = 300, figsize = (5, 5))
    
    for id_ in index:
        ax[id_].plot(Td_array, I_array[:, id_])
        
        peaks = argrelextrema(I_array[:, id_], np.less)
        
        print('Minimum MI for index', id_, Td_array[peaks[0].min()])
        
        ax[id_].vlines(Td_array[peaks[0]], 0, 3)
        
        ax[id_].set_ylabel(r"MI")
    ax[id_].set_xlabel(r"$\tau$")  


#Plot the Mutual information wrt time lag 
plot_MI_minimum = True
if plot_MI_minimum:
    index = [0, 1, 2]
    plot_MI(X_t_train[:, index], index = index, id_xlim = 50)
#%%
#Embedding dimension for the input time series
# Code based on Dr. Ozge Canli to compute the embedding dimension
# using false nearest neighbors.
def data_construction(data, de_max, Td):
    # data: input
    # de_max = max_embedding time delay
    # Td = time delay obtained by mutual information time lag
    """

    Args:
        data:
        de_max: max_embedding dimension
        Td:  time delay obtained by computing argument of first minimum of mutual information

    Returns: reconstructed phase space

    """
    n, p = data.shape  # n number of points, p is dimension
    M = n - (de_max - 1) * Td
    const_output = np.zeros([M, de_max])  # new reconstructed data shape
    for i in range(de_max):
        const_output[:, i] = data[np.arange(0, M) + i * Td].ravel()  # different functions from ravels
    return const_output

def fnn_array(data, de_max, Td, Ra=15):
    fnn_empty = np.zeros([de_max, 1])
    for i in range(de_max):
        co = data_construction(data, i + 1, Td)
        n, p = co.shape
        M = n - ((i + 1) * Td)
        co = co[:M]
        tree = ss.cKDTree(co)
        dist, ind = tree.query(co, 2, p=2)  # nearest neighbor k=2, in euclid distance(p=2)
        dist_one = dist[:M, 1]
        ind_one = ind[:M, 1]
        # todo: check that if part is same inside and outside
        if (i == 0):
            #Rd = dist_one
            difference = np.abs(data[np.arange(0, M) + (i + 1) * Td] - data[ind_one + (i + 1) * Td])
            fnn_empty[i, :] = np.count_nonzero((difference.T / dist_one) > Ra)
        else:
            difference = np.abs(data[np.arange(0, M) + (i + 1) * Td] - data[ind_one + (i + 1) * Td])
            fnn_empty[i, :] = np.count_nonzero((difference.T / dist_one) > Ra)

        #percentage = Rd / difference

    return fnn_empty, (fnn_empty / fnn_empty[0, :])

   
def plot_fnn(X_t, de_max = 10, time_lag = 15):
    result, percentage = fnn_array(X_t, de_max, time_lag)
    
    plt.figure(dpi = 300)
    plt.plot(np.arange(10)+1, percentage*100, 'bo-')
    plt.xlabel(r'Embedding dimension')
    plt.ylabel(r'Percentage of FNN')
    plt.title(r"False Nearest Neighbors of Double Scroll attractor", fontsize=12)
    plt.xticks([i for i in range(0, 11)])
    plt.yticks([i*10 for i in range(0, 11)])
    #plt.savefig('fnnlorenz.pdf')
    plt.show()

fnn_analysis = True

if fnn_analysis:
    index = [0]
    plot_fnn(X_t_train[:, index], time_lag = 13)
#%%
#============================##============================##============================#
def compute_angles_td(X_t, params, ts_max = 250):
    
    #Select the indices corresponding to the basis functions
    #For polynomial basis functions, 
    #id_t picks the basis functions evaluated at time t
    
    id_t = np.array([1, 2, 3, 4, 5, 6,
                     13, 14, 18]) - 1
    
    #id_tau picks the basis functions evaluated at the delayed coordinates
    id_tau = np.array([7, 8, 9, 10, 11, 12,
                       25, 26, 27]) - 1
    
    id_union = np.append(id_t, id_tau)
    
    #id_t = np.array([1, 2, 3, 4, 5]) - 1
    #id_tau = np.array([7, 8, 9, 10, 11]) - 1
    
    ts_vec = np.arange(1, ts_max + 1, 2, dtype = int)

    angles_ttau = np.zeros((ts_vec.shape[0], id_t.shape[0]))
    angles_cross = np.zeros((ts_vec.shape[0], id_t.shape[0]))

    for cter, ts in enumerate(ts_vec):
        ## Training phase
        RC = ng(params, 
                delay = params['delay_dimension'], 
                time_skip = ts,
                ind_term = True)
        
        #ids = np.arange(0, X_t.shape[1] - ts_max + ts, 1, dtype = int)
        R_ = RC.run(X_t.T)
    
        R = R_[1:, :] 
        
        R_t = R[id_t, :]
        R_tau = R[id_tau, :]
        
        ids = np.arange(0, R.shape[0], 1, dtype = int)
        mask = np.isin(ids, id_union)
    
        #crossed terms corresponds to the basis functions that 
        #have multiplication of current and delayed states
        R_ttau = R[~mask, :]
        
        angles_ttau[cter, :] = subspace_angles(R_t.T, R_tau.T)
        angles_cross[cter, :] = subspace_angles(R_t.T, R_ttau.T)
        
    return angles_ttau, angles_cross, ts_vec

def plot_angles(angles_ttau, angles_cross, ts_vec):
    
    
    fig, ax = plt.subplots(2, 1, sharex = True, dpi = 300, figsize = (5, 5))
    
    cmap = plt.get_cmap('bone', 12)
    
    col = cmap(range(angles_ttau.shape[1]))
    
    for id_, i in enumerate(angles_ttau[0, :]):
        
        ax[0].plot(ts_vec, angles_ttau[:, id_], '-',
                   color = col[id_])
        ax[1].plot(ts_vec, angles_cross[:, id_], '-',
                   color = col[id_])
    
    ax[0].hlines(0.0, ts_vec[0], ts_vec[-1], 
                 linestyle = 'dashed', 
                 color = "tab:red",
                 alpha = 1.0)
    
    ax[1].hlines(0.0, ts_vec[0], ts_vec[-1], 
                 linestyle = 'dashed', 
                 color = "tab:red",
                 alpha = 1.0)

    y = [0, np.pi/4, np.pi/2]
    ylabels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$']
    ax[0].set_yticks(y, ylabels, fontsize = 18)
    ax[1].set_yticks(y, ylabels, fontsize = 18)
        
    ax[1].set_xlabel(r"$\tau$")  
    #ax[1].set_ylabel(r"$\theta_{\mathbf{A} \mathbf{B}}$, principal angles")

    return
#%%
#============================##============================##============================#
############# Construct the parameters dictionary ##############
parameters = dict()

degree = 2
parameters['exp_name'] = 'processin_DoubleScroll'
parameters['network_name'] = 'DoubleScroll'
parameters['Nseeds'] = 1
parameters['random_seed'] = 1
parameters['max_deg_monomials'] = degree
parameters['expansion_crossed_terms'] = True

parameters['use_lebesgue'] = False
parameters['use_kernel'] = True
parameters['noisy_measurement'] = False
parameters['use_canonical'] = True
parameters['normalize_cols'] = False
parameters['use_orthonormal'] = False
parameters['single_density'] = False
parameters['cluster_density'] = True

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
    
parameters['use_chebyshev'] = False
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
    
plot_principal_angles = False

if plot_principal_angles:    
    angles_ttau, angles_cross, ts_vec = compute_angles_td(X_t_train, params, ts_max = 150)
    
    plot_angles(angles_ttau, angles_cross, ts_vec)