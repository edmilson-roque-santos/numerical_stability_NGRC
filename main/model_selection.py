"""
Set of methods for model selection

Created on Thu Apr  6 14:04:17 2023

@author: Edmilson Roque dos Santos
"""

import numpy as np 
from scipy import linalg as LA
import scipy.interpolate as interpolate
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.linear_model import OrthogonalMatchingPursuit

import h5dict
from . import tools as tls
#============================##============================##============================#
#Error measures
#============================##============================##============================#
def NRMSE(s_t_true, v_t, transient_plot = 0):
    '''
    Normalized mean squared error given by ||s_true - v_t||_2^2/sqrt{||s_true||_2}

    Parameters
    ----------
    s_t_true : numpy array
        True component time series.
    v_t : numpy array
        Trained component time series.
    transient_plot : float, optional
        Transient time to be considered in order to plot the E. The default is 100.

    Returns
    -------
    MSE_ : float
        
    '''
    s_t_true, v_t = s_t_true[:, transient_plot:], v_t[:, transient_plot:]
    
    diff = s_t_true - v_t    
    num = np.mean(np.sum(np.square(diff), axis = 0))**(0.5)
    true = np.mean(np.sum(np.square(s_t_true), axis = 0))**(0.5)
    NMSE = num/(true)
    
    return NMSE

def valid_prediction_time(s_t_true, v_t, dt, 
                          threshold = 0.9,
                          scale = 1/0.9056):

    
    true = np.mean(np.sum(np.square(s_t_true), axis = 0))**(0.5)
    
    for id_t in range(s_t_true.shape[1]):
        
        diff = s_t_true[:, id_t] - v_t[:, id_t]
        num = np.sum(np.square(diff))**(0.5)
        
        NMSE = num/(true)
        
        VTP = id_t*dt/scale
        
        if NMSE > threshold:
            break
    
    return VTP
            
        

def E(s_t_true, v_t):
    '''
    E error given by |s_true - s_train| component-wise.

    Parameters
    ----------
    s_t_true : numpy array
        
    v_t : numpy array
        
    Returns
    -------
    diff : numpy array
        
    '''
    diff = np.absolute(s_t_true - v_t)
    
    return diff

def loss_t(s_t_true, v_t, t, tau, dt):
    '''
    Error measure that considers lyapunov time at time t.

    Parameters
    ----------
    s_t_true : numpy array
        
    v_t : numpy array
        
    t : float
        
    tau : float
        Time period to evaluate the loss function
    dt : float
        time step

    Returns
    -------
    float

    '''
    t_, tau_t = int(t/dt), int(tau/dt)
    t_f = t_ + tau_t

    if t_f > t_:

        diff = np.square(LA.norm(s_t_true[:, t_:t_f] - v_t[:, t_:t_f], axis = 0))
        
        
        true = (dt*np.sum(np.square(LA.norm(s_t_true[:, t_:t_f], axis = 0)))/tau)
        #train = dt*np.sum(LA.norm(v_t[:, t_:t_f], axis = 0))/tau
        norm = true# + train)
        
        return (dt*np.sum(diff)/tau)/(norm)

def loss(s_t_true, v_t, t_vec, tau, dt):
    '''
    Error measure that considers lyapunov time for every time at t_vec.

    Parameters
    ----------
    s_t_true : numpy array
        
    v_t : numpy array
        
    t_vec : numpy array
        
    lyap : float
        Maximum Lyapunov exponent of the system
    dt : float
        time step

    Returns
    -------
    float

    '''
    
    L = 0
    for t in t_vec:
        eps_1 = loss_t(s_t_true, v_t, t, tau, dt)
        L = L + eps_1
        
    L = L/t_vec.shape[0]

    return L


def l2_zmax_map(s_t, v_t, degree_int = 3):
    
    # Compute the spline of the succesive maxima map

    # Model
    z_max_v_t = tls.maximum_over_z(v_t)
    
    if z_max_v_t.shape[0] >= degree_int:  
    
        Z_max_v_t = np.array([z_max_v_t[0:-1], z_max_v_t[1:]]).T
        Z_max_v_t = Z_max_v_t[Z_max_v_t[:, 0].argsort()]
        
        try:
            #We compute the interpolation using BSplines
            tspl, c, k = interpolate.splrep(Z_max_v_t[:, 0], Z_max_v_t[:, 1], 
                                            s=0.02, 
                                            k=degree_int)
            
            spline_v_t = interpolate.BSpline(tspl, c, k, extrapolate=True)
        
        
            
            # Original trajectory
            z_max_s_t = tls.maximum_over_z(s_t)
                
            Z_max_s_t = np.array([z_max_s_t[0:-1], z_max_s_t[1:]]).T
            Z_max_s_t = Z_max_s_t[Z_max_s_t[:, 0].argsort()]
            
            #We compute the interpolation using BSplines
            tspl, c, k = interpolate.splrep(Z_max_s_t[:, 0], Z_max_s_t[:, 1], 
                                            s=0.02, 
                                            k=degree_int)
        
            spline_s_t = interpolate.BSpline(tspl, c, k, extrapolate=True)
        
        
            # Plotting the reference BSpline and the trajectory
            N = 1000
            xmin, xmax = np.max([z_max_s_t.min(), z_max_v_t.min()]), np.min([z_max_s_t.max(), z_max_v_t.max()])
            xx = np.linspace(xmin, xmax, N)
        
            return np.mean(np.absolute(spline_s_t(xx) - spline_v_t(xx)))
        
        except:
            print("The interpolation failed to proceed.")
            
            return -1
    else:
        return -1

def kl_div_psd(s_t, v_t, dt, window = 'hann', nperseg = None):
    id_f_stop = -1
    
    AE = np.zeros(s_t.shape[0])
    
    for id_row in range(s_t.shape[0]):
        f_s_t, psd_s_t = welch(s_t[id_row, :], 
                               window = window,
                               fs = 1/dt, nperseg=nperseg, scaling='density')
        
        f_v_t, psd_v_t = welch(v_t[id_row, :],  
                               window = window,
                               fs = 1/dt, nperseg=nperseg, scaling='density')

        AE[id_row] = entropy(psd_s_t[:id_f_stop], psd_v_t[:id_f_stop], base = 10)
    return AE.sum()

def error_wrt_true(c_est, c_true):
    
    return LA.norm(c_est - c_true)

#============================##============================##============================#
#Model selection
#============================##============================##============================#
def ridge(u_out, R, reg_param = 1e-6, solver = 'SVD'): 
    '''
    Ridge regression with regularizer parameter chosen by the user.
    
    min_W |W R - u_out|_2^2 + \lambda |W|_2^2
    
    Parameters
    ----------
    u_out : numpy array
        
    R : numpy array

    reg_param : float, optional
        Regularizer parameter. The default is 1e-6.

    Returns
    -------
    W_out : numpy array
        Solution of the Ridge regression.

    '''
    N = R.shape[0]
    M = R.shape[1]
    
    if solver == 'SVD':
        u, s, v = LA.svd(R.T, full_matrices = False,
                         lapack_driver='gesvd', compute_uv=True)
        
        factor = s/(s**2 + reg_param)
        W_out = (v.T @ (np.diag(factor) @ (u.T @ u_out.T))).T
    
    if solver == 'cholesky':
        try:
            W_out = LA.solve(R @ R.T + reg_param*np.identity(R.shape[0]), R @ u_out.T, 
                             assume_a = 'pos').T
        except:
            W_out = np.zeros((u_out.shape[0], N))
            
    if solver == 'LU':
        try:
            W_out = u_out @ (R.T @ LA.inv(R @ R.T + reg_param*np.identity(N)))
    
        except:
            W_out = np.zeros((u_out.shape[0], N))
    
    return W_out/np.sqrt(M)

#============================##============================##============================#
# Orthogonal Matching Pursuit
#============================##============================##============================#
def OMP(Y, R, n_nonzero_coefs = None, tol = 1e-8):

    M = R.shape[0]
    
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=tol, fit_intercept = False)
    omp.fit(R, Y)
    return omp.coef_/np.sqrt(M)




