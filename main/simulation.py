"""
Class to run test of RC performance for selected fixed and varying parameters.

Created on Mon Jan 20 17:27:56 2025

@author: Edmilson Roque dos Santos
"""

from datetime import datetime
import itertools
from multiprocessing import Pool 
import numpy as np
from numpy.random import default_rng
from scipy import linalg as LA
import os
import sys

import h5dict

from .ng_reservoir import ng_reservoir as ng
from .model_selection import ridge, loss, NRMSE, valid_prediction_time, l2_zmax_map, kl_div_psd, error_wrt_true
from .signal import signal as sgn
from .dyn_sys.Lorenz import parametric_Lorenz, get_true_coeff_Lorenz
from .base_polynomial import pre_settings as pre_set 
from . import tools as tls
#============================##============================##============================#
#Pre-settings for saving hdf5 files
#============================##============================##============================#
def out_dir(exp_name): 
    '''
    Create the folder name for save comparison  
    locally inside results folder.

    Parameters
    ----------
    exp_name : str
        Filename.
    
    Returns
    -------
    out_results_direc : str
        Out results directory.

    '''       
    folder_name = 'results'
    out_results_direc = os.path.join(folder_name, exp_name)
    out_results_direc = os.path.join(out_results_direc, '')
    
    if os.path.isdir(out_results_direc) == False:
        try:
            os.makedirs(out_results_direc)
        except:
            'Folder has already been created'
    return out_results_direc

#==============================================================================#

def default_params():
   
    #Default parameters for running an instance of the sample code ng_rc_test
    df_params = dict() 
    
    #A few parameters to set the time series signal
    
    #Time step - sampling
    df_params['dt'] = 0.01
    df_params['dt_fine'] = 0.01
    #Delayed coordinates and time skip
    df_params['delay_dimension'] = 1
    #Time skip between time points
    df_params['time_skip'] = 1
    #Warm up of the NGRC
    df_params['warmup'] = (df_params['delay_dimension'] - 1)*df_params['time_skip']
    #Training and testing data
    df_params['ttrain'] = 100
    df_params['ttest'] = 25
    #Random seed identifier
    df_params['random_seed'] = 1
    #Number of seeds
    df_params['Nseeds'] = 1
    #Normalize the input time series
    df_params['normalize_ts'] = False
    
    #A few parameters to set the reconstruction method
    df_params['max_deg_monomials'] = 2
    df_params['use_canonical'] = True 
    df_params['use_orthonormal'] = False
    df_params['normalize_cols'] = False
    df_params['use_chebyshev'] = False
    df_params['reg_params'] = 1e-8
    df_params['solver_ridge'] = 'SVD'
    
    return df_params
    
class simulation():
    '''
    Class for simulating different experiments - combining parameters to be
    kept fixed while other are varied.
    '''
    
    def __init__(self, experiment, vary_params, fixed_params = None):
        '''
        

        Parameters
        ----------
        experiment : dict
            'exp_name' : str
                 Name of the experiment - it matches the folder name.
        fixed_params : dict
            dictionary containing parameters that are kept fixed 
            for a given simulation.
        vary_params : dict
            Dictionary containing the parameters that are varied 
            for a given simulation. Each value contains the list
            or array of values to be taken into account in a 
            cartesian product.
        exp : 
        
        
        Returns
        -------
        None.

        '''
                     
        if experiment['dependencies']:
            self.combination_tools = zip
        else:
            self.combination_tools = itertools.product    
        
        self.experiment = dict()
        self.experiment['script'] = experiment['script']
        self.experiment['testing'] = experiment['testing']
        
        self.parameters = default_params()
        self.parameters['exp_name'] = experiment['exp_name']
        self.parameters['network_name'] = experiment['network_name']
        
        if fixed_params is not None:
            for key in fixed_params.keys():
                self.parameters[key] = fixed_params[key]
                
        self.seed_vec = np.arange(1, self.parameters['Nseeds'] + 1, 1, dtype = int)
        
        self.vary_params = vary_params
       

#============================##============================##============================##============================#      
    def file_output(self, filename):
        '''
        Create a filename allocating to the correct folder.

        Parameters
        ----------
        filename : str
            Filename for output results.

        Returns
        -------
        str
            Updated filename allocating at the correct folder.

        '''
        #Filename for output results
        out_results_direc = out_dir(self.parameters['exp_name'])
        if filename is not None:
            return out_results_direc+filename+".hdf5"
        else:
            filename = 'dt:{}-d:{}-s:{}_rs:{}_'.format(self.parameters['dt'],
                                                 self.parameters['delay_dimension'],
                                                 self.parameters['time_skip'],
                                                 self.parameters['random_seed'])
            
            date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+".hdf5"
            
            return out_results_direc+filename+date
    
    def read_hdf5(self, fname):
        '''
        Read the hdf5 for a given experiment.

        Parameters
        ----------
        fname : str
            Filename of the hdf5 file.

        Returns
        -------
        dict
            Experiment dictionary.

        '''
        out_results_hdf5 = h5dict.File(fname, 'r')
        self.exp_dict = out_results_hdf5.to_dict()  
        out_results_hdf5.close()      
        
        print('Experiment has been run already using this filename!')
        return self.exp_dict
    
    def comb_fname(self, comb):
        '''
        Create a filename for a given combination 

        Parameters
        ----------
        comb : tuple
            Identifier of the parameter being varied.

        Returns
        -------
        filename : str
            Filename of the experiment.

        '''
        keys = list(self.vary_params.keys())
        filename = ''
        for i, key in enumerate(keys):
            filename = filename + '{}-{}_'.format(key, comb[i])
    
        return filename 
    
    def comb_string(self, comb, id_comb):
        '''
        Create a filename for a given combination 

        Parameters
        ----------
        comb : tuple
            Identifier of the parameter being varied.

        Returns
        -------
        filename : str
            Filename of the experiment.

        '''
        comb_name = []
        for i, key in enumerate(comb):
            if len('{}'.format(key)) > 10:
                comb_name.append('{}'.format(id_comb))
            else:                
                comb_name.append('{}'.format(key))
    
        return tuple(comb_name)
    
#============================##============================##============================##============================#   
    
    
#============================##============================##============================##============================#
#%%
    def set_rc(self, params):
        '''
        To train and test performance of a NGRC for the given parameters.
        It imports a fixed Lorenz time series generated using Runge-Kutta 4 (5) 
        with dt = 0.001 and max total time 1000.
        
        Parameters
        ----------
        params : dict
            Parameters.

        Returns
        -------
        exp_dict : dictionary
            W_out : numpy array
                Readout matrix 
            v_t_train : numpy array 
                Training NGRC trajectory for plotting training.
            is_bounded : boolean
                Check if the reconstructed dynamics is bounded.
            s_t_test : numpy array
                True trajectory within testing phase.
            v_out_drive : numpy array
                Reservoir output within testing phase.
            t_train : numpy array
                Time span evaluated within training phase.
            t_test : numpy array
                Time span evaluated within testing phase.
            sigvals : numpy array 
                Singular values of the library matrix evaluated on the training trajectory.
                
            results with different measures during training and testing - below {} corresponds to train,test:
              
            NRMSE_{} : normalized root mean square error
            VPT_{} : valid prediction time
            Loss_{} : loss
            abs_psd_train_{} : Kullback Leibler divergence between the power spectrum density
            zmax_{} : computing distance between the z-max map.
            
        '''
        #Time step - sampling
        dt = params['dt']
        #Delayed coordinates and time skip
        delay_dimension = params['delay_dimension']
        #Time skip between time points
        time_skip = params['time_skip']
        #Warm up of the NGRC
        warmup = (params['delay_dimension'] - 1)*params['time_skip']
        #Training and testing data
        ttrain = params['ttrain']
        ttest = params['ttest']
        #Random seed identifier
        seed = params['random_seed']
        #============================##============================##============================#
        #Generate synthetic data
        ts_sgn = sgn(dt, ttrain, ttest, 
                     delay_dimension, 
                     time_skip,                                  
                     trans_t = 100, 
                     normalize = True,
                     seed = seed)
        
        
        #TO DO
        # Make an automatic time series generation given a ttrain and ttest.
        
        folder = 'data/input_data/'
        ts_filename = folder+'Lorenz_ts_1000_0.001_{}.txt'.format(seed)
        ts_sgn.generate_signal(parametric_Lorenz, 
                               np.array([10.0, 8.0/3.0, 28]),
                               ts_filename)

        X_t_train, X_t_test = ts_sgn.X_t_train, ts_sgn.X_t_test
        u_t_train, s_t_train = ts_sgn.u_t_train, ts_sgn.s_t_train
        t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
        #============================##============================##============================#
        ############# Construct the parameters dictionary ##############
        reconstr_params = dict()

        reconstr_params['exp_name'] = params['exp_name']
        reconstr_params['network_name'] = params['network_name']
        reconstr_params['max_deg_monomials'] = params['max_deg_monomials']
        
        reconstr_params['use_canonical'] = params['use_canonical']
        reconstr_params['use_orthonormal'] = params['use_orthonormal']
        reconstr_params['single_density'] = False
        reconstr_params['use_chebyshev'] = params['use_chebyshev']

        reconstr_params['lower_bound'] = np.min(X_t_train)
        reconstr_params['upper_bound'] = np.max(X_t_train)

        reconstr_params['X_time_series_data'] =  X_t_train
        reconstr_params['length_of_time_series'] = X_t_train.shape[0]
        reconstr_params['delay_dimension'] = delay_dimension
        reconstr_params['time_skip'] = time_skip
        reconstr_params['number_of_vertices'] = delay_dimension*X_t_train.shape[1]

        if params['use_orthonormal']:
            out_dir_ortho_folder = 'orth_data/orth_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(reconstr_params['exp_name'],
                                                                            reconstr_params['random_seed'],
                                                                            reconstr_params['max_deg_monomials'],
                                                                            dt,
                                                                            delay_dimension,
                                                                            time_skip,
                                                                            ttrain,
                                                                            ttest)
            
            output_orthnormfunc_filename = out_dir_ortho_folder

            if not os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                reconstr_params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(reconstr_params, 
                                                                         save_orthnormfunc = True)
            if os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                      
            reconstr_params['build_from_reduced_basis'] = False

        ## Training phase
        RC = ng(reconstr_params, 
                delay = delay_dimension, 
                time_skip = time_skip,
                ind_term = True)
        R = RC.run(X_t_train.T)

        reg_param = params['reg_params']

        S = R @ R.T
        s = LA.svd(R.T, lapack_driver='gesvd', compute_uv=False)

        #Readout matrix calculation
        W_out = ridge(s_t_train.T - u_t_train.T, R, reg_param = reg_param)
        if reconstr_params['use_orthonormal']:
            M = s_t_train.shape[0]
            W_out = W_out/np.sqrt(M)

        v_t_train = u_t_train.T + W_out @ R
        #============================##============================##============================#
        ## Testing phase
        hist = X_t_train[-(warmup + 1):, :].copy()

        v_t_test = RC.gen_autonomous_state(W_out, hist.T, t_test)
        s_t_test, v_t_test, t_test = tls.select_bounded(X_t_test.T, v_t_test, t_test)
        
        # Dictionary associated with one specific experiment
        exp_dict = dict()
        exp_dict['W_out'] = W_out

        # Import the time series generated by the NGRC
        exp_dict['v_t_train'] = v_t_train
        exp_dict['v_out_test'] = v_t_test
        exp_dict['is_bounded'] = RC.is_bounded 
        exp_dict['t_train'] = t_train
        exp_dict['t_test'] = t_test
        # Compute different error measures.

        #During training 
        exp_dict['NRMSE_train'] = NRMSE(s_t_train.T, exp_dict['v_t_train'])

        exp_dict['VPT_train'] = valid_prediction_time(s_t_train.T, exp_dict['v_t_train'], dt)

        tau_lyap = 1/0.9056
        t_loss_train = np.linspace(0, t_train[-1]-tau_lyap, 50)
        exp_dict['Loss_train'] = loss(s_t_train.T, exp_dict['v_t_train'], t_vec = t_loss_train, 
                                      tau = tau_lyap, 
                                      dt = dt)

        exp_dict['zmax_train'] = l2_zmax_map(s_t_train.T, v_t_train)

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

        exp_dict['zmax_test'] = l2_zmax_map(s_t_test, v_t_test)

        exp_dict['abs_psd_test'] = kl_div_psd(s_t_test, 
                                              exp_dict['v_out_test'], 
                                              dt,
                                              nperseg=int(1/dt)*5
                                              )

        # Import the singular values of the matrix $R$
        exp_dict['sigvals'] = s
        
        # Import the many parameters for the given simulation. Easier to discover later on.
        exp_dict['script_params'] = params
        exp_dict['reconstr_params'] = reconstr_params
        
        return exp_dict
#%%    
    def ngrc_euler_method(self, params):
        '''
        To train and test performance of a NGRC for the given parameters.
        It creates dynamic time series generated using Euler method.
        
        Parameters
        ----------
        params : dict
            Parameters.

        Returns
        -------
        exp_dict : dictionary
            W_out : numpy array
                Readout matrix 
            v_t_train : numpy array 
                Training NGRC trajectory for plotting training.
            is_bounded : boolean
                Check if the reconstructed dynamics is bounded.
            s_t_test : numpy array
                True trajectory within testing phase.
            v_out_drive : numpy array
                Reservoir output within testing phase.
            t_train : numpy array
                Time span evaluated within training phase.
            t_test : numpy array
                Time span evaluated within testing phase.
            sigvals : numpy array 
                Singular values of the library matrix evaluated on the training trajectory.
                
            results with different measures during training and testing - below {} corresponds to train,test:
              
            NRMSE_{} : normalized root mean square error
            VPT_{} : valid prediction time
            Loss_{} : loss
            abs_psd_train_{} : Kullback Leibler divergence between the power spectrum density
            zmax_{} : computing distance between the z-max map.
            error_wrt_true : computes the difference between W_out and true coefficient.
            
        '''
        #Time step - sampling
        dt = params['dt']
        dt_fine = params['dt_fine']
        #Delayed coordinates and time skip
        delay_dimension = params['delay_dimension']
        #Time skip between time points
        time_skip = params['time_skip']
        #Warm up of the NGRC
        warmup = (params['delay_dimension'] - 1)*params['time_skip']
        #Training and testing data
        ttrain = params['ttrain']
        ttest = params['ttest']
        #Random seed identifier
        seed = params['random_seed']
        #============================##============================##============================#
        #Generate synthetic data
        ts_sgn = sgn(dt, ttrain, ttest, 
                     delay_dimension, 
                     time_skip,                                  
                     trans_t = 100, 
                     normalize = params['normalize_ts'],
                     seed = seed,
                     method = 'Euler',
                     dt_fine = params['dt_fine'])
        
        folder = 'data/input_data/'
        ts_filename = folder+'Lorenz_ts_Euler_{}_{}_{}.txt'.format(ttrain+ttest+warmup*dt, dt_fine, seed)
        ts_sgn.generate_signal(parametric_Lorenz, 
                               np.array([10.0, 8.0/3.0, 28]),
                               ts_filename,
                               subsampling=True)

        X_t_train, X_t_test = ts_sgn.X_t_train, ts_sgn.X_t_test
        u_t_train, s_t_train = ts_sgn.u_t_train, ts_sgn.s_t_train
        t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
        #============================##============================##============================#
        ############# Construct the parameters dictionary ##############
        reconstr_params = dict()

        reconstr_params['exp_name'] = params['exp_name']
        reconstr_params['network_name'] = params['network_name']
        reconstr_params['max_deg_monomials'] = params['max_deg_monomials']
        
        reconstr_params['use_canonical'] = params['use_canonical']
        reconstr_params['use_orthonormal'] = params['use_orthonormal']
        reconstr_params['normalize_cols'] = params['normalize_cols']
        reconstr_params['single_density'] = False
        reconstr_params['use_chebyshev'] = params['use_chebyshev']

        reconstr_params['lower_bound'] = np.min(X_t_train)
        reconstr_params['upper_bound'] = np.max(X_t_train)

        reconstr_params['X_time_series_data'] =  X_t_train
        reconstr_params['length_of_time_series'] = X_t_train.shape[0]
        reconstr_params['delay_dimension'] = delay_dimension
        reconstr_params['time_skip'] = time_skip
        reconstr_params['number_of_vertices'] = delay_dimension*X_t_train.shape[1]

        if params['use_orthonormal']:
            out_dir_ortho_folder = 'orth_data/orth_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(reconstr_params['exp_name'],
                                                                            reconstr_params['random_seed'],
                                                                            reconstr_params['max_deg_monomials'],
                                                                            dt,
                                                                            delay_dimension,
                                                                            time_skip,
                                                                            ttrain,
                                                                            ttest)
            
            output_orthnormfunc_filename = out_dir_ortho_folder

            if not os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                reconstr_params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(reconstr_params, 
                                                                         save_orthnormfunc = True)
            if os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                      
            reconstr_params['build_from_reduced_basis'] = False

        ## Training phase
        RC = ng(reconstr_params, 
                delay = delay_dimension, 
                time_skip = time_skip,
                ind_term = True)
        R = RC.run(X_t_train.T)
        reconstr_params = RC.params
        reg_param = params['reg_params']

        S = R @ R.T
        s = LA.svd(R.T, lapack_driver='gesvd', compute_uv=False)

        #Readout matrix calculation
        W_out = ridge(s_t_train.T - u_t_train.T, R, 
                      reg_param = reg_param, solver = params['solver_ridge'])
        
        if reconstr_params['normalize_cols']:
            W_out = W_out/reconstr_params['norm_column']

        v_t_train = u_t_train.T + W_out @ R
        
        # Dictionary associated with one specific experiment
        exp_dict = dict()
        exp_dict['W_out'] = W_out.T
        reconstr_params = RC.params
        c_matrix_true = get_true_coeff_Lorenz(reconstr_params)

        exp_dict['error_wrt_true'] = error_wrt_true(W_out.T/dt, c_matrix_true)
    
        # Import the singular values of the matrix $R$
        exp_dict['sigvals'] = s

        if self.experiment['testing']:
            #============================##============================##============================#
            ## Testing phase
            hist = X_t_train[-(warmup + 1):, :].copy()
    
            v_t_test = RC.gen_autonomous_state(W_out, hist.T, t_test)
            s_t_test, v_t_test, t_test = tls.select_bounded(X_t_test.T, v_t_test, t_test)
            
            # Import the time series generated by the NGRC
            exp_dict['v_t_train'] = v_t_train
            exp_dict['v_out_test'] = v_t_test
            exp_dict['is_bounded'] = RC.is_bounded 
            exp_dict['t_train'] = t_train
            exp_dict['t_test'] = t_test
            
            # Compute different error measures whenever the autonomous NGRC dynamics is bounded.
            
            if exp_dict['is_bounded']:
                #During training 
                exp_dict['NRMSE_train'] = NRMSE(s_t_train.T, exp_dict['v_t_train'])
        
                exp_dict['VPT_train'] = valid_prediction_time(s_t_train.T, exp_dict['v_t_train'], dt)
        
                tau_lyap = 1/0.9056
                t_loss_train = np.linspace(0, t_train[-1]-tau_lyap, 50)
                exp_dict['Loss_train'] = loss(s_t_train.T, exp_dict['v_t_train'], t_vec = t_loss_train, 
                                              tau = tau_lyap, 
                                              dt = dt)
        
                exp_dict['zmax_train'] = l2_zmax_map(s_t_train.T, v_t_train)
        
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
        
                exp_dict['zmax_test'] = l2_zmax_map(s_t_test, v_t_test)
        
                exp_dict['abs_psd_test'] = kl_div_psd(s_t_test, 
                                                      exp_dict['v_out_test'], 
                                                      dt,
                                                      nperseg=int(1/dt)*5
                                                      )
            else:
                exp_dict['NRMSE_train'] = -1
                exp_dict['NRMSE_test'] = -1
                exp_dict['VPT_train'] = -1
                exp_dict['VPT_test'] = -1 
                exp_dict['Loss_train'] = -1 
                exp_dict['Loss_test'] = -1 
                exp_dict['zmax_train'] = -1
                exp_dict['zmax_test'] = -1
                exp_dict['abs_psd_train'] = -1
                exp_dict['abs_psd_test'] = -1
                
            # Import the many parameters for the given simulation. Easier to discover later on.
            exp_dict['script_params'] = params
                
        return exp_dict
#%%    
    def ngrc_euler_method_index(self, params):
        '''
        To train and test performance of a NGRC for the given parameters.
        It creates dynamic time series generated using Euler method.
        
        Parameters
        ----------
        params : dict
            Parameters.

        Returns
        -------
        exp_dict : dictionary
            W_out : numpy array
                Readout matrix 
            v_t_train : numpy array 
                Training NGRC trajectory for plotting training.
            is_bounded : boolean
                Check if the reconstructed dynamics is bounded.
            s_t_test : numpy array
                True trajectory within testing phase.
            v_out_drive : numpy array
                Reservoir output within testing phase.
            t_train : numpy array
                Time span evaluated within training phase.
            t_test : numpy array
                Time span evaluated within testing phase.
            sigvals : numpy array 
                Singular values of the library matrix evaluated on the training trajectory.
                
            results with different measures during training and testing - below {} corresponds to train,test:
              
            NRMSE_{} : normalized root mean square error
            VPT_{} : valid prediction time
            Loss_{} : loss
            abs_psd_train_{} : Kullback Leibler divergence between the power spectrum density
            zmax_{} : computing distance between the z-max map.
            error_wrt_true : computes the difference between W_out and true coefficient.
            
        '''
        #Time step - sampling
        dt = params['dt']
        dt_fine = params['dt_fine']
        #Delayed coordinates and time skip
        delay_dimension = params['delay_dimension']
        #Time skip between time points
        time_skip = params['time_skip']
        #Warm up of the NGRC
        warmup = (params['delay_dimension'] - 1)*params['time_skip']
        #Training and testing data
        ttrain = params['ttrain']
        ttest = params['ttest']
        #Random seed identifier
        seed = params['random_seed']
        #============================##============================##============================#
        #Generate synthetic data
        ts_sgn = sgn(dt, ttrain, ttest, 
                     delay_dimension, 
                     time_skip,                                  
                     trans_t = 100, 
                     normalize = params['normalize_ts'],
                     seed = seed,
                     method = 'Euler',
                     dt_fine = params['dt_fine'])
        
        folder = 'data/input_data/'
        ts_filename = folder+'Lorenz_ts_Euler_{}_{}_{}.txt'.format(ttrain+ttest+warmup*dt, dt_fine, seed)
        ts_sgn.generate_signal(parametric_Lorenz, 
                               np.array([10.0, 8.0/3.0, 28]),
                               ts_filename,
                               subsampling=True)
        index = params['index']
        X_t_train, X_t_test = ts_sgn.X_t_train[:, params['index']], ts_sgn.X_t_test[:, params['index']]
        u_t_train, s_t_train = ts_sgn.u_t_train[:, params['index']], ts_sgn.s_t_train[:, params['index']]
        t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
        #============================##============================##============================#
        ############# Construct the parameters dictionary ##############
        reconstr_params = dict()

        reconstr_params['exp_name'] = params['exp_name']
        reconstr_params['network_name'] = params['network_name']
        reconstr_params['max_deg_monomials'] = params['max_deg_monomials']
        
        reconstr_params['use_canonical'] = params['use_canonical']
        reconstr_params['use_orthonormal'] = params['use_orthonormal']
        reconstr_params['normalize_cols'] = params['normalize_cols']
        reconstr_params['single_density'] = False
        reconstr_params['use_chebyshev'] = params['use_chebyshev']

        reconstr_params['lower_bound'] = np.min(X_t_train)
        reconstr_params['upper_bound'] = np.max(X_t_train)

        reconstr_params['X_time_series_data'] =  X_t_train
        reconstr_params['length_of_time_series'] = X_t_train.shape[0]
        reconstr_params['delay_dimension'] = delay_dimension
        reconstr_params['time_skip'] = time_skip
        reconstr_params['number_of_vertices'] = delay_dimension*X_t_train.shape[1]

        if params['use_orthonormal']:
            out_dir_ortho_folder = 'orth_data/orth_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(reconstr_params['exp_name'],
                                                                            reconstr_params['random_seed'],
                                                                            reconstr_params['max_deg_monomials'],
                                                                            dt,
                                                                            delay_dimension,
                                                                            time_skip,
                                                                            ttrain,
                                                                            ttest)
            
            output_orthnormfunc_filename = out_dir_ortho_folder

            if not os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                reconstr_params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(reconstr_params, 
                                                                         save_orthnormfunc = True)
            if os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                      
            reconstr_params['build_from_reduced_basis'] = False

        ## Training phase
        RC = ng(reconstr_params, 
                delay = delay_dimension, 
                time_skip = time_skip,
                ind_term = True)
        R = RC.run(X_t_train.T)
        reconstr_params = RC.params
        reg_param = params['reg_params']

        S = R @ R.T
        s = LA.svd(R.T, lapack_driver='gesvd', compute_uv=False)

        #Readout matrix calculation
        W_out = ridge(s_t_train.T - u_t_train.T, R, reg_param = reg_param)
        
        if reconstr_params['normalize_cols']:
            W_out = W_out/reconstr_params['norm_column']

        v_t_train = u_t_train.T + W_out @ R
        
        # Dictionary associated with one specific experiment
        exp_dict = dict()
        exp_dict['W_out'] = W_out.T
        reconstr_params = RC.params
    
        # Import the singular values of the matrix $R$
        exp_dict['sigvals'] = s

        if self.experiment['testing']:
            #============================##============================##============================#
            ## Testing phase
            hist = X_t_train[-(warmup + 1):, :].copy()
    
            v_t_test = RC.gen_autonomous_state(W_out, hist.T, t_test)
            s_t_test, v_t_test, t_test = tls.select_bounded(X_t_test.T, v_t_test, t_test)
            
            # Import the time series generated by the NGRC
            exp_dict['v_t_train'] = v_t_train
            exp_dict['v_out_test'] = v_t_test
            exp_dict['is_bounded'] = RC.is_bounded 
            exp_dict['t_train'] = t_train
            exp_dict['t_test'] = t_test
            
            # Compute different error measures whenever the autonomous NGRC dynamics is bounded.
            
            if exp_dict['is_bounded']:
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
                                                      nperseg=int(1/dt)*5
                                                      )
            else:
                exp_dict['NRMSE_train'] = -1
                exp_dict['NRMSE_test'] = -1
                exp_dict['VPT_train'] = -1
                exp_dict['VPT_test'] = -1 
                exp_dict['Loss_train'] = -1 
                exp_dict['Loss_test'] = -1 
                exp_dict['abs_psd_train'] = -1
                exp_dict['abs_psd_test'] = -1
                
            # Import the many parameters for the given simulation. Easier to discover later on.
            exp_dict['script_params'] = params
                
        return exp_dict
#%%    
    def ngrc_euler_method_lgth_train(self, params):
        '''
        To train and test performance of a NGRC for the given parameters.
        It creates dynamic time series generated using Euler method.
        The input is the length of training data, instead of the ttrain.
        
        Parameters
        ----------
        params : dict
            Parameters.

        Returns
        -------
        exp_dict : dictionary
            W_out : numpy array
                Readout matrix 
            v_t_train : numpy array 
                Training NGRC trajectory for plotting training.
            is_bounded : boolean
                Check if the reconstructed dynamics is bounded.
            s_t_test : numpy array
                True trajectory within testing phase.
            v_out_drive : numpy array
                Reservoir output within testing phase.
            t_train : numpy array
                Time span evaluated within training phase.
            t_test : numpy array
                Time span evaluated within testing phase.
            sigvals : numpy array 
                Singular values of the library matrix evaluated on the training trajectory.
                
            results with different measures during training and testing - below {} corresponds to train,test:
              
            NRMSE_{} : normalized root mean square error
            VPT_{} : valid prediction time
            Loss_{} : loss
            abs_psd_train_{} : Kullback Leibler divergence between the power spectrum density
            zmax_{} : computing distance between the z-max map.
            error_wrt_true : computes the difference between W_out and true coefficient.
            
        '''
        #Time step - sampling
        dt = params['dt']
        dt_fine = params['dt_fine']
        #Delayed coordinates and time skip
        delay_dimension = params['delay_dimension']
        #Time skip between time points
        time_skip = params['time_skip']
        #Warm up of the NGRC
        warmup = (params['delay_dimension'] - 1)*params['time_skip']
        #Training and testing data
        ttrain = params['Ntrain']*params['dt']
        ttest = params['Ntest']*params['dt']
        #Random seed identifier
        seed = params['random_seed']
        #============================##============================##============================#
        #Generate synthetic data
        ts_sgn = sgn(dt, ttrain, ttest, 
                     delay_dimension, 
                     time_skip,                                  
                     trans_t = 100, 
                     normalize = params['normalize_ts'],
                     seed = seed,
                     method = 'Euler',
                     dt_fine = params['dt_fine'])
        
        folder = 'data/input_data/'
        ts_filename = folder+'Lorenz_ts_Euler_{}_{}_{}.txt'.format(ttrain+ttest+warmup*dt, 
                                                                   dt_fine, seed)
        ts_sgn.generate_signal(parametric_Lorenz, 
                               np.array([10.0, 8.0/3.0, 28]),
                               ts_filename,
                               subsampling=True)

        X_t_train, X_t_test = ts_sgn.X_t_train, ts_sgn.X_t_test
        u_t_train, s_t_train = ts_sgn.u_t_train, ts_sgn.s_t_train
        t_train, t_test = ts_sgn.t_train, ts_sgn.t_test
        #============================##============================##============================#
        ############# Construct the parameters dictionary ##############
        reconstr_params = dict()

        reconstr_params['exp_name'] = params['exp_name']
        reconstr_params['network_name'] = params['network_name']
        reconstr_params['max_deg_monomials'] = params['max_deg_monomials']
        
        reconstr_params['use_canonical'] = params['use_canonical']
        reconstr_params['use_orthonormal'] = params['use_orthonormal']
        reconstr_params['normalize_cols'] = params['normalize_cols']
        reconstr_params['single_density'] = False
        reconstr_params['use_chebyshev'] = params['use_chebyshev']

        reconstr_params['lower_bound'] = np.min(X_t_train)
        reconstr_params['upper_bound'] = np.max(X_t_train)

        reconstr_params['X_time_series_data'] =  X_t_train
        reconstr_params['length_of_time_series'] = X_t_train.shape[0]
        reconstr_params['delay_dimension'] = delay_dimension
        reconstr_params['time_skip'] = time_skip
        reconstr_params['number_of_vertices'] = delay_dimension*X_t_train.shape[1]

        if params['use_orthonormal']:
            out_dir_ortho_folder = 'orth_data/orth_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(reconstr_params['exp_name'],
                                                                            reconstr_params['random_seed'],
                                                                            reconstr_params['max_deg_monomials'],
                                                                            dt,
                                                                            delay_dimension,
                                                                            time_skip,
                                                                            ttrain,
                                                                            ttest)
            
            output_orthnormfunc_filename = out_dir_ortho_folder

            if not os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                reconstr_params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(reconstr_params, 
                                                                         save_orthnormfunc = True)
            if os.path.isfile(output_orthnormfunc_filename):
                reconstr_params['orthnorm_func_filename'] = output_orthnormfunc_filename
                      
            reconstr_params['build_from_reduced_basis'] = False

        ## Training phase
        RC = ng(reconstr_params, 
                delay = delay_dimension, 
                time_skip = time_skip,
                ind_term = True)
        R = RC.run(X_t_train.T)
        reconstr_params = RC.params
        reg_param = params['reg_params']

        S = R @ R.T
        s = LA.svd(R.T, lapack_driver='gesvd', compute_uv=False)

        #Readout matrix calculation
        W_out = ridge(s_t_train.T - u_t_train.T, R, reg_param = reg_param)
        
        if reconstr_params['normalize_cols']:
            W_out = W_out/reconstr_params['norm_column']

        v_t_train = u_t_train.T + W_out @ R
        
        # Dictionary associated with one specific experiment
        exp_dict = dict()
        exp_dict['W_out'] = W_out.T
        reconstr_params = RC.params
        c_matrix_true = get_true_coeff_Lorenz(reconstr_params)

        exp_dict['error_wrt_true'] = error_wrt_true(W_out.T/dt, c_matrix_true)
    
        # Import the singular values of the matrix $R$
        exp_dict['sigvals'] = s

        if self.experiment['testing']:
            #============================##============================##============================#
            ## Testing phase
            hist = X_t_train[-(warmup + 1):, :].copy()
    
            v_t_test = RC.gen_autonomous_state(W_out, hist.T, t_test)
            s_t_test, v_t_test, t_test = tls.select_bounded(X_t_test.T, v_t_test, t_test)
            
            # Import the time series generated by the NGRC
            exp_dict['v_t_train'] = v_t_train
            exp_dict['v_out_test'] = v_t_test
            exp_dict['is_bounded'] = RC.is_bounded 
            exp_dict['t_train'] = t_train
            exp_dict['t_test'] = t_test
            
            # Compute different error measures whenever the autonomous NGRC dynamics is bounded.
            
            if exp_dict['is_bounded']:
                #During training 
                exp_dict['NRMSE_train'] = NRMSE(s_t_train.T, exp_dict['v_t_train'])
        
                exp_dict['VPT_train'] = valid_prediction_time(s_t_train.T, exp_dict['v_t_train'], dt)
        
                tau_lyap = 1/0.9056
                t_loss_train = np.linspace(0, t_train[-1]-tau_lyap, 50)
                exp_dict['Loss_train'] = loss(s_t_train.T, exp_dict['v_t_train'], t_vec = t_loss_train, 
                                              tau = tau_lyap, 
                                              dt = dt)
        
                exp_dict['zmax_train'] = l2_zmax_map(s_t_train.T, v_t_train)
        
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
        
                exp_dict['zmax_test'] = l2_zmax_map(s_t_test, v_t_test)
        
                exp_dict['abs_psd_test'] = kl_div_psd(s_t_test, 
                                                      exp_dict['v_out_test'], 
                                                      dt,
                                                      nperseg=int(1/dt)*5
                                                      )
            else:
                exp_dict['NRMSE_train'] = -1
                exp_dict['NRMSE_test'] = -1
                exp_dict['VPT_train'] = -1
                exp_dict['VPT_test'] = -1 
                exp_dict['Loss_train'] = -1 
                exp_dict['Loss_test'] = -1 
                exp_dict['zmax_train'] = -1
                exp_dict['zmax_test'] = -1
                exp_dict['abs_psd_train'] = -1
                exp_dict['abs_psd_test'] = -1
                
            # Import the many parameters for the given simulation. Easier to discover later on.
            exp_dict['script_params'] = params
                
        return exp_dict
#%%    
    def exp_instance(self, params):        
        '''
        Method to run an experiment for a given set of parameters in params.

        Parameters
        ----------
        params : dict
            Parameters to run an experiment.

        Returns
        -------
        exp_dict : dict
            Experiment dictionary.

        '''
        #============================##============================#
        '''
        #Train and test a given NGRC model - Fixed Lorenz time series 
        '''
        
        if self.experiment['script'] == 'set_rc':
            exp_dict = self.set_rc(params)
        
        if self.experiment['script'] == 'ngrc_euler_method':
            exp_dict = self.ngrc_euler_method(params)
        
        if self.experiment['script'] == 'ngrc_euler_method_lgth_train':
            exp_dict = self.ngrc_euler_method_lgth_train(params)
            
        if self.experiment['script'] == 'ngrc_euler_method_index':
            exp_dict = self.ngrc_euler_method_index(params)
            
        return exp_dict
        
#============================##============================##============================##============================#     
    def build_iter_params(self, ):
        '''
        Construct a list of dictionaries containing the different params dictionaries.

        '''
        self.iterable_params = dict()
        
        keys = list(self.vary_params.keys())
        v_ps = []
        for key in keys:
            v_ps.append(self.vary_params[key])
                    
        self.iterable = list(self.combination_tools(*v_ps))
        
        for i, comb in enumerate(self.iterable):
            comb_ = self.comb_string(comb, i)
            
            self.iterable_params[comb_] = []
            params = self.parameters.copy()
            
            for id_key, key in enumerate(keys):
                params[key] = comb[id_key]
            
            for seed in self.seed_vec:
                params_ = params.copy() 
                params_['random_seed'] = seed
                
                self.iterable_params[comb_].append(params_)
                
                
    def check_comb_hdf5(self, fname_output):    
        '''
        Check if the hdf5 file exists.

        Parameters
        ----------
        comb : tuple
            Identifier of the parameter being varied.
        fname_output : str
            Filename of the hdf5 file.

        Returns
        -------
        TYPE
           dict, boolean
           In case the file already exits, return the experiment dictionary.
           Otherwise, the simulation runs over such parameters.

        '''
        #Verify if the experiment has been run before!
        if os.path.isfile(fname_output):
            exp_dict = self.read_hdf5(fname_output)
            return exp_dict, True
        else:
            return None, False
#============================##============================##============================##============================#        
    def run_params(self, comb):
        '''
        Run and save in a hdf5 file an experiment for a given combination of 
        parameters.

        Parameters
        ----------
        comb : tuple
            Identifier of the parameter being varied.

        Returns
        -------
        exp_dict : dict
            Experiment dictionary.

        '''
        Nseeds = len(self.iterable_params[comb])
        #Verify if the experiment has been run before!
        filename = self.comb_fname(comb)
        fname_output = self.file_output(filename)
        exp_dict, _run_exp = self.check_comb_hdf5(fname_output)
        
        if _run_exp:
            return exp_dict
        else:
            out_results_hdf5 = h5dict.File(fname_output, 'a')    
            
            exp_dict = dict()
          
            for i in range(Nseeds):
                params = self.iterable_params[comb][i]    
                exp_dict[i] = self.exp_instance(params)
                out_results_hdf5[i] = exp_dict[i].copy()
            
            out_results_hdf5.close()
            
            return exp_dict
    
    def run(self, ):
        '''
        Run an experiment. For a given set of parameters, it takes a cartesian
        product checking all possible combination of parameters and running 
        the experiment out of Nseeds and saving in a hdf5 file.

        Returns
        -------
        dict
            Experiment dictionary.

        '''
        self.build_iter_params()
        self.exps_dict = dict()
        total_number_comb = len(self.iterable_params)
        for i, comb in enumerate(self.iterable_params):
            
            print('Combination', i, '/', total_number_comb, ':', comb)
            print('\n')
            self.exps_dict[comb] = self.run_params(comb)
    
        return self.exps_dict
    
    def run_instance(self, comb, id_seed):
        params = self.iterable_params[comb][id_seed]    
        return self.exp_instance(params)
        
    def run_parallel(self, nproc):
        
        self.build_iter_params()
                
        for i, comb in enumerate(self.iterable_params):
            
            print('Combination', i, ':', comb)
            print('\n')
            Nseeds = len(self.iterable_params[comb])
            #Verify if the experiment has been run before!
            filename = self.comb_fname(comb)
            fname_output = self.file_output(filename)
            exp_dict, _run_exp = self.check_comb_hdf5(fname_output)
            
            if _run_exp:
                try:
                    self.exps_dict[comb] = exp_dict
                except:
                    raise ValueError("The file does not have the experiment\
                                     to be accessed from.")
            else:
                ps = range(Nseeds)    
                
                pool = Pool(processes=nproc)
                async_results = [pool.apply_async(self.run_instance, 
                                                  args=(comb, id_seed)) 
                                 for id_seed in ps]
                pool.close()
                    
                out_results_hdf5 = h5dict.File(fname_output, 'a')    
                          
                for i, ar in enumerate(async_results):
                    out_results_hdf5[i] = ar.get() 
                    
                out_results_hdf5.close()
            
        return self.exps_dict