"""
Signal composition class - Input - summing over components

Created on Mon Apr 17 13:24:27 2023

@author: Edmilson Roque dos Santos
"""
import numpy as np
import os 

#Upper level class
class signal:
    def __init__(self, dt, ttrain, ttest, 
                 delay_dimension, 
                 time_skip,                                  
                 trans_t = 5, 
                 normalize = True,
                 seed = 1,
                 method = 'RK45',
                 dt_fine = 0.001):
        '''
        Input - summing over components

        Parameters
        ----------
        
        time_total : float
            Total time of integration of the ODEs.
        dt : float
            time step.
        trans_t : float
            Transient time.
        num_k : int
            Number of components.
        ic_k : numpy array
            Initial conditions for each component.
        alpha_vec : numpy array
            Proportion vectors for construction of input signal as sum over 
            u(t) = \sum_k alpha_vec[k] s_k(t)
            
        coord : int
            Select which coordinates of the components to be used in the reconstruction.
        normalize : boolean, optional
            Normalize the input components signals and input data to have mean zero
            and standard deviation 1. The default is True.

        Returns
        -------
        None.

        '''
        self.dt = dt
        self.dt_fine = dt_fine
        self.delay_dimension = delay_dimension
        self.time_skip = time_skip
        self.warmup = (delay_dimension - 1)*time_skip
        
        self.ttrain = ttrain
        self.total_train = self.ttrain + (self.warmup + 1)*self.dt
        self.ttest = ttest
        self.trans_t = trans_t
        self.time_total = self.total_train + self.ttest + self.trans_t
        self.transient_time = int(self.trans_t/self.dt)

        self.normalize = normalize
        
        self.random_seed = seed
        
        self.method = method        
        
        
    def generate_signal(self, dyn_sys, params, ts_filename = None, 
                        subsampling = True):
        '''
        Parameters
        ----------
        dyn_sys : function
            Autonomous dynamical system of each component.
        params_components : dict, optional
            Parameters for each component.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        if not os.path.isfile(ts_filename):
            self.X_ts_fine = dyn_sys(params, 
                                time_total = self.time_total,
                                dt = self.dt_fine,
                                initial_condition = None, 
                                random_seed = self.random_seed,
                                save_data = True,
                                filename = ts_filename,
                                method = self.method)
            
        if os.path.isfile(ts_filename):
            self.X_ts_fine = np.loadtxt(ts_filename)
            
        if subsampling:
            step = int(round(self.dt/self.dt_fine))
            N = int(self.time_total/self.dt)
                        
            X_ts_coarse = self.X_ts_fine[0::step,:].copy()
            self.X_ts = X_ts_coarse[0:N,:].copy()
        
        else:
            self.X_ts = self.X_ts_fine[0:int(self.time_total/self.dt),:].copy()
            
        self.X_ts = self.X_ts[self.transient_time:, :]
        
        # Normalize the input data to have normalized mean and standard deviation
        if self.normalize:
            self.max_X_ts = self.X_ts.max(axis = 0)
            self.X_ts = (self.X_ts)/self.max_X_ts              
            
        lgt_train = int(self.total_train/self.dt)
        
        # Split the time series data into two chucks - training and testing        
        self.X_t_train, self.X_t_test = self.X_ts[:lgt_train, :], self.X_ts[lgt_train:, :]
                
        #Give particular names for shifted training data.
        self.u_t_train, self.s_t_train = self.X_t_train[self.warmup:-1, :], self.X_t_train[self.warmup + 1:, :] 
        
        self.t_train = np.linspace(0, self.s_t_train.shape[0]*self.dt, self.s_t_train.shape[0])
        self.t_test = np.linspace(0, self.X_t_test.shape[0]*self.dt, self.X_t_test.shape[0])

        
        
    
    
    
    
    
    
    
    
    