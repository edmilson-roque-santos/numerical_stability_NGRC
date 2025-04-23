"""
Next Generation Reservoir computing - echo state network

Created on Mon Aug 12 10:01:25 2024

@author: Edmilson Roque dos Santos
"""

import numpy as np
import scipy.special
import sympy as spy

from .base_polynomial import pre_settings as pre_set 
from .base_polynomial import poly_library as polb
from .base_polynomial import triage as trg

class ng_reservoir:
    
    def __init__(self, params, 
                 delay = 2, 
                 time_skip = 1,
                 ind_term = True):
        
        self.delay = delay
        self.time_skip = time_skip
        self.warmup = (self.delay - 1)*self.time_skip
        
        self.ind_term = ind_term
        self.params = params
        
    def delay_embeding(self, u):
        
        M, T_ = u.shape
        
        if T_ < self.warmup:
            raise ValueError(f"The length of training should be larger than warmup = {self.warmup} dt.")
        
        T = T_ - self.warmup
        if T > 1:
            Rlin = np.zeros((self.dlin, T))
        else:
            Rlin = np.zeros(self.dlin)
        # fill in the linear part of the feature vector for all times
        
        
        #Delay coordinates are positioned first in decreasing order. 
        #So the order goes as follows:
        #-(k - 1) tau, -(k - 2) tau, ... -2 tau, -1  tau, t 
                
        for delay in range(self.delay):
            
            idx_vec = np.arange(delay*self.time_skip, T + delay*self.time_skip, 1, dtype = int)
            
            if T > 1:
                Rlin[M*delay : M*(delay+1), :] = u[:, idx_vec]
            else:
                # During the testing phase, the current and delayed states have
                # only one time step to evalute the different coordinates of the model.
                Rlin[M*delay : M*(delay+1)] = u[:, idx_vec[0]]
                
        return Rlin
    
    def run(self, u):
        '''
        Generate the state of the ng reservoir for a given input data.

        Parameters
        ----------
        u : numpy array
            Input data - shape = (time_steps, dimensions/number_of_features).
        perc_transt: float 
            Percentage of input data that is discarded because it is considered
            transient dynamics. The default is 50% of input data. 
        Returns
        -------
        R : numpy array
           Gathering the activations of a node over a timeseries.
           shape - (states, time_steps - trans_t).

        ''' 
        M, T = u.shape
               
        #dimension linear
        self.dlin = self.delay*M

        Rlin = self.delay_embeding(u).T        
        
        params_ = self.params.copy()
        params_['number_of_vertices'] = self.dlin
        if params_['use_canonical']:
            self.params = trg.triage_params(params_)
            
        if params_['use_chebyshev']:
            self.params = trg.triage_params(params_)    
         
        if params_['use_orthonormal'] and params_['build_from_reduced_basis']:
            self.params = pre_set.set_orthnormfunc(params_['orthnorm_func_filename'], params_)
        
        if params_['use_orthonormal']:
            params_['X_time_series_data'] = Rlin
            self.params = trg.triage_params(params_)
            
        self.params['length_of_time_series'] = Rlin.shape[0]
        
        PHI, self.params = polb.library_matrix(Rlin, self.params)
        
        # total size of feature vector: linear + nonlinear
        if self.ind_term:
            self.dtot = self.params['L']
            self.R = PHI.T.copy()
        
        else:
            self.dtot = self.params['L'] - 1
            self.R = PHI[:, 1:].T.copy()
        
        return self.R[:, :-1]
    

    def check_boundedness(self, v_t, params, max_multiplier = 100):
        
        mask_lower = (v_t < max_multiplier*params['lower_bound'])
        mask_upper = (v_t > max_multiplier*params['upper_bound']) 
        mask_isnan = (np.any(np.isnan(v_t)))
        mask_bounds = mask_lower | mask_upper | mask_isnan
        
        if np.any(mask_bounds):
            print('Warning: Trajectory reached infinity!')
            return False
        else:
            return True
    
    def gen_autonomous_state(self, W_out, hist, t): 
        '''
        Generate autonomously the reservoir states for the history function u_0

        Parameters
        ----------
        W_out : TYPE
            DESCRIPTION.
        hist : numpy array
            History function for the autonomous phase of the NGRC model.
        t : numpy array
            Time vector for testing phase.

        Returns
        -------
        v_out_t : numpy array 
            DESCRIPTION.

        '''
        T = t.shape[0]
        u_0 = hist.copy()
        M = u_0.shape[0]
        
        Rlin = self.delay_embeding(u_0)
        
        v_out_t = np.zeros((M, T))
        
        if u_0.shape[1] > 1:
            v_out_t[:, 0] = u_0[:, -1].copy()
        else:
            v_out_t[:, 0] = u_0[:, 0].copy()
        
        params_ = self.params.copy()
        N = params_['number_of_vertices']
        x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
        
        sym_PHI = spy.Matrix(params_['symbolic_PHI'], evaluate=False)
        sym_PHI = sym_PHI.T
        sym_expr = spy.lambdify([x_t], sym_PHI, 'numpy')
        
        for id_t in range(1, T):
            
            if self.ind_term:
                # create an array to hold the feature vector
                R = sym_expr(Rlin)[0]
            
            else:
                # create an array to hold the feature vector
                R = sym_expr(Rlin)[0][1:]
            
            #We update the current state
            v_out_t[:, id_t] = v_out_t[:, id_t - 1] + W_out @ R
            
            #We check the boundedness of the iteration.
            self.is_bounded = self.check_boundedness(v_out_t, params_)
            
            if not self.is_bounded:
                v_out_t = v_out_t[:, :id_t]
                break 
            
            #The history function must be updated accordingly
            #The delayed states are iterated one step forward in time
            #Note that we respect the order of the position of the delays
            if u_0.shape[1] > 1:
                u_0[:, 0:self.warmup] = u_0[:, 1:]
                u_0[:, -1] = v_out_t[:, id_t].copy()
                
            else:
                u_0 = np.array([v_out_t[:, id_t]]).T
                
            Rlin = self.delay_embeding(u_0)
        
        return v_out_t
    
    # The comment here that this method only works for a NGRC model that
    # it does not assume a numerical integrator scheme.
    
    def gen_aut_driven_state(self, W_out, hist, t, input_t): 
        '''
        Generate autonomously the reservoir states but with presence of external 
        input that are always present. 

        Parameters
        ----------
        W_out : TYPE
            DESCRIPTION.
        hist : numpy array
            History function for the autonomous phase of the NGRC model.
        t : numpy array
            Time vector for testing phase.

        Returns
        -------
        v_out_t : numpy array 
            DESCRIPTION.

        '''
        T = t.shape[0]
        u_0 = hist.copy()
        M = u_0.shape[0]
        
        v_out_t = np.zeros((M, T))
        
        # First, record the current time for the prediction
        if u_0.shape[1] > 1:
            v_out_t[:, 0] = u_0[:, -1].copy()
        else:
            v_out_t[:, 0] = u_0[:, 0].copy()
        
        
        # The input_t is also given to populate the reservoir state.
        Rlin = self.delay_embeding(np.vstack((u_0, input_t[:, 0:self.warmup + 1])))
        
        params_ = self.params.copy()
        N = params_['number_of_vertices']
        x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
        
        sym_PHI = spy.Matrix(params_['symbolic_PHI'], evaluate=False)
        sym_PHI = sym_PHI.T
        sym_expr = spy.lambdify([x_t], sym_PHI, 'numpy')
        
        for id_t in range(1, T):
            
            if self.ind_term:
                # create an array to hold the feature vector
                R = sym_expr(Rlin)[0]
            
            else:
                # create an array to hold the feature vector
                R = sym_expr(Rlin)[0][1:]
            
            #We update the current state
            v_out_t[:, id_t] = W_out @ R
            
            #We check the boundedness of the iteration.
            self.is_bounded = self.check_boundedness(v_out_t, params_)
            
            if not self.is_bounded:
                v_out_t = v_out_t[:, :id_t]
                break 
            
            #The history function must be updated accordingly
            #The delayed states are iterated one step forward in time
            #Note that we respect the order of the position of the delays
            if u_0.shape[1] > 1:
                u_0[:, 0:self.warmup] = u_0[:, 1:]
                u_0[:, -1] = v_out_t[:, id_t].copy()
                
            else:
                u_0 = np.array([v_out_t[:, id_t]]).T
            
            # The input_t is also given to populate the reservoir state.
            Rlin = self.delay_embeding(np.vstack((u_0, input_t[:, id_t:id_t + self.warmup + 1])))
        
        
        return v_out_t
    
    
    # TO DO this method does not work in the current form. This must be updated!
    '''
    def gen_aut_state_supp(self, W_out, u_0, t, supp_dict): 
        
        Generate autonomously the reservoir states for a initial state u_0
        supported in a set of indices. This means that not all basis functions
        are considered from the library. 
        
        Parameters
        ----------
        W_out : TYPE
            DESCRIPTION.
        u_0 : TYPE
            DESCRIPTION.
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        u_out : TYPE
            DESCRIPTION.

        
        T = t.shape[0]
        M = u_0.shape[0]
        
        Rlin = np.zeros((self.dlin, T))
        Rlin[:, 0] = self.state_train[:, -1]
           
        v_out_t = np.zeros((M, T))
        v_out_t[:, 0] = u_0
                
        params_ = self.params.copy()
        N = params_['number_of_vertices']
        x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
        
        for id_node in range(W_out.shape[0]):
            new_symb = [params_['symbolic_PHI'][i] for i in supp_dict[id_node]['supp']]
            sym_PHI = spy.Matrix(new_symb, evaluate=False)
            
            sym_PHI = sym_PHI.T
            sym_expr = spy.lambdify([x_t], sym_PHI, 'numpy')
        
            supp_dict[id_node]['sym_expr'] = sym_expr
            
        for id_t in range(1, T):
            
            for id_node in range(M):
                R = supp_dict[id_node]['sym_expr'](Rlin[:, id_t - 1])[0]
                v_out_t[id_node, id_t] = v_out_t[id_node, id_t - 1] + W_out[id_node, :] @ R
            
            #The delay states must be updated accordingly
            #So, the delay are iterated one step forward in time
            #Note that we respect the order of the position of the delays
            
            Rlin[0 : self.dlin - M, id_t] = Rlin[M : self.dlin, id_t - 1]
            
            #Finally we update the current states
            Rlin[self.dlin - M : self.dlin, id_t] = v_out_t[:, id_t].copy()           
            
        self.state_test = Rlin.copy()
        
        return v_out_t
    '''
    
    
    
    
    
    
    
    
    
    
    
    