import numpy as np
import os

from . import poly_library as polb
from .triage import triage_params

def delay_embeding(u, params):
    
    warmup = (params['delay_dimension'] - 1)*params['time_skip']
    
    T_, M = u.shape
    
    if T_ < warmup:
        raise ValueError(f"The length of training should be larger than warmup = {warmup} dt.")
    
    T = T_ - warmup
    dlin = params['delay_dimension']*M
    Rlin = np.zeros((dlin, T))
    
    # fill in the linear part of the feature vector for all times
    
    
    #Delay coordinates are positioned first in decreasing order. 
    #So the order goes as follows:
    #-(k - 1) tau, -(k - 2) tau, ... -2 tau, -1  tau, t 
            
    for delay in range(params['delay_dimension']):
        
        idx_vec = np.arange(delay*params['time_skip'], 
                            T + delay*params['time_skip'], 
                            1, dtype = int)
        
        Rlin[M*delay : M*(delay+1), :] = u[idx_vec, :].T
    
    return Rlin.T

def create_orthnormfunc_kde(params, save_orthnormfunc = True):
    """
    Create the orthonormal functions adapted for the measure estimated from
    the data X_times series using Kernel density estimation. 

    Parameters
    ----------
    params : dict
        DESCRIPTION.
    save_orthnormfunc : boolean, optional
        The default is True. It determines if the dictionary corresponding to the 
        set of orthonormal functions will be saved in a file. 

    Returns
    -------
    dictionary of orthornomal functions
        
    """

    parameters = params.copy()
    
    parameters['use_orthnorm_func_data'] = False
    parameters['save_orthnormfunc'] = save_orthnormfunc
    parameters['build_from_reduced_basis'] = False
        
    #Pick data to estimate the kernel density
    parameters['X_time_series_data'] = delay_embeding(params['X_time_series_data'], params)

    parameters['number_of_vertices'] = params.get('number_of_vertices')

    #Generate a fake data simply to construct the basis functions      
    X_t = np.zeros((parameters['length_of_time_series'], parameters['number_of_vertices']))
    
    #Any program using the parameters must be triaged
    parameters = triage_params(parameters)
        
    if save_orthnormfunc:
        parameters['orthnorm_func_filename'] = params['orthnorm_func_filename']
        
    if(parameters['use_orthonormal']):
        #It creates the library matrix associates to the orthonormal function.
        #The expression of the orthonormal functions are saved inside parameters['orthnormfunc']
        #The symbolic expressions are saved in  parameters['symbolic_PHI']
        PHI, parameters = polb.library_matrix(X_t, parameters)
    
    return parameters['orthnormfunc']

def create_orthnormfunc_filename(params):
    '''
    To create the orthonormal filename using parameters for a simulation.

    Parameters
    ----------
    params : dict
    
    Returns
    -------
    scenario_output : str
        path for orthonormal functions file.

    '''
    parameters = params.copy()
    
    parameters = triage_params(parameters)
    
    ##### Identification for output
    out_direc = os.path.join(parameters['exp_name'], parameters['network_name'])
    if os.path.isdir(out_direc) == False:
        os.makedirs(out_direc)
    
    #For coupling analysis it is necessary to save each orthonormal function 
    #with respect to this coupling.
    filename = 'onf_deg_{}_seed_{}_lgth_ts_{}_coupling_{}_crossed_{}'.format(parameters['max_deg_monomials'], 
                                                              parameters['random_seed'], 
                                                              parameters['length_of_time_series'], 
                                                              parameters['coupling'],
                                                              parameters['expansion_crossed_terms'])
    scenario_output = os.path.join(out_direc, filename)
    
    return scenario_output

    
def set_orthnormfunc(output_orthnormfunc_filename, params):
    '''
    Construct an orthonormal basis functions from file named:
    output_orthnormfunc_filename
    The construction assumes a reduced set of orthornormal functions instead of 
    the full set. To use the full set use from the file use simply:
        tools.SympyDict.load(output_orthnormfunc_filename)

    Parameters
    ----------
    output_orthnormfunc_filename : str
        filename containing the orthonormal functions to be used.
    params : dict
        
    Returns
    -------
    parameters : dict
        Updated parameters dictionary to be used throughout the simulation, where
        'orthnormfunc' is updated.

    '''
    parameters = params.copy()
    
    parameters['use_orthnorm_func_data'] = True
    parameters['save_orthnormfunc'] = False
    parameters['build_from_reduced_basis'] = False
    
    parameters['orthnorm_func_filename'] = output_orthnormfunc_filename
    parameters['orthnorm_func_map'] = True
    parameters['N_inputdata'] = params['X_time_series_data']
    parameters['max_deg_inputdata'] = params['max_deg_monomials']

    parameters = triage_params(parameters)
    
    return parameters
