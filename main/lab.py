"""
Lab class for analysis the experiments over different parameters values.

Created on Thu Jan 30 16:51:33 2025

@author: Edmilson Roque dos Santos
"""

import itertools
import numpy as np
from numpy.random import default_rng
import os

import h5dict

from .simulation import simulation as sim
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

#Upper class function that inherents attributes and methods from simulation
class lab(sim):
    '''
    Class for analysing results from different experiments.
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
        Returns
        -------
        None.

        '''
        sim.__init__(self, experiment, vary_params, fixed_params)
        super().__init__(experiment, vary_params, fixed_params)
        
        self.exp_dict = self.run()
        
        self.keys = list(self.exp_dict.keys())
        self.array_keys = np.array(self.keys)
        
    def get_stats(self, key_metric, print_res = False):
        '''
        Obtain the statistics: 
            mean, variance, maximum and minimum out of a metric.

        Returns
        -------
        Print average +/- standard deviation.

        '''
        exp_dict = self.exp_dict.copy()
        combs = list(exp_dict.keys())
        
        stats_dict = dict()
        
        for comb in combs:
            seeds = list(exp_dict[comb].keys())
            stats_dict[comb] = dict()
            metric_vec = np.zeros(len(seeds))
    
            for seed in seeds:
            
                if key_metric == 'sigvals':
                    sigvals = exp_dict[comb][seed][key_metric]
                        
                    metric_vec[seed] = sigvals.max()/sigvals.min()
                else:
                    metric_vec[seed] = exp_dict[comb][seed][key_metric]
                    
            stats_dict[comb]['min'] = metric_vec.min()
            stats_dict[comb]['max'] = metric_vec.max()
            stats_dict[comb]['mean'] = metric_vec.mean()
            stats_dict[comb]['quantiles'] = np.array([np.quantile(metric_vec, 0.25), np.quantile(metric_vec, 0.5), np.quantile(metric_vec, 0.75)])
            stats_dict[comb]['std'] = metric_vec.std()

            if print_res:
                print(comb, 'mean', stats_dict[comb]['mean'], '+/-', 
                      stats_dict[comb]['std'], 
                      '[{}, {}]'.format(stats_dict[comb]['min'], stats_dict[comb]['max']),
                      'quantiles:', stats_dict[comb]['quantiles']
                      )
        
        self.stats_dict = stats_dict   
        
        
    def get_isbounded(self, ):    
        
        exp_dict = self.exp_dict.copy()
        combs = list(exp_dict.keys())
        
        isbounded_dict = dict()
        
        for comb in combs:
            seeds = list(exp_dict[comb].keys())
            isbounded_dict[comb] = dict()
            isbounded_vec = np.zeros(len(seeds))
    
            for seed in seeds:
                isbounded_vec[seed] = exp_dict[comb][seed]['is_bounded']
                
            isbounded_dict[comb]['all'] = isbounded_vec.all()
            
            if not isbounded_dict[comb]['all']:
                isbounded_dict[comb]['seed_vec'] = isbounded_vec
            
        self.isbounded_dict = isbounded_dict   
        
        return isbounded_dict
    
    def plot_violin(self, x_keys, panel_keys, key_metric):   
        '''
        Plot a violin plot over all realizations.

        Parameters
        ----------
        x_keys : string
            Name of the variable to be kept in the horizontal axis.
        panel_keys : list
            Names of variables to be fix for each plot.
        '''
                
        x_axis = self.vary_params[x_keys]
        N_seeds = self.parameters['Nseeds']
        
        v_ps = [self.vary_params.get(key) for key in panel_keys]

        iterable = list(itertools.product(*v_ps))

        for sub_comb in iterable:
            akeys_temp = self.array_keys.copy()
            title = ''
            for i, exp in enumerate(sub_comb):
                mask = np.isin(akeys_temp, '{}'.format(exp))
                mask = np.any(mask, axis = 1)
                akeys_temp = akeys_temp[mask].copy()
                title = title + '{} = {}'.format(panel_keys[i], exp)
                title = title + ' '
               
            optf_array = np.zeros((x_axis.shape[0], N_seeds))
            
            for id_ in range(akeys_temp.shape[0]):
                comb_id = tuple(akeys_temp[id_])
                
                for id_seed in range(N_seeds):
                    optf_array[id_, id_seed] = self.exp_dict[comb_id][id_seed][key_metric]

            tls.plot_metric_fun(x_axis, optf_array, key_metric, x_keys, title)
        
    def results_sigvals(self, subdict, 
                        x_keys = 'max_deg_monomials',
                        filename = None, 
                        plot_results = False):
        
        # Select the variable to plot in the horizontal axis
        x_axis = subdict[x_keys]
        # Pick the number of seeds to be simulated
        N_seeds = self.parameters['Nseeds']
        
        # Creates the dictionary to be used during the simulation
        res_dict = dict()

        # Pick the keys associated to the subdictionary created        
        keys = list(subdict.keys())
        # Construct the itertools over the combinations of the subdictionary
        v_ps = [subdict.get(key) for key in subdict.keys()]
        
        iterable = list(itertools.product(*v_ps))

        loc_x_keys = keys.index(x_keys)
        
        for id_, x_value in enumerate(x_axis):
            
            # Create a dictionary for each value of the horizontal axis dictionary
            res_dict[x_axis[id_]] = dict()    
            
            sub_combs = []
            for combination in iterable:
                if x_value == combination[loc_x_keys]:
                    sub_combs.append(combination)
            
            for jd_, sub_comb in enumerate(sub_combs):
                                
                comb = self.comb_string(sub_comb, jd_)
                for id_seed in range(N_seeds):
                    res_dict[x_axis[id_]][id_seed] = self.exp_dict[comb][id_seed]['sigvals']
        
        if filename is not None:
            folder = 'Figures/'
            out_direc = os.path.join('', folder)
            
            if os.path.isdir(out_direc) == False:
                os.makedirs(out_direc)
            filename = os.path.join(out_direc, filename)
            
        if plot_results:    
            tls.plot_cond_number(res_dict,
                                 filename,
                                 ax = None,
                                 labels = None)    
        
        return res_dict
        
    def results_metric_reg(self, key_metric, 
                        x_keys = 'reg_params', 
                        curve_keys = ['max_deg_monomials']):
        
        x_axis = self.vary_params[x_keys]
        N_seeds = self.parameters['Nseeds']
        
        if curve_keys is not None:
            v_ps = [self.vary_params.get(key) for key in curve_keys]
    
            iterable = list(itertools.product(*v_ps))
            res_dict = dict()
            for sub_comb in iterable:
                akeys_temp = self.array_keys.copy()
                
                for i, exp in enumerate(sub_comb):
                    mask = np.isin(akeys_temp, '{}'.format(exp))
                    mask = np.any(mask, axis = 1)
                    akeys_temp = akeys_temp[mask].copy()
                 
                optf_array = np.zeros((x_axis.shape[0], N_seeds))
                
                for id_ in range(akeys_temp.shape[0]):
                    comb_id = tuple(akeys_temp[id_])
                      
                    for id_seed in range(N_seeds):
                        if key_metric == 'sigvals':
                            sigvals = self.exp_dict[comb_id][id_seed][key_metric]
                            optf_array[id_, id_seed] = sigvals.max()/sigvals.min()
                        else:
                            optf_array[id_, id_seed] = self.exp_dict[comb_id][id_seed][key_metric]
                
                res_dict[exp] = optf_array
        
        if curve_keys is None:
            iterable = list(self.exp_dict.keys())
    
            optf_array = np.zeros((x_axis.shape[0], N_seeds))
            for id_, comb in enumerate(iterable):
                
                for id_seed in range(N_seeds):
                    if key_metric == 'sigvals':
                        sigvals = self.exp_dict[comb][id_seed][key_metric]
                        optf_array[id_, id_seed] = sigvals.max()/sigvals.min()
                    else:
                        optf_array[id_, id_seed] = self.exp_dict[comb][id_seed][key_metric]
            
        return optf_array
        
    
    def plot_fig_metrics_reg(self, list_metrics, 
                            x_keys = 'reg_params', 
                            curve_keys = ['max_deg_monomials'],
                            filename = None,
                            plot = False):
        
        res_dict = dict()
        for id_list, key_metric in enumerate(list_metrics):    
            
            res_dict[key_metric] = self.results_metric_reg(key_metric,
                                                           x_keys,
                                                           curve_keys)
                
        x_axis = self.vary_params[x_keys]
        
        if plot:
            if filename is not None:
                folder = 'Figures/'
                out_direc = os.path.join('', folder)
                
                if os.path.isdir(out_direc) == False:
                    os.makedirs(out_direc)
                filename = os.path.join(out_direc, filename)
            
            tls.plot_fig_metrics_reg(res_dict, x_axis, filename = filename,
                                     reference_value = True)
        
        return res_dict
        
    def metrics_time_skip(self, list_metrics, 
                                x_keys = 'time_skip'): 
        
        x_axis = self.vary_params[x_keys]
        res_dict = dict()
        
        if len(list_metrics) > 1:
            for id_list, key_metric in enumerate(list_metrics):    
                
                N_seeds = self.parameters['Nseeds']
                list_keys = list(self.exp_dict.keys())
                
                optf_array = np.zeros((x_axis.shape[0], N_seeds))
                for id_, comb in enumerate(list_keys):
                
                    for id_seed in range(N_seeds):
                        if key_metric == 'sigvals':
                            sigvals = self.exp_dict[comb][id_seed][key_metric]
                            optf_array[id_, id_seed] = sigvals.max()/sigvals.min()
                        else:
                            optf_array[id_, id_seed] = self.exp_dict[comb][id_seed][key_metric]
                
                res_dict[key_metric] = dict()
                res_dict[key_metric][id_list] = optf_array
        
        else:
            for id_list, key_metric in enumerate(list_metrics):    
                
                res_dict[key_metric] = self.results_metric_reg(key_metric,
                                                               x_keys,
                                                               None)
        
        return res_dict
        
        
    def metrics_lgth_train(self, list_metrics, 
                           x_keys = 'Ntrain', 
                           curve_keys = ['dt']):
        
        res_dict = dict()
        for id_list, key_metric in enumerate(list_metrics):    
            
            res_dict[key_metric] = self.results_metric_reg(key_metric,
                                                           x_keys,
                                                           curve_keys)
                
        return res_dict
        
    def metrics_poly_deg(self, list_metrics, 
                         x_keys = 'max_deg_monomials', 
                         curve_keys = ['max_deg_monomials']):
        
        res_dict = dict()
        for id_list, key_metric in enumerate(list_metrics):    
            
            res_dict[key_metric] = self.results_metric_reg(key_metric,
                                                           x_keys,
                                                           curve_keys)
                
        return res_dict     
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    