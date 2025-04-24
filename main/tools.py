"""
Plotting utils

Created on Tue Mar 14 13:24:54 2023

@author: Edmilson Roque dos Santos
"""

import itertools
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import os
import seaborn as sns
from scipy.signal import find_peaks
import scipy.interpolate as interpolate
from scipy.signal import welch
import scipy.special
from scipy import stats
from scipy import signal
from scipy.optimize import curve_fit


# Set plotting parameters
params_plot = {'axes.labelsize': 14,
              'axes.titlesize': 14,
              'axes.linewidth': 1.0,
              'axes.xmargin':0.05, 
              'axes.ymargin': 0.05,
              'legend.fontsize': 10,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'figure.figsize': (7, 3),
              'figure.titlesize': 15,
              'font.serif': 'Computer Modern Serif',
              'mathtext.fontset': 'cm',
              'lines.linewidth': 1.0
             }

plt.rcParams.update(params_plot)
plt.rc('text', usetex=True)

list_colors = sns.color_palette("deep")
colors = [list_colors[0], list_colors[3], 'gray']
list_colors_gray = sns.color_palette("gray")
list_colors_blue = sns.color_palette("mako_r")
list_colors_red = sns.color_palette("rocket_r")
list_markers = list(Line2D.filled_markers)
#============================##============================##============================##============================#
# Auxiliary methods    
#============================##============================##============================##============================#
def select_bounded(s_true, v_t, t):
    
    if s_true.shape[0] == 1:
        lgth = np.min([s_true[0].shape[0], v_t[0].shape[0], t.shape[0]])
        
        return s_true[:lgth], v_t[:lgth], t[:lgth]
        
    else:
        lgth = np.min([s_true.shape[1], v_t.shape[1], t.shape[0]])
        
        return s_true[:, :lgth], v_t[:, :lgth], t[:lgth]
    

def maximum_over_z(s_t):
    '''
    Compute the successive maxima over the z-coordinate of the Lorenz system.

    Parameters
    ----------
    s_t : TYPE
        DESCRIPTION.

    Returns
    -------
    z_max_t : TYPE
        DESCRIPTION.

    '''
    id_coord = 2
    z_t = s_t[id_coord, :]
    
    peaks, _ = find_peaks(z_t, height=0)
        
    z_max_t = z_t[peaks]
    
    return z_max_t

#============================##============================##============================##============================#
# Plotting methods
#============================##============================##============================##============================#

def plot_training(y_true, y_pred, t, scale = 1, fig = None):
    '''
    Plot training phase for during time t

    Parameters
    ----------
    y_true : numpy array
        True trajectories.
    y_pred : numpy array
        Reconstructed trajectories.
    t : numpy array
        time array.
    ax: matplotlib axes object
        It can import an axes object externally to plot in a large figure. 
    Returns
    -------
    

    '''
    
    lgth = np.min([y_true.shape[1], y_pred.shape[1], t.shape[0]])
    
    if fig is None:
        fig = plt.figure(dpi = 300)
        
    nrows = y_true.shape[0]
    
    if nrows == 1:
        
        ax = fig.subplots(nrows=nrows)
        
        ax.plot(t/scale, y_true[0, :lgth], label = r'True', color = colors[0])
        ax.plot(t/scale, y_pred[0, :lgth], '--',  
                label = r'Reservoir', color = colors[1], alpha = 1.0)
        
        ax.set_xlabel(r'Lyapunov Time')
        ax.set_ylabel(r'$x$')
        
        ax.set_title(r'Training phase')
        
        ax.legend(loc=0)
        
        
    else:
        ax = fig.subplots(nrows=nrows, sharex = True)
       
        ax[0].set_title(r'Training phase')        
        labels = [r'$x$', r'$y$', r'$z$']
        
        for id_row in range(nrows):
            ax[id_row].plot(t/scale, y_true[id_row, :lgth], label = r'True', color = colors[0])
            ax[id_row].plot(t/scale, y_pred[id_row, :lgth], '--',  
                    label = r'Reservoir', color = colors[1], alpha = 1.0)
            
            try:
                ax[id_row].set_ylabel(labels[id_row])
            except:
                ax[id_row].set_ylabel(r'$x_{}$'.format(id_row))
            
            if id_row == 0:
                ax[id_row].legend(loc=5)
                
        ax[id_row].set_xlabel(r'Lyapunov Time')    
        
def plot_testing(y_true, y_pred, t, 
                 transient_plot = -1, 
                 scale = 1, 
                 fig = None):
    '''
    Plot testing phase for during time t
    
    Parameters
    ----------
    y_true : numpy array
        True trajectories.
    y_pred : numpy array
        Reconstructed trajectories.
    t : numpy array
        time array.
    transient_plot : float, optional
        transient time to be discarded when plotting.
    ax: matplotlib axes object
        It can import an axes object externally to plot in a large figure. 
    Returns
    -------
    
    
    '''
    
    if fig is None:
        fig = plt.figure(dpi = 300)
    
    nrows = y_true.shape[0]
    
    lgth = np.min([y_true.shape[1], y_pred.shape[1], t.shape[0]])
    
    if nrows == 1:
        y_true = y_true[:lgth]
        y_pred = y_pred[:lgth]
        t = t[-lgth:]
        
        ax = fig.subplots(nrows=nrows)
        
        ax.plot(t[:transient_plot]/scale, y_true[0, :transient_plot], label = r'True', color = colors[0])
        ax.plot(t[:transient_plot]/scale, y_pred[0, :transient_plot], '--',  
                label = r'Reservoir', color = colors[1], alpha = 1.0)
        
        ax.set_xlabel(r'Lyapunov Time')
        ax.set_ylabel(r'$x$')
        
        ax.set_title(r'Testing phase')
        
        ax.legend(loc=0)
        
    else:
        y_true = y_true[:, :lgth]
        y_pred = y_pred[:, :lgth]
        t = t[-lgth:]
        
        
        labels = [r'$x$', r'$y$', r'$z$']
        
        ax = fig.subplots(nrows=nrows, sharex = True)
            
        ax[0].set_title(r'Testing phase')
        
        for id_row in range(nrows):
            ax[id_row].plot(t[:transient_plot:]/scale, y_true[id_row, :transient_plot], 
                            label = r'True', color = colors[0])
            ax[id_row].plot(t[:transient_plot:]/scale, y_pred[id_row, :transient_plot], 
                            '--', 
                            label = r'Reservoir', color = colors[1], alpha = 1.0)
            
            try:
                ax[id_row].set_ylabel(labels[id_row])
            except:
                ax[id_row].set_ylabel(r'$x_{}$'.format(id_row))
            
            max_y, min_y = np.max(y_true[id_row, :transient_plot]), np.min(y_true[id_row, :transient_plot])
            ax[id_row].set_ylim(min_y, max_y)
            if id_row == 0:
                ax[id_row].legend(loc=5)      
                
        ax[id_row].set_xlabel(r'Lyapunov Time') 
        
def plot_error(error, t, transient_plot = 100):
    '''
    Plot Error array

    Parameters
    ----------
    error : numpy array
        
    t : numpy array
        
    transient_plot : float, optional
        transient time to be discarded when plotting. The default is 100.

    Returns
    -------

    '''
    nrows = error.shape[0]
    
    if nrows == 1:
        fig, ax = plt.subplots(dpi = 300)
        ax.plot(t[transient_plot:-1], error[0, transient_plot:])
        
        ax.set_xlabel(r'Time')
        ax.set_ylabel(r'error$(t)$')
        
        plt.show()
    else:
        labels = [r'error $x$', r'error $y$', r'error $z$']
        fig, ax = plt.subplots(nrows=nrows, sharex = True, dpi = 300)
        ax[0].set_title(r'Error - Testing phase')
        
        for id_row in range(nrows):
            ax[id_row].plot(t[transient_plot:-1], error[id_row, transient_plot:])
            
            ax[id_row].set_ylabel(labels[id_row])
                
        ax[id_row].set_xlabel(r'Time') 
          
        plt.show()

def plot_2d(x_t_true, z_t_true, 
            x_t, z_t, pair, 
            transient_plot,
            ax = None,
            filename = None, 
            titles = [r'True attractor', r'Reconstructed attractor']):

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=300, sharey=True)
    else:
        ax1, ax2 = ax[0], ax[1]
        
    ax1.plot(x_t_true[transient_plot:], z_t_true[transient_plot:],
             color=colors[0])
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r"$z$")
    ax1.set_title(titles[0])

    ax2.plot(x_t[transient_plot:], z_t[transient_plot:],
             color=colors[1])
    ax2.set_xlabel(r'$x$')
    
    ax2.set_title(titles[1])
    
    if filename is not None:
        folder = 'Figures'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        output = os.path.join(out_direc, filename)
        plt.savefig(output+".pdf", format = 'pdf', bbox_inches='tight')
        
def plot_2d_all_combinations(X_t, u_out_t, transient_plot = 100):
    '''
    Plot the phase space - attractor - of the true and reconstructed trajectories

    Parameters
    ----------
    X_t : numpy array
        
    u_out_t : numpy array 
        
    transient_plot : float, optional
        transient time to be discarded when plotting. The default is 100.

    Returns
    -------


    '''

    id_vec = np.arange(u_out_t.shape[0], dtype = int)
    pairs = itertools.combinations(id_vec, 2)

    for pair in pairs:
        x_t_true, z_t_true = X_t[pair[0], :], X_t[pair[1], :]
        x_t, z_t = u_out_t[pair[0], :], u_out_t[pair[1], :]
        
        plot_2d(x_t_true, z_t_true, x_t, z_t, pair, transient_plot)

def fig_compare(x_t_train, u_t_train, t_train,
                x_t_test, u_t_test, t_test,
                scale = 1,
                transient_plot = 100, filename = None):
    
    
    fig = plt.figure(figsize=(8, 7), dpi = 300)
    
    (fig1, fig2) = fig.subfigures(2, 1, height_ratios=[1, 1.35])
    
    
    # Extract the plane x - z of the input data.
    pair = [0, 2]
    x_t_true, z_t_true = x_t_test[pair[0], :], x_t_test[pair[1], :]
    x_t, z_t = u_t_test[pair[0], :], u_t_test[pair[1], :]
        
    (ax1, ax2) = fig1.subplots(1, 2, sharey=True, 
                               gridspec_kw={
                                "bottom": 0.18,
                                "top": 0.80,
                                "left": 0.18,
                                "right": 0.80,
                                "wspace": 0.1
                                })
    
    plot_2d(x_t_true, z_t_true, x_t, z_t, pair, transient_plot, ax = [ax1, ax2])
    
    # Plot training and testing 
    
    (fig_train, fig_test) = fig2.subfigures(1, 2)
    
    plot_training(x_t_train, u_t_train, t_train, scale = scale, fig = fig_train)
    
    fig_test = plot_testing(x_t_test, u_t_test, t_test, transient_plot, scale = scale, fig = fig_test)
    
    if filename == None:
        plt.show()
    else:
        folder = 'Figures'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        output = os.path.join(out_direc, filename)
        plt.savefig(output+".pdf", format = 'pdf', bbox_inches='tight')
    
    plt.show()
    return 
    

def plot_traj(X_time_series, plot_length = None, plot_legend = False):
    
    # Plot the time series of the input data
    
    number_of_iterations = X_time_series.shape[0]
    t = np.arange(0, number_of_iterations, 1, dtype = int)
    N = X_time_series.shape[1]
    nodelist = np.arange(0, N, 1, dtype = int)
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    
    fig, ax = plt.subplots(1, 1, figsize = (4, 4), dpi = 300)
    
    col = sns.color_palette("mako_r", n_colors = N)
    
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        
        ax.plot(t, 
                X_time_series[:, index], 
                '-', 
                label = r'{}'.format(index),
                color = col[index])

    ax.set_ylabel(r'$x_i(t)$')
    ax.set_xlabel(r'$t$')

    if plot_length is not None:
        ax.set_xlim(plot_length[0], plot_length[1])

    if plot_legend:
        ax.legend(loc = 0, ncols = 5)
    
    plt.show()
    
def plot_return_map(X_time_series, plot_list):
    '''
    Plot return map for each node from multivariate time series.

    Parameters
    ----------
    
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
       Multivariate time series.
    
    Returns
    -------
    None.
    '''
    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    nodelist = np.arange(N, dtype = int)
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    col = sns.color_palette("mako_r", n_colors = N)
    
    fig, ax = plt.subplots(1, 1, figsize = (4, 4), dpi = 300)
    
    for index in plot_list:
        if len(index) > 1:
            ax.plot(X_time_series[:number_of_iterations-1, index[1]], 
                    X_time_series[1:number_of_iterations, index[0]], 
                    'o', 
                    label = r'{}'.format(index[0]),
                    color = col[index[0]],
                    markersize=5)
        else:    
            ax.plot(X_time_series[:number_of_iterations-1, index], 
                    X_time_series[1:number_of_iterations, index], 
                    'o', 
                    label = r'{}'.format(index),
                    color = col[index],
                    markersize=5)
    
    ax.legend(loc = 0, ncols = 5)
    ax.set_ylabel(r'$y(t + 1)$')
    ax.set_xlabel(r'$y(t)$')
    
def plot_density(X_time_series, 
                 cluster_list, 
                 labels = None, 
                 filename = None,
                 discrete = True):
    
    
    # Plot the density of the time series
    number_of_iterations = X_time_series.shape[0]
    t = np.arange(0, number_of_iterations, 1, dtype = int)
    N = X_time_series.shape[1]
    nodelist = np.arange(0, N, 1, dtype = int)
    
    # Set labels default
    if labels is None:
        labels = ['{}'.format(id_node) for id_node in nodelist]
        
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    fig, ax = plt.subplots(1, 1, figsize = (4, 4), dpi = 300)
   
    if discrete:
        col = sns.color_palette("deep", n_colors = N)
    else:
        col = sns.color_palette('Blues', n_colors = N)
    
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
        ax.plot(interval, 
                kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                label="{}".format(labels[index]),
                color = col[index])
    
    for id_index, indices in enumerate(cluster_list):
        if indices.shape[0] > 1:
            Opto_orbit = X_time_series[:, indices].T.flatten()
            kernel = stats.gaussian_kde(Opto_orbit, bw_method = 5e-2)
    
            ax.plot(interval,
                    kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                    '--', color = list_colors_gray[id_index])
    
    ax.set_ylabel(r'$\rho$')
    ax.set_xlabel(r'$x$')
    l = lower_bound
    u = upper_bound
    ax.set_xlim(l, u)
    if discrete:
        plt.legend(loc = 0)

    if filename == None:
        plt.show()
    else:
        folder = 'Figures'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        output = os.path.join(out_direc, filename)
        plt.savefig(output+".pdf", format = 'pdf', bbox_inches='tight')
    
        plt.show()
    return 
        
def plot_succ_max(s_t, v_t, degree_int = 3, Bspline = False, 
                  fig = None, 
                  labels = [r'True', r'Reconstructed', r'spl - True', r'spl - Reconstructed']):
    
    # Compute the spline of the succesive maxima map
    
    # Model
    z_max_v_t = maximum_over_z(v_t)
        
    if z_max_v_t.shape[0] >= degree_int:
    
        Z_max_v_t = np.array([z_max_v_t[0:-1], z_max_v_t[1:]]).T
        Z_max_v_t = Z_max_v_t[Z_max_v_t[:, 0].argsort()]
        
        # Original trajectory
        z_max_s_t = maximum_over_z(s_t)
            
        Z_max_s_t = np.array([z_max_s_t[0:-1], z_max_s_t[1:]]).T
        Z_max_s_t = Z_max_s_t[Z_max_s_t[:, 0].argsort()]
        
        if fig is None:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        else:
            ax = fig.subplots()

        ax.plot(z_max_s_t[:-1], z_max_s_t[1:], 'o', label = labels[0],
                 color=colors[0], markersize=2.5)
            
        ax.plot(z_max_v_t[:-1], z_max_v_t[1:], 'o', label = labels[1],
                 color=colors[1], markersize=2.5)
        
        ax.set_xlabel(r'$z_{n}$')
        ax.set_ylabel(r"$z_{n + 1}$")
        ax.set_title(r'Successive maxima map of the $z(t)$ variable')
        
        if Bspline:
            #We compute the interpolation using BSplines
            tspl, c, k = interpolate.splrep(Z_max_v_t[:, 0], Z_max_v_t[:, 1], 
                                            s=0.02, 
                                            k= degree_int)
            
            spline_v_t = interpolate.BSpline(tspl, c, k, extrapolate=True)
            
            #We compute the interpolation using BSplines
            tspl, c, k = interpolate.splrep(Z_max_s_t[:, 0], Z_max_s_t[:, 1], 
                                            s=0.02, 
                                            k= degree_int)
        
            spline_s_t = interpolate.BSpline(tspl, c, k, extrapolate=True)
    
        
    
            # Plotting the reference BSpline and the trajectory
            N = 1000
            xmin, xmax = np.max([z_max_s_t.min(), z_max_v_t.min()]), np.min([z_max_s_t.max(), z_max_v_t.max()])
            xx = np.linspace(xmin, xmax, N)
            
            ax.plot(xx, spline_s_t(xx), '--', 
                    label = labels[2],
                    color=colors[0], alpha=0.5)
            
            ax.plot(xx, spline_v_t(xx), '--', 
                    label = labels[3],
                    color=colors[1], alpha=0.50)
        
        ax.legend(loc = 0, fontsize = 10)


def plot_psd(X_t, u_t, dt, 
             nperseg = None, fig = None, labels = [r'True', r'Reconstructed']):
    
    title_names = [r'x-component', r'y-component', r'z-component']
    ncols = X_t.shape[0]
    
    if fig is None:
        fig, ax = plt.subplots(1, ncols, sharey=True, figsize=(8, 4), dpi=300)
    else:
        ax = fig.subplots(1, ncols, sharey=True)
            
    for id_row in range(X_t.shape[0]):
    
        f_X_t, psd_X_t = welch(X_t[id_row, :], 
                               window = 'boxcar',
                               fs = 1/dt, nperseg=nperseg, scaling='density')
        
        id_f_stop = int(f_X_t.shape[0]/2)
        if ncols > 1:
            ax[id_row].semilogy(f_X_t[:id_f_stop], psd_X_t[:id_f_stop], 
                         color=colors[0], 
                         label=labels[0])
            
        else:
            ax.semilogy(f_X_t[:id_f_stop], psd_X_t[:id_f_stop], 
                         color=colors[0], 
                         label=labels[0])
            
        f_u_t, psd_u_t = welch(u_t[id_row, :], 
                               window = 'boxcar', 
                               fs = 1/dt, nperseg=nperseg, scaling='density')
        
        if ncols > 1:
            ax[id_row].semilogy(f_u_t[:id_f_stop], psd_u_t[:id_f_stop], 
                         color=colors[1], 
                         label=labels[1])
            
            if id_row == 0:
                ax[id_row].legend()
                ax[id_row].set_ylabel(r"$PSD$")
            
            ax[id_row].set_xlabel(r'Frequency')
            ax[id_row].set_ylim(1e-5,1e2)
            ax[id_row].set_title(title_names[id_row])
        #ax[id_row].set_xlim(0, 0.3)
        else:
            ax.semilogy(f_u_t[:id_f_stop], psd_u_t[:id_f_stop], 
                         color=colors[1], 
                         label=labels[1])
            
            if id_row == 0:
                ax.legend()
                ax.set_ylabel(r"$PSD$")
            
            ax.set_xlabel(r'Frequency')
            ax.set_ylim(1e-7,1e1)
    
def fig_top_stat(X_t, u_t, dt, nperseg, filename = None):
    
    fig = plt.figure(figsize=(10, 4), dpi = 300)
    
    (fig1, fig2) = fig.subfigures(1, 2, width_ratios = [1.0, 1.6])
    
    plot_succ_max(X_t, u_t, fig = fig1)
        
    # Plot training and testing 
    plot_psd(X_t, u_t, dt, nperseg = nperseg, fig = fig2)
    fig2.suptitle('Power spectrum density comparison')
    
    if filename == None:
        plt.show()
    else:
        folder = 'Figures'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        output = os.path.join(out_direc, filename)
        plt.savefig(output+".pdf", format = 'pdf', bbox_inches='tight')
    
    plt.show()
    return 


def fig_x_coord_reconstr(X_t, u_t, t, dt, scale,
                         transient_plot, nperseg, filename = None):
    
    fig = plt.figure(figsize=(8, 3), dpi = 300)
    
    (fig1, fig2) = fig.subfigures(1, 2, width_ratios = [1.0, 1.0])
    
    plot_testing(X_t, u_t, t, transient_plot, scale = scale, fig = fig1)

    plot_psd(X_t, u_t, dt, nperseg = nperseg, fig = fig2)
    fig2.suptitle('Power spectrum density comparison')
    
    if filename == None:
        plt.show()
    else:
        folder = 'Figures'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        output = os.path.join(out_direc, filename)
        plt.savefig(output+".pdf", format = 'pdf', bbox_inches='tight')
    
    plt.show()
    return 

def fig_2d_top_stat(X_t, u_t, pair, dt, 
                    nperseg, 
                    filename = None, 
                    titles = [r'True attractor', r'Reconstructed attractor'],
                    labels_succ_max = [r'True', r'Reconstructed', r'spl - True', r'spl - Reconstructed'],
                    labels_psd = [r'True', r'Reconstructed']):
    
    fig = plt.figure(figsize=(9.0, 8.0), dpi = 300)
    
    (fig1, fig2) = fig.subfigures(2, 1, height_ratios = [1., 1.2])
    
    (ax1, ax2) = fig1.subplots(1, 2, sharey=True, 
                               gridspec_kw={
                                "bottom": 0.18,
                                "top": 0.80,
                                "left": 0.18,
                                "right": 0.80,
                                "wspace": 0.1
                                })
    ax = [ax1, ax2]
    plot_2d(X_t[pair[0], :], X_t[pair[1], :], 
            u_t[pair[0], :], u_t[pair[1], :], 
            pair, 
            transient_plot = 0,
            ax = ax,
            titles = titles)
    
    fig_tot_stat = fig2.subfigures(1, 2, width_ratios = [1.0, 1.5])
    
    plot_succ_max(X_t, u_t, fig = fig_tot_stat[0],
                  labels=labels_succ_max)
        
    # Plot training and testing 
    plot_psd(X_t, u_t, dt, nperseg = nperseg, fig = fig_tot_stat[1],
             labels=labels_psd)
    fig_tot_stat[1].suptitle('Power spectrum density comparison')
    
    if filename == None:
        plt.show()
    else:
        folder = 'Figures'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        output = os.path.join(out_direc, filename)
        plt.savefig(output+".pdf", format = 'pdf', bbox_inches='tight')
    
    plt.show()
    
def plot_metric_fun(x_axis, optf_array, key_metric, x_axis_label, title):    
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=300)
    
    ax.violinplot(optf_array.T,
                  showmeans=True,
                  showmedians=True)
    ax.set_ylabel(r'{}'.format(key_metric))
    ax.set_xlabel(r'{}'.format(x_axis_label))
    
    ax.set_xticks([y + 1 for y in range(len(x_axis))],
                 labels=[r"${}$".format(y) for y in x_axis])
    
    ax.set_title(r'{}'.format(title))

def plot_metric_reg(x_axis, 
                    res_dict, 
                    plot_dict, 
                    ax = None,
                    plot_panel = False):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=300)
    
    for id_key, key in enumerate(res_dict.keys()):
        
        optf_array = res_dict[key]
        
        mean_array, std_array = optf_array.mean(axis = 1), optf_array.std(axis = 1)
        
        ax.errorbar(x_axis, 
                    mean_array, 
                    std_array,
                    color = list_colors[id_key],
                    fmt = '-'+list_markers[id_key+1],
                    linewidth = 1,
                    capsize = 6,
                    label=r'$r_{}$ = {}'.format('{\max}',key))
    
    ax.set_ylabel(r'{}'.format(plot_dict['y_label']))
    ax.set_xlabel(r'{}'.format(plot_dict['x_label']))
    ax.set_xscale(plot_dict['x_scale'])        
    ax.set_yscale(plot_dict['y_scale'])
    ax.set_xlim(plot_dict['x_lim'][0], plot_dict['x_lim'][1])
    if plot_dict['legend']:            
        ax.legend(loc = 0, ncols = 1, fontsize = 10)
    
    if plot_panel:
        return ax
    
def plot_metric_box_plot(x_axis, 
                         res, 
                         plot_dict, 
                         ax = None,
                         plot_panel = False,
                         positions = None):
    
    if 'color' in plot_dict:
        boxprops = dict(linestyle='-', color=plot_dict['color'], 
                        facecolor = plot_dict['color'])
    else:
        boxprops = dict(linestyle='-', color=list_colors[0], facecolor = list_colors[0])
    
    medianprops = dict(linestyle='-', color='black')
    flierprops = dict(marker='o', markersize=4, markeredgecolor='gray')
    
    if positions is None:
        positions = np.arange(1, x_axis.shape[0] + 1, 1, dtype = int)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=300)
    
    optf_array = res.T.copy()
    
    array = []
    for id_ts in range(optf_array.shape[1]):
        mask = (optf_array[:, id_ts] >= 0)
    
        array.append(optf_array[:, id_ts][mask])        
    
    ax.boxplot(array, 
               patch_artist=True, boxprops= boxprops, medianprops= medianprops,
               flierprops = flierprops,
               positions = positions,
               label = plot_dict['label'])

    ax.set_ylabel(r'{}'.format(plot_dict['y_label']))
    ax.set_xlabel(r'{}'.format(plot_dict['x_label']))
    ax.set_xscale(plot_dict['x_scale'])        
    ax.set_yscale(plot_dict['y_scale'])
    #ax.set_xlim(plot_dict['x_lim'][0], plot_dict['x_lim'][1])
    ax.set_ylim(plot_dict['y_lim'][0], plot_dict['y_lim'][1])
    
    
    labels = np.concatenate(([r'$0$'], [fr'$10^{{{int(np.log10(val))}}}$' for val in x_axis[1:]]))
    ax.set_xticks(positions, 
                  labels = labels)              
    
    if plot_dict['legend']:            
        ax.legend(loc = 0, ncols = 1, fontsize = 10)
    
    if plot_panel:
        return ax    
    
def plot_fig_metrics_reg(res_dict, 
                         x_axis,
                         ax = None,
                         plot_dict = None,
                         filename = None,
                         reference_value = False):
    
    positions = np.linspace(1, x_axis.shape[0] + 5, x_axis.shape[0])
    
    if plot_dict is None:
        plot_dict = dict()
        
    nrows = len(list(res_dict.keys()))
    
    if ax is None:
        if nrows > 1:
            fig, ax = plt.subplots(nrows, 1, sharex= True, figsize = (5, 7), dpi = 300)
        else:    
            fig, ax = plt.subplots(nrows, 1, sharex= True, figsize = (4, 2), dpi = 300)
    
    for id_key, key_metric in enumerate(res_dict.keys()):
        
        if key_metric == 'sigvals':
            plot_dict['y_label'] = r'$\kappa(\hat{\Psi})$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e4, 1e18]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
            # Here we add a reference value. Location is the stats from the file: results/ref_k_1_normalized
            if reference_value:
                ref_dict = dict()
                ref_dict['h1'] = 8496.46152179
                ref_dict['h2'] = 10494.85460934 
                ref_dict['h3'] = 12474.29117526
                        
        if key_metric == 'VPT_test':
            plot_dict['y_label'] = r'$VPT$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e-2, 15]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
            # Here we add a reference value. Location is the stats from the file: results/ref_k_1
            if reference_value:
                ref_dict = dict()
                ref_dict['x1'] = 7.0
                ref_dict['x2'] = 8.0
                
            
        if key_metric == 'zmax_test':
            plot_dict['y_label'] = r'Distance - Succ Max Map'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e-3, 1e2]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
            if reference_value:
                # Here we add a reference value. Location is the stats from the file: results/ref_k_1
                ref_dict = dict()
                ref_dict['x1'] = 7.0
                ref_dict['x2'] = 8.0
                
                
        if key_metric == 'abs_psd_test':
            plot_dict['y_label'] = r'$E$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e-3, 1e1]
            plot_dict['x_label'] = r'$\beta$'
            plot_dict['legend'] = False
            plot_dict['label'] = ''   
            if reference_value:
                # Here we add a reference value. Location is the stats from the file: results/ref_k_1
                ref_dict = dict()
                ref_dict['x1'] = 7.0
                ref_dict['x2'] = 8.0
                
                
        if nrows > 1:
            ax[id_key] = plot_metric_box_plot(x_axis, 
                                              res_dict[key_metric], 
                                              plot_dict, 
                                              ax = ax[id_key],
                                              plot_panel = True,
                                              positions = positions)
            
            if reference_value:
                x_axis_ = positions
                '''
                ax[id_key].vlines(ref_dict['h2'], 1e-3, 
                                  ref_dict['y'], 
                                  list_colors[1],
                                  linestyles = 'dashed')
                '''
                ax[id_key].fill_between(np.arange(ref_dict['x1'], ref_dict['x2'], 0.01), 
                                        plot_dict['y_lim'][0], plot_dict['y_lim'][1],
                                        color = list_colors[1], alpha = 0.40)

                ax[id_key].fill_between(np.arange(0.5, 7, 0.01), 
                                        plot_dict['y_lim'][0], plot_dict['y_lim'][1],
                                        color = list_colors[2], alpha = 0.20)
                
                ax[id_key].fill_between(np.arange(8, 14.5, 0.01), 
                                        plot_dict['y_lim'][0], plot_dict['y_lim'][1],
                                        color = list_colors[4], alpha = 0.20)
            
        else:
            ax = plot_metric_box_plot(x_axis, 
                                      res_dict[key_metric], 
                                      plot_dict, 
                                      ax = ax,
                                      plot_panel = True)
            
          
    if filename == None:
        return ax
    else:
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')

def plot_fig_cond_num_time_skip(res_dict, 
                                x_dict, 
                                ref_values,
                                ax = None,
                                plot_dict = None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
       
    plot_dict['y_label'] = r'$\kappa(\hat{\Psi})$'
    plot_dict['x_scale'] = 'linear'
    plot_dict['y_scale'] = 'log'
    plot_dict['x_label'] = r'$\tau$'
    plot_dict['color'] = list_colors[0]
    
    optf_array = res_dict['sigvals']
    
    lower, median, upper  = np.array([np.quantile(optf_array, 0.25, axis = 1), 
                                      np.quantile(optf_array, 0.5, axis = 1), 
                                      np.quantile(optf_array, 0.75, axis = 1)])
    
    ax.plot(x_dict, 
            median,
            '-',
            color = plot_dict['color'],
            linewidth=1.0)

    ax.fill_between(x_dict, lower, median,
                    color = plot_dict['color'], 
                    alpha = 0.25)

    ax.fill_between(x_dict, median, upper,
                    color = plot_dict['color'], 
                    alpha = 0.25)
      
    ax.hlines(ref_values['h2'], x_dict[0], 
              x_dict[-1], 
              list_colors[1],
              linestyles = 'dashed')
    
    ax.fill_between(x_dict, ref_values['h1'], ref_values['h3'],
                    color = list_colors[1], alpha = 0.25)
    
    ax.set_xlabel(r'{}'.format(plot_dict['x_label']))
    ax.set_xscale(plot_dict['x_scale'])        
    ax.set_yscale(plot_dict['y_scale'])
    ax.set_ylabel(r'{}'.format(plot_dict['y_label']))
    
    if ax == None:
        plt.tight_layout()
        plt.show()
    else:
        return ax

def plot_fig_metrics_time_skip(res_dict, 
                               x_axis,
                               plot_dict = None,
                               filename = None,
                               reference_value = False,
                               ax = None):
    
    if plot_dict is None:
        plot_dict = dict()
        
    nrows = len(list(res_dict.keys()))
    
    if ax is None:
        if nrows > 1:
            fig, ax = plt.subplots(nrows, 1, sharex= True, figsize = (5, 7), dpi = 300)
        else:    
            fig, ax = plt.subplots(nrows, 1, sharex= True, figsize = (4, 2), dpi = 300)
    
    
    for id_key, key_metric in enumerate(res_dict.keys()):
        
        if key_metric == 'sigvals':
            plot_dict['y_label'] = r'$\kappa(\hat{\Psi})$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e2, 1e17]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
            # Here we add a reference value. Location is the stats from the file: results/ref_k_1_normalized
            if reference_value:
                ref_dict = dict()
                ref_dict['h1'] = 8496.46152179
                ref_dict['h2'] = 10494.85460934 
                ref_dict['h3'] = 12474.29117526
                        
        if key_metric == 'VPT_test':
            plot_dict['y_label'] = r'$VPT$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'linear'
            plot_dict['y_lim'] = [-1, 15]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
            # Here we add a reference value. Location is the stats from the file: results/ref_k_1
            if reference_value:
                ref_dict = dict()
                ref_dict['h1'] = 8.397176
                ref_dict['h2'] = 9.187312
                ref_dict['h3'] = 9.932168
            
        if key_metric == 'zmax_test':
            plot_dict['y_label'] = r'Distance - Succ Max Map'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e-2, 1e3]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
            if reference_value:
                # Here we add a reference value. Location is the stats from the file: results/ref_k_1
                ref_dict = dict()
                ref_dict['h1'] = 0.07299443
                ref_dict['h2'] = 0.12289242
                ref_dict['h3'] = 0.28731724
               
        if key_metric == 'abs_psd_test':
            plot_dict['y_label'] = r'$E$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e-3, 1e1]
            plot_dict['x_label'] = r'$\tau$'
            plot_dict['legend'] = False
            plot_dict['label'] = ''  
            if reference_value:
                # Here we add a reference value. Location is the stats from the file: results/ref_k_1
                ref_dict = dict()
                ref_dict['h1'] = 0.01084659
                ref_dict['h2'] = 0.01682403
                ref_dict['h3'] = 0.02656172
                
        if nrows > 1:
            ax[id_key] = plot_metric_box_plot(x_axis, 
                                              res_dict[key_metric], 
                                              plot_dict, 
                                              ax = ax[id_key],
                                              plot_panel = True)
            
            if reference_value:
                x_axis_ = np.arange(1, x_axis.shape[0] + 1, 1, dtype = int)
                ax[id_key].hlines(ref_dict['h2'], x_axis_[0], x_axis_[-1], list_colors[1],
                                  linestyles = 'dashed')
                
                ax[id_key].fill_between(x_axis_, ref_dict['h1'], ref_dict['h3'],
                                        color = list_colors[1], alpha = 0.25)

        else:
            ax = plot_metric_box_plot(x_axis, 
                                      res_dict[key_metric], 
                                      plot_dict, 
                                      ax = ax,
                                      plot_panel = True)
            
            if reference_value:
                x_axis_ = np.arange(1, x_axis.shape[0] + 1, 1, dtype = int)
                ax.hlines(ref_dict['h2'], x_axis_[0], x_axis_[-1], list_colors[1],
                          linestyles = 'dashed')
                
                ax.fill_between(x_axis_, ref_dict['h1'], ref_dict['h3'],
                                color = list_colors[1], alpha = 0.25)
                
            
    if filename == None:
        return ax
    
    else:
        if filename is not None:
            folder = 'Figures/'
            out_direc = os.path.join('', folder)
            
            if os.path.isdir(out_direc) == False:
                os.makedirs(out_direc)
            filename = os.path.join(out_direc, filename)
            
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')

def plot_fig_metrics_lgth_train(res_dict, 
                                x_dict, 
                                plot_dict_ = None,
                                filename = None,
                                reference_value = False):
    
    ncols = len(list(res_dict.keys()))
    
    if ncols > 1:
        fig, ax = plt.subplots(1, ncols, sharey = True, figsize = (7, 3), dpi = 300)
        
    else:    
        fig, ax = plt.subplots(1, ncols, figsize = (4, 2), dpi = 300)
    
    for id_exp, exp in enumerate(res_dict.keys()):
        
        if plot_dict_ is None:
            plot_dict = dict()
        else:
            plot_dict = plot_dict_[exp]
            
        # Inset with the results    
        
        axins = ax[id_exp].inset_axes(
                [0.45, 0.45, 0.5, 0.5], xticklabels=[], yticklabels=[])
        
        for id_key, key_metric in enumerate(res_dict[exp].keys()):   
            for id_key_1, key_1 in enumerate(res_dict[exp][key_metric].keys()):
                if key_metric == 'sigvals':
                    plot_dict['y_label'] = r'$\kappa(\hat{\Psi})$'
                    if id_exp == 0:
                        plot_dict['x_scale'] = 'log'
                    else:
                        plot_dict['x_scale'] = 'log'
                    plot_dict['y_scale'] = 'log'
                    plot_dict['x_label'] = r'$N_{\mathrm{train}}$'
                    plot_dict['label'] = r'$h = {}$'.format(key_1)
                    plot_dict['legend'] = True
                    plot_dict['color'] = list_colors[id_key_1]
                    
                    
                    optf_array = res_dict[exp][key_metric][key_1]
                    
                    lower, median, upper  = np.array([np.quantile(optf_array, 0.25, axis = 1), 
                                                      np.quantile(optf_array, 0.5, axis = 1), 
                                                      np.quantile(optf_array, 0.75, axis = 1)])
                    
                    ax[id_exp].plot(x_dict[exp], 
                                    median,
                                    '-',
                                    color = plot_dict['color'],
                                    label=plot_dict['label'],
                                    linewidth=1.1)
                
                    ax[id_exp].fill_between(x_dict[exp], lower, median,
                                            color = plot_dict['color'], 
                                            alpha = 0.25)
                
                    ax[id_exp].fill_between(x_dict[exp], median, upper,
                                            color = plot_dict['color'], 
                                            alpha = 0.25)
                    
                    
                    axins.plot(x_dict[exp], 
                               median, 
                               '-',
                               color = plot_dict['color'],
                               label=plot_dict['label'],
                               linewidth=1.1)
                    
                    axins.fill_between(x_dict[exp], lower, median,
                                       color = plot_dict['color'], 
                                       alpha = 0.25)
                    
                    axins.fill_between(x_dict[exp], median, upper,
                                            color = plot_dict['color'], 
                                            alpha = 0.25)
                    axins.set_xscale('linear')
                    axins.set_yscale(plot_dict['y_scale'])
                    x1, x2 = x_dict[exp][0], x_dict[exp][10]  # subregion of the original image
                    y1, y2 = 1e2, 1e6  # subregion of the original image
                    axins.set_xlim(x1, x2)
                    axins.set_ylim(y1, y2)
                    
                ax[id_exp].set_xlabel(r'{}'.format(plot_dict['x_label']))
                ax[id_exp].set_xlim(plot_dict['x_lim'][0], plot_dict['x_lim'][1])
                ax[id_exp].set_xscale(plot_dict['x_scale'])        
                ax[id_exp].set_yscale(plot_dict['y_scale'])
                
                ax[id_exp].set_title(plot_dict['title'])
                
                if (plot_dict['legend']) and (id_exp == 1):            
                    ax[id_exp].legend(loc='center left', ncols = 1, 
                                      bbox_to_anchor=(1, 0.5))
        
                
    ax[0].set_ylabel(r'{}'.format(plot_dict['y_label']))
    
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')

def plot_cond_number(res_dict,
                     filename = None,
                     ax = None,
                     labels = None):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=300)
    
    keys = list(res_dict.keys())
    Nseeds = len(list(res_dict[keys[0]]))
    
    cond_num_seeds = np.zeros((Nseeds, len(keys)))
    for id_key, key in enumerate(keys):
        
        sigvals_dict = res_dict[key]
        
        for id_seed, seed in enumerate(sigvals_dict.keys()):
            cond_num_seeds[id_seed, id_key] = np.max(sigvals_dict[seed])/np.min(sigvals_dict[seed])
        
    
    min_cond = np.min(cond_num_seeds)    
    max_cond = np.max(cond_num_seeds)    
    
    n_bins = 100
    logbins = np.logspace(np.log10(min_cond/10), np.log10(max_cond*10), n_bins+1)
    
    for id_key, key in enumerate(keys):
        
        if labels is None:
            labels = r'{}'.format(key)
        
        if id_key == 0:
            color = list_colors[1]
        
        else:
            color = list_colors_gray[id_key - 1]
            
        ax.hist(cond_num_seeds[:, id_key],
                bins= logbins,
                alpha = 0.8,
                density= False,
                label= labels[id_key],
                color = color)
        
    ax.set_ylabel(r"count")
    ax.legend(loc=9, ncols = 1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa(\hat{\Psi})$")
    
    ax.set_xlim(1e0, 1e20)
    x_ticks = [1e2, 1e6, 1e10, 1e14, 1e18]
    ax.set_xticks(x_ticks, 
                  labels = [fr'$10^{{{int(np.log10(val))}}}$' for val in x_ticks]) 
                          
    
    if filename != None:
        plt.savefig(filename+".pdf", format = 'pdf')
    
    if filename is None:
        return ax
    
    
def plot_cond_num_poly_deg(res, 
                           x_axis,
                           ax = None,
                           plot_dict = None,
                           filename = None):
    
        
    if filename is not None:
        folder = 'Figures/'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        filename = os.path.join(out_direc, filename)
    
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    
    
    plot_dict['y_label'] = r'$\kappa(\hat{\Psi})$'
    plot_dict['y_scale'] = 'log'
    plot_dict['x_scale'] = 'linear'
    plot_dict['y_lim'] = [1e1, 1e13]
    plot_dict['x_label'] = r'$p$'
    plot_dict['legend'] = False
    plot_dict['label'] = ''
        
    ax = plot_metric_box_plot(x_axis, 
                              res, 
                              plot_dict, 
                              ax = ax,
                              plot_panel = True,
                              positions = None)
    
    
    results = res.T.copy()
    
    
    from scipy import optimize
    
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    y_data = np.quantile(results, 0.5, axis=0)
    
    logy = np.log(y_data)
    x = np.arange(1, 10, 1)
    logyerr = 1
    pinit = [1.0, 1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(x, logy, logyerr), full_output=1)

    pfinal = out[0]
    covar = out[1]
    print('pfinal-n_0', pfinal)
    print('covar-n_0', covar)

    amp = pfinal[1]

    x_ = np.arange(1, 9.1, 0.01)
    ax.plot(x_, np.exp(1 + amp * x_), ls = 'dashed', 
            color='k',
            label = r'$\kappa(\hat{{{\Psi}}})$'fr'$~\propto e^{{{amp:.0f} p}}$',
            alpha = 0.5)
    
    
    #f = lambda x: np.power(2/(x + 1), 0.5)*np.power(1 + np.sqrt(2), x - 1)
    
    #ax.plot(x_, f(x_), 'r--')
    
    ax.legend(loc = 0, fontsize = 12)
    if filename != None:
        plt.savefig(filename+".pdf", format = 'pdf', bbox_inches='tight')
    
    if filename is None:
        plt.tight_layout()
        plt.show()
    
    
def fig_cond_number_delay_poly(res_dicts, labels_dict, titles_dict, filename = None):

    keys = list(res_dicts.keys())
    num_cols = len(keys)
    
    fig, ax = plt.subplots(1, num_cols, figsize=(6, 3), dpi=300)

    for id_col, key in enumerate(keys):
        ax[id_col] = plot_cond_number(res_dicts[key],
                                      filename = None,
                                      ax = ax[id_col],
                                      labels = labels_dict[key])
        
        ax[id_col].set_title(titles_dict[key])
        
    ax[id_col].set_ylabel(r"")
    
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
        
        folder = 'Figures/'
        out_direc = os.path.join('', folder)
            
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        filename = os.path.join(out_direc, filename)
            
        plt.savefig(filename+".pdf", format = 'pdf', bbox_inches='tight')

    return    

def fig_cond_number_delay(res_dicts, 
                          labels_dict, 
                          titles_dict,
                          plot_dict,
                          filename = None):

    keys = list(res_dicts.keys())
    num_cols = len(keys)
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=300, layout='constrained')

    
    ax[0] = plot_cond_number(res_dicts['delay_dimension'],
                             filename = None,
                             ax = ax[0],
                             labels = labels_dict['delay_dimension'])
        
    #ax[0].set_title(titles_dict['delay_dimension'])
    ref_vals = dict()
    sigvals = np.zeros(100)
    for id_key, key in enumerate(res_dicts['delay_dimension'][1].keys()):
        sigval = res_dicts['delay_dimension'][1][key]
        sigvals[id_key] = sigval.max()/sigval.min()
    
    ref_vals['h1'] = np.quantile(sigvals, 0.25)
    ref_vals['h2'] = np.quantile(sigvals, 0.5)
    ref_vals['h3'] = np.quantile(sigvals, 0.75)
    
    ax[1] = plot_fig_cond_num_time_skip(res_dicts['time_skip'], 
                                        plot_dict['time_skip']['x_axis'],
                                        ref_vals,
                                        ax = ax[1],
                                        plot_dict = plot_dict)
       
    axins = ax[1].inset_axes(
            [0.40, 0.40, 0.55, 0.55], xticklabels=[], yticklabels=[])
    
    
    sigvals = res_dicts['x_coord']['sigvals']
    
    ref_vals['h1'] = np.quantile(sigvals, 0.25, axis = 1)
    ref_vals['h2'] = np.quantile(sigvals, 0.5, axis = 1)
    ref_vals['h3'] = np.quantile(sigvals, 0.75, axis = 1)
    
    axins.plot(plot_dict['time_skip']['x_axis'], 
               ref_vals['h2'], 
               '-',
               color = list_colors[0],
               linewidth=1.0)
    
    axins.fill_between(plot_dict['time_skip']['x_axis'], 
                       ref_vals['h1'], ref_vals['h2'],
                       color = list_colors[0], 
                       alpha = 0.25)
    
    axins.fill_between(plot_dict['time_skip']['x_axis'], 
                       ref_vals['h2'], ref_vals['h3'],
                       color = list_colors[0],
                       alpha = 0.25)
    
    axins.set_xscale('linear')
    axins.set_yscale('linear')
    
    
    a0, a1, a2 = 1.51, 0.21, 0.5#0.9695, 0.01 #1.02, 0.015, 1
    #f = lambda x, a0, a1, a2: np.sqrt((1 + a0*np.exp(-a1*x**(a2)))/(1 - a0*np.exp(-a1*x**(a2))))
    g = lambda x, a0, a1: np.sqrt((1 + a0*np.exp(-a1*x))/(1 - a0*np.exp(-a1*x)))
    
    x = np.arange(1, plot_dict['time_skip']['x_axis'][-1], 0.01)
    
    popt, pcov = curve_fit(g, plot_dict['time_skip']['x_axis'], ref_vals['h2'], 
                           p0 = [1.0, 0.05])
    
    print('a0, a1, a2')
    print(*[f"{val:.2f}+/-{err:.2f}" for val, err in zip(popt, np.sqrt(np.diag(pcov)))])
    print('\n')

    axins.plot(x, g(x, *popt), 'k--', alpha = 0.4)
    #axins.plot(x, g(x, a0, a1, a2), 'k--', alpha = 0.4)
    
    axins.set_xlim(0, 50)
    axins.set_ylim(0, 40)
    axins.text(15, 30, r'$\sqrt{\frac{1 + K e^{-\gamma \tau}}{1 - K e^{-\gamma \tau}}}$',
               fontsize = 14)
    axins.set_xlabel(r'$\tau$')
    axins.set_ylabel(r'$\kappa(\hat{\Psi}_{x})$')
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
        
        folder = 'Figures/'
        out_direc = os.path.join('', folder)
            
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        filename = os.path.join(out_direc, filename)
            
        plt.savefig(filename+".pdf", format = 'pdf', bbox_inches='tight')

    return        

def fig_metrics_lgth_train(results_dict, 
                           x_dict,
                           plot_dict = None,
                           filename = None):
    
        
    if filename is not None:
        folder = 'Figures/'
        out_direc = os.path.join('', folder)
        
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        filename = os.path.join(out_direc, filename)
    
    plot_fig_metrics_lgth_train(results_dict, 
                                x_dict,  
                                plot_dict_ = plot_dict,
                                filename = filename)   
    

def fig_reg_solver(results_dict, x_axis, filename = None):

    
    keys = list(results_dict.keys())
    
    fig, ax = plt.subplots(3, 3, sharex=True,
                           figsize=(12, 6), dpi=300, layout='constrained')

    
    for id_key, key in enumerate(keys):
        
        ax[:, id_key] = plot_fig_metrics_reg(results_dict[key], 
                                             x_axis, ax[:, id_key], 
                                             plot_dict = None,
                                             reference_value=True)
        
        ax[0, id_key].set_title(fr'{key}')
        
        if id_key >= 1:
            ax[0, id_key].set_ylabel(r'')
            ax[1, id_key].set_ylabel(r'')
            ax[2, id_key].set_ylabel(r'')
        
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
        
        folder = 'Figures/'
        out_direc = os.path.join('', folder)
            
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        filename = os.path.join(out_direc, filename)
            
        plt.savefig(filename+".pdf", format = 'pdf', bbox_inches='tight')

        
    return     
    
def fig_x_coord_time_skip(res_dict, 
                          x_dict,
                          plot_dict = None,
                          filename = None):
    
    if plot_dict is None:
        plot_dict = dict()
        plot_dict['color'] = list_colors[0]
        
    nrows = len(list(res_dict.keys()))
    
    if nrows > 1:
        fig, ax = plt.subplots(nrows, 1, sharex= True, figsize = (4, 6), dpi = 300)
    else:    
        fig, ax = plt.subplots(nrows, 1, sharex= True, figsize = (4, 2), dpi = 300)
    
    for id_key, key_metric in enumerate(res_dict.keys()):
        
        if key_metric == 'sigvals':
            plot_dict['y_label'] = r'$\kappa(\Psi)$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e2, 1e14]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
            
            
            
        if key_metric == 'VPT_test':
            plot_dict['y_label'] = r'$VPT$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'linear'
            plot_dict['y_lim'] = [-1e0, 8]
            plot_dict['x_label'] = ''
            plot_dict['legend'] = False
            plot_dict['label'] = ''
                
        if key_metric == 'abs_psd_test':
            plot_dict['y_label'] = r'$E$'
            plot_dict['x_scale'] = 'linear'
            plot_dict['y_scale'] = 'log'
            plot_dict['y_lim'] = [1e-3, 1e0]
            plot_dict['x_label'] = r'$\tau$'
            plot_dict['legend'] = False
            plot_dict['label'] = ''   
       
    
        optf_array = res_dict[key_metric][id_key]
        
        lower, median, upper  = np.array([np.quantile(optf_array, 0.25, axis = 1), 
                                          np.quantile(optf_array, 0.5, axis = 1), 
                                          np.quantile(optf_array, 0.75, axis = 1)])
        
        if id_key == 0:
            ax[id_key].plot(x_dict, 
                    median,
                    '-',
                    color = plot_dict['color'],
                    linewidth=1.1)
    
            ax[id_key].fill_between(x_dict, lower, median,
                                    color = plot_dict['color'], 
                                    alpha = 0.25)
    
            ax[id_key].fill_between(x_dict, median, upper,
                                    color = plot_dict['color'], 
                                    alpha = 0.25)
            
            
        else:
            
            error = np.zeros((2, x_dict.shape[0]))
            error[0, lower >= 0] = lower[lower >= 0]
            error[1, upper >= 0] = upper[upper >= 0]
            
            ax[id_key].errorbar(x_dict, 
                                median,
                                yerr = error,
                                color = list_colors[0],
                                fmt = list_markers[0],
                                capsize=3,
                                ms = 8)
        
        ax[id_key].fill_between(np.arange(13, 17), 
                                plot_dict['y_lim'][0], plot_dict['y_lim'][1],
                                color = list_colors[1], 
                                alpha = 0.25)
        
        
        ax[id_key].set_xlabel(r'{}'.format(plot_dict['x_label']))
        ax[id_key].set_xscale(plot_dict['x_scale'])        
        ax[id_key].set_yscale(plot_dict['y_scale'])
        ax[id_key].set_ylabel(r'{}'.format(plot_dict['y_label']))
    
    labels = [1e2, 1e6, 1e10, 1e14]
    ax[0].set_yticks(labels, 
                     labels = [fr'$10^{{{int(np.log10(val))}}}$' for val in labels])
    
    
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
        
        folder = 'Figures/'
        out_direc = os.path.join('', folder)
            
        if os.path.isdir(out_direc) == False:
            os.makedirs(out_direc)
        filename = os.path.join(out_direc, filename)
            
        plt.savefig(filename+".pdf", format = 'pdf', bbox_inches='tight')
    
    
def num_poly_basis(delays, max_degree, dimension):
    
    return np.round(scipy.special.comb(delays*dimension + max_degree, max_degree))

def upper_bound_num_poly_basis(delays, max_degree, dimension):
    return np.power((delays*dimension + max_degree)/delays*dimension,delays*dimension)
    
def plot_proj_num_poly_basis(dimension = 3):

    delays = np.arange(1, 6, 1, dtype = int)
    max_degree = np.arange(1, 6, 1, dtype = int)
    
    fig, ax = plt.subplots(1, 1, sharey=True, dpi = 300)
    
    for id_k, k in enumerate(delays):
        
        ax.plot(max_degree, num_poly_basis(k, max_degree, dimension),
                label = r'$k = {}$'.format(k),
                color = list_colors_blue[id_k],
                marker = list_markers[len(list_markers) - id_k - 1])
    
    ax.set_xlabel(r'')
    ax.set_ylabel(r'$m_d(p, k)$')
    ax.set_yscale('log')
    ax.legend(loc=0)
    
    for id_p, p in enumerate(max_degree):
        
        ax.plot(delays, num_poly_basis(delays, p, dimension),
                label = r'$p = {}$'.format(p),
                color = list_colors_red[id_p],
                marker = list_markers[id_p])
    
    ax.set_xlabel(r'')
    
    ax.legend(loc=0, ncol= 2)
    

def plot_contour_num_poly_basis(dimension = 3):
    
    delays = np.arange(1, 6, 1, dtype = int)
    max_degree = np.arange(1, 6, 1, dtype = int)
    
    k, p = np.meshgrid(delays, max_degree)
    z = np.log(num_poly_basis(k, p, dimension))
    
    fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 300)
    cs = ax[0].contourf(k, p, z)
    ax[0].contour(cs, colors='k')
    
    # Plot grid.
    ax[0].grid(c='k', ls='-', alpha=0.3)
    
    ax[1].imshow(z)
    fig.colorbar(cs, ax=ax)
    
    plt.show()
    
    return 

#Auto correlation function for the input time series     
def x_corr(sign1, sign2, normalize = False):
    
    n = np.max([sign1.shape[0], sign2.shape[0]])
    
    if normalize:
        s1 = (sign1 - sign1.mean())/(sign1.std()*n)
        s2 = (sign2 - sign2.mean())/sign2.std()
    else:    
        s1 = (sign1)/(np.sqrt(n))
        s2 = (sign2)/np.sqrt(n)
        
    lags = signal.correlation_lags(s1.size, s2.size, mode="full")
    cx = signal.correlate(s1, s2, mode='full')
   
    return lags, cx    

def exp(t, A, lbda):
    r"""y(t) = A \cdot \exp(-\lambda t)"""
    return A * np.exp(-lbda * t)

def exp_gamma(t, A, lbda, gamma):
    r"""y(t) = A \cdot \exp(-\lambda t)"""
    return A * np.exp(-lbda * t**gamma)

def sine(t, omega, phi):
    r"""y(t) = \sin(\omega \cdot t + phi)"""
    return np.sin(omega * t + phi)

def damped_sine(t, A, lbda, gamma, omega, phi):
    r"""y(t) = A \cdot \exp(-\lambda t) \cdot \left( \sin \left( \omega t + \phi ) \right)"""
    return exp(t, A, lbda) * sine(t, omega, phi)


def plot_corr(X_t, index, normalize, id_xlim = 500, 
              labels = [r'corr - $x$', r'corr - $y$', r'corr - $z$'],
              fit = False):
    
    nrows = X_t.shape[1]
    fig, ax = plt.subplots(nrows, 1, sharex = True, dpi = 300, figsize = (5, 5))
    
    for id_ in index:
        lags, cx = x_corr(X_t[:, id_], X_t[:, id_], normalize)
        lgn = int(lags.shape[0]/2 + 1)
        
        if fit:
            popt, pcov = curve_fit(exp, lags[lgn:lgn+id_xlim], cx[lgn:lgn+id_xlim])
            
            print(labels[id_])
            print('A, lbda, gamma, omega, phi')
            print(*[f"{val:.2f}+/-{err:.2f}" for val, err in zip(popt, np.sqrt(np.diag(pcov)))])
            print('\n')
        
        if nrows > 1:
            ax[id_].plot(lags[lgn:lgn+id_xlim], np.absolute(cx[lgn:lgn+id_xlim]),
                         color = list_colors[0])
           
            ax[id_].set_ylabel(labels[id_])
        
            if normalize:
                ax[id_].set_ylim(0, 1)
        
            if fit:
                ax[id_].plot(lags[lgn:lgn+id_xlim], exp(lags[lgn:lgn+id_xlim], *popt), '--') 
            
        else:
            ax.plot(lags[lgn:lgn+id_xlim], np.absolute(cx[lgn:lgn+id_xlim]),
                         color = list_colors[0])
           
            ax.set_ylabel(labels[id_])
            ax.set_xlabel(r"$\tau$")
            
            if normalize:
                ax.set_ylim(0, 1)

            if fit:
                ax.plot(lags[lgn:lgn+id_xlim], exp(lags[lgn:lgn+id_xlim], *popt), '--') 
        
    if nrows > 1:
        ax[id_].set_xlabel(r"$\tau$")

    plt.tight_layout()


def plot_cross_corr(X_t, index, normalize, id_xlim = 100):
    
    fig, ax = plt.subplots(dpi = 300, figsize = (5, 5))
    
    lags, cx = x_corr(X_t[:, index[0]], X_t[:, index[1]], normalize)
    lgn = int(lags.shape[0]/2)
    ax.plot(lags[lgn:lgn+id_xlim], np.absolute(cx[lgn:lgn+id_xlim]),
                 color = list_colors[0])
    
    ax.set_ylabel(r'cross corr')
    ax.set_xlabel(r"$\tau$")
    
    if normalize:
        ax.set_ylim(0, 1)

    plt.tight_layout()
            
def plot_spatial_Lorenz96(X_t, t):
    
    time_stamp = int(0.5*t.shape[0])
    id_start, id_end = int(0.5*t.shape[0]), int(0.5*t.shape[0] + 0.1*t.shape[0])
    
    fig = plt.figure(dpi = 300)
    gs = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0:, 1])
    
    index = [0, 1, 2]
    for id_ in index:
        ax1.plot(t, X_t[:, id_], color = list_colors[id_])
    
    ax1.set_xlim(t[id_start], t[id_end])
    ax1.set_xlabel(r'Time')
    ax1.set_ylabel(r'$x_i(t)$')
       
    ax2.plot(X_t[time_stamp, :], color = list_colors[0])
    ax2.set_ylabel(r'$\mathbf{x}(t_*))$')
    ax2.set_xlabel(r'index')
    
    X, Y = np.meshgrid(np.arange(X_t.shape[1]), np.arange(id_end - id_start))
    pos = ax3.pcolormesh(Y.T, X.T, X_t[id_start:id_end, :].T, 
                         cmap='Blues',
                         shading = 'gouraud')
    
    ax3.set_xticks([0, np.arange(id_end - id_start)[-1]], 
                  labels = [r'${}$'.format(np.round(t[id_start])), 
                            r'${}$'.format(np.round(t[id_end]))])
    
    ax3.set_xlabel(r'Time')
    ax3.set_ylabel(r'$x_i(t)$')
    
    fig.colorbar(pos, ax=ax3)

    plt.tight_layout()

