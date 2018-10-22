#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
from lib import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'  
fs = 20.   

def make_A_s_space_3D(N_obs, s1, s2, A, ln_L):
    A_target = np.logspace(-14., -10., 100)
    A_3D, s1_3D, s2_3D, ln_L_3D = [], [], [], []
    
    for (_s1,_s2,_A,_ln_L) in zip(s1,s2,A,ln_L):
        for _At in A_target:
            _f = _At / _A
            s1_3D.append(_s1)
            s2_3D.append(_s2)
            A_3D.append(_At)
            ln_L_3D.append(_ln_L + N_obs * (1. - _f + np.log(_f)))
    return np.array(A_3D), np.array(s1_3D), np.array(s2_3D), np.array(ln_L_3D) 

class Plot_Fullpar(object):
    """
    Description:
    ------------
    Uses the fiducial sample to create a plot a likelihood contour in 3D,
    where the axis are A, s1 and s2.

    Parameters:
    -----------
    elev : ~int
        Elevation angle to be used to rotate figure axes.
    azim : ~int
        Azimutal angle to be used to rotate figure axes.
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.
             
    Outputs:
    --------
    ./../../OUTPUT_FILES/ANALYSES_FIGURES/Fig_3D.pdf
    """         
    def __init__(self, elev=0, azim=0, show_fig=True, save_fig=False):
        
        self.elev = elev
        self.azim = azim
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        fig = plt.figure(figsize=(10,10))
        self.ax = fig.add_subplot(111, projection='3d')
        
        self.xx, self.yy, self.zz = None, None, None
        
        self.make_plot()
        
    def set_fig_frame(self):
        self.ax.set_xlabel(r'$s_1$',fontsize=fs,labelpad=30)
        self.ax.set_ylabel(r'$s_2$',fontsize=fs,labelpad=30)
        self.ax.set_zlabel(r'$\mathrm{log}\, A$',fontsize=fs,labelpad=30)

        self.ax.set_xlim(-3., -0.)
        self.ax.set_ylim(-3., -0.)
        self.ax.set_zlim(-14., -10.)
        
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=16)      
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=16)        
        self.ax.tick_params(axis='z', which='major', labelsize=fs, pad=16)        
        self.ax.xaxis.set_major_locator(MultipleLocator(.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(.5))
        self.ax.zaxis.set_major_locator(MultipleLocator(.5))

    def read_and_plot_data(self):

        fpath = os.path.join(
          './../OUTPUT_FILES/RUNS/sys_Chabrier_exponential_0.0300_PADOVA'
          + '_BASEL_100_1/likelihoods/', 'sSNRL_s1_s2.csv')
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)
      
        a, x, y, l = make_A_s_space_3D(N_obs, s1, s2, A, ln_L)
        l = stats.clean_array(l)

        ln_at_sigma = stats.get_contour_levels(l, 0.68)
        #cond = (abs((l - ln_at_sigma) / ln_at_sigma) < 0.03)        
        cond = (l >= ln_at_sigma)        
        xx, yy, zz = x[cond], y[cond], a[cond]
        self.ax.scatter(xx, yy, np.log10(zz), c='r', marker='s')
        
    def manage_output(self):
        self.ax.view_init(self.elev,self.azim)
        if self.save_fig:
            directory = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_3D_'
            fpath = directory + str(self.elev) + '_' + str(self.azim) + '.pdf'
            plt.savefig(fpath , format='pdf')
        if self.show_fig:
            plt.show() 
            
    def make_plot(self):
        self.set_fig_frame()
        self.read_and_plot_data()
        self.manage_output()

if __name__ == '__main__':
    #Plot_Fullpar(elev=45, azim=45, show_fig=True, save_fig=True)
    elev_list, azim_list = [90,0,45,45], [270,90,45,135]
    for (_elev,_azim) in zip(elev_list,azim_list):
        Plot_Fullpar(elev=_elev, azim=_azim, show_fig=False, save_fig=True)
 
