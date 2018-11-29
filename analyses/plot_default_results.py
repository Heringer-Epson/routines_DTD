#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib import colors
import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fs = 24.
c = ['#1b9e77','#d95f02','#7570b3']
cmap = plt.cm.get_cmap('viridis')

def draw(fig, _ax1, _ax2, _fpath, _c):
    N_obs, s1, s2, A, ln_L = stats.read_lnL(_fpath)
    x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
    a, b, c, d = stats.plot_contour(_ax1, np.log10(x), y, z, c=_c, add_max=False)
    print a, b, c, d
    a, b, c, d = stats.plot_contour(_ax2, s1, s2, ln_L, c=_c, add_max=True)          
    print a, b, c, d
    stats.plot_A_contours(_ax2, s1, s2, A)                
    
class Make_Fig(object):
    """
    Description:
    ------------
    TBW.
  
    Parameters:
    -----------
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_method_comp.pdf

    References:
    -----------
    Maoz+ 2012 (M12): http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    Heringer+ 2017 (H17): http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    """         
  
    def __init__(self, show_fig, save_fig):
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        self.fig = plt.figure(figsize=(16,8))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.run_plot()

    def set_fig_frame(self):

        plt.subplots_adjust(left=0.125, bottom=0.125, right=0.9, top=0.9,
                            wspace=0.3)
                                    
        xlabel = r'$\mathrm{log}\, A\,\,\,\, \mathrm{[SN\ yr^{-1}\ M_\odot^{-1}]}$'
        self.ax1.set_xlabel(xlabel, fontsize=fs + 4)
        self.ax1.set_ylabel(r'$s=s_1=s_2$', fontsize=fs + 4)
        self.ax1.set_xlim(-12.8,-12.)
        self.ax1.set_ylim(-2.,-1.)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params('both', length=12, width=2., which='major',
                             direction='in', right=True, top=True)
        self.ax1.tick_params('both', length=6, width=2., which='minor',
                             direction='in', right=True, top=True) 
        self.ax1.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.xaxis.set_major_locator(MultipleLocator(.2))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.yaxis.set_major_locator(MultipleLocator(.2))  

        self.ax2.set_xlabel(r'$s_1$', fontsize=fs + 4)
        self.ax2.set_ylabel(r'$s_2$', fontsize=fs + 4)
        self.ax2.set_xlim(-3.,0.)
        self.ax2.set_ylim(-2.5,0.)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params('both', length=12, width=2., which='major',
                                 direction='in', right=True, top=True)
        self.ax2.tick_params('both', length=6, width=2., which='minor',
                                 direction='in', right=True, top=True) 
        self.ax2.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax2.xaxis.set_major_locator(MultipleLocator(.5))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax2.yaxis.set_major_locator(MultipleLocator(.5))  

        self.ax2.plot([-3., 0.], [-3., 0.], ls='--', c='k', lw=1.)

    def add_contours(self):
        
        fpath =  './../OUTPUT_FILES/RUNS/H17_S18_zSNIa_0.2/likelihoods/sSNRL_s1_s2.csv'
        draw(self.fig, self.ax1, self.ax2, fpath, c[0])

    def manage_output(self):
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_results.pdf'
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.set_fig_frame()
        self.add_contours()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(show_fig=False, save_fig=True)
