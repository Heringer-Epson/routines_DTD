#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fs = 24.
c = ['#1b9e77','#d95f02','#7570b3']

def draw(_ax1, _ax2, _fpath, _c):
    N_obs, s1, s2, A, ln_L = stats.read_lnL(_fpath)
    x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
    stats.plot_contour(_ax1, np.log10(x), y, z, c=_c, add_max=True)
    stats.plot_contour(_ax2, s1, s2, ln_L, c=_c, add_max=True)          

    
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
        
        fig = plt.figure(figsize=(16,8))
        self.ax1 = fig.add_subplot(121)
        self.ax2 = fig.add_subplot(122)
        self.run_plot()

    def set_fig_frame(self):
        
        xlabel = r'$\mathrm{log}\, A\,\,\,\, \mathrm{[SN\ yr^{-1}\ M_\odot^{-1}]}$'
        self.ax1.set_xlabel(xlabel, fontsize=fs + 4)
        self.ax1.set_ylabel(r'$s=s_1=s_2$', fontsize=fs + 4)
        self.ax1.set_xlim(-13.,-12.2)
        self.ax1.set_ylim(-1.5,-0.7)
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
        self.ax2.set_ylim(-2.,0.)
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

        plt.subplots_adjust(bottom=0.14, wspace=.3)

    def add_contours(self):
        
        fpath =  './../OUTPUT_FILES/RUNS/M12/likelihoods/sSNRL_s1_s2.csv'
        draw(self.ax1, self.ax2, fpath, c[0])

        fpath =  './../OUTPUT_FILES/RUNS/M12/likelihoods/vespa_s1_s2.csv'
        draw(self.ax1, self.ax2, fpath, c[1])

    def add_fit_results(self):
        s, s_err = -1.23, 0.19
        A, A_err = 3.00e-13, 0.79e-13
        logA, logA_err = np.log10(A), np.log10(np.exp(1.)) / A * A_err 
        self.ax1.axhspan(-1.12 + 0.08, -1.12 - 0.08, alpha=0.2, color='gray')

    def add_legend(self):

        self.ax1.plot([np.nan], [np.nan], color='gray', ls='-', lw=15., alpha=0.2,
                      marker='None', label=r'Maoz ${\it \, et\, al}$ (2012)')
        self.ax1.plot([np.nan], [np.nan], color=c[0], ls='-', lw=15., alpha=0.5,
                      marker='None', label=r'$sSNR_L$')
        self.ax1.plot([np.nan], [np.nan], color=c[1], ls='-', lw=15., alpha=0.5,
                      marker='None', label=r'VESPA (direct)')

        handles, labels = self.ax1.get_legend_handles_labels()                             
        self.ax1.legend(
          handles[::-1], labels[::-1], frameon=False, fontsize=fs, numpoints=1,
          ncol=1, loc=1)  

    def manage_output(self):
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_method_comp.pdf'
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.set_fig_frame()
        self.add_contours()
        self.add_fit_results()
        self.add_legend()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(show_fig=False, save_fig=True)
