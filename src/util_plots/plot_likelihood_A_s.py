#!/usr/bin/env python
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 24.
c = ['#1b9e77','#d95f02','#7570b3']

class Plot_As(object):
    """
    Description:
    ------------
    Given SN rate = A*t**s, this code creates a contour plot in the A vs s
    parameter space. The contour plot derived from the sSNR method is always
    plotted, whereas the results derived from an analysis using VESPA is
    also plotted if the data is available for that sample. Note that the VESPA
    contour is different than what is done by Maoz+ 2012, in the sense that
    the A and s parameters are directly determined from a bayesian analysis,
    rather than fit to rates retrieved by a bayesian analysis. Vespa data was
    provided by Maox in priv. comm.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/Fig_grid_A-s.pdf
    
    References:
    -----------
    Maoz+ 2012: http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    """       
    def __init__(self, _inputs):

        self._inputs = _inputs
        
        self.s = None

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.add_vespa = 'M12' in self._inputs.case.split('_')       

        #Initialize output file with the best fit.
        fpath = self._inputs.subdir_fullpath + 'likelihoods/Best_A_s.csv'
        header = 'Method,A,A_unc_low,A_unc_high,s,s_unc_low,s_unc_high'
        self.out = stats.Write_Outpars(fpath, header)

        self.run_plot()

    def set_fig_frame(self):        
        x_label = r'$\mathrm{log}\, A$'
        y_label = r'$s$'
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        #self.ax.set_xlim(min(np.log10(self.A)), max(np.log10(self.A)))
        #self.ax.set_ylim(min(self.s), max(self.s))
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.xaxis.set_major_locator(MultipleLocator(.5)) 
        self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.yaxis.set_major_locator(MultipleLocator(.5)) 
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')             

    def plot_Maoz_result(self):
        self.ax.axhspan(-1.07 + 0.07, -1.07 - 0.07, alpha=0.5, color='gray')
        self.ax.plot([np.nan], [np.nan], color='gray', ls='-', lw=15., alpha=0.5,
                     marker='None', label=r'Maoz ${\it \, et\, al}$ (2012)')
                
    def plot_contours(self):
        
        fpath = self._inputs.subdir_fullpath + 'likelihoods/sSNRL_s1_s2.csv'
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
        x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
        nx, ny = len(np.unique(x)), len(np.unique(y))
        X, Y, XErr, YErr = stats.plot_contour(
          self.ax, np.log10(x), y, z, c[0], nx, ny, r'$sSNR_L$')
        self.out.add_line('sSNRL', X, Y, XErr, YErr)
                
        if self.add_vespa:
            fpath = self._inputs.subdir_fullpath + 'likelihoods/vespa_s1_s2.csv'
            N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
            x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
            nx, ny = len(np.unique(x)), len(np.unique(y))
            X, Y, XErr, YErr = stats.plot_contour(
              self.ax, np.log10(x), y, z, c[1], nx, ny, r'$\tt{vespa}$')
            self.out.add_line('VESPA', X, Y, XErr, YErr)
       
        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, ncol=1,
          loc=2, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)            

    def manage_output(self):
        plt.tight_layout()
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/Fig_grid_A-s.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.set_fig_frame()
        self.plot_Maoz_result()
        self.plot_contours()
        self.manage_output()             

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Plot_As(class_input(case='SDSS_gr_Maoz'))

