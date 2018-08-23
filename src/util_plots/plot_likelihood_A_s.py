#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
import lib
from lib import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fs = 24.
c = ['slateblue', 'orangered', 'limegreen']

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
        self.A = None
        self.s = None

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.add_vespa = 'M12' in self._inputs.case.split('_')       
   
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
        self.ax.minorticks_off()
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')             

    def plot_Maoz_result(self):
        self.ax.axhspan(-1.07 + 0.07, -1.07 - 0.07, alpha=0.5, color='gray')
        self.ax.plot([np.nan], [np.nan], color='gray', ls='-', lw=15., alpha=0.5,
                     marker='None', label=r'Maoz ${\it \, et\, al}$ (2012)')
                
    def plot_contours(self):
        
        fpath = self._inputs.subdir_fullpath + 'likelihoods/sSNRL_A_s.csv'
        x, y, z = lib.stats.read_lnL(fpath, colx='A', coly='s1', colz='ln_L')
        lib.stats.plot_contour(self.ax, np.log10(x), y, z, c[0], r'$sSNR_L$ method')
        
        if self.add_vespa:

            fpath = self._inputs.subdir_fullpath + 'likelihoods/vespa_A_s.csv'
            x, y, z = lib.stats.read_lnL(fpath, colx='A', coly='s1', colz='ln_L')
            lib.stats.plot_contour(
              self.ax, np.log10(x), y, z, c[1], r'Vespa method: all')

            fpath = self._inputs.subdir_fullpath + 'likelihoods/vespatrim_A_s.csv'
            x, y, z = lib.stats.read_lnL(fpath, colx='A', coly='s1', colz='ln_L')
            lib.stats.plot_contour(
              self.ax, np.log10(x), y, z, c[2], r'Vespa method: $\Delta$')
        
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

