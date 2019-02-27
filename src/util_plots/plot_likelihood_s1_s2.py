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

class Plot_s1s2(object):
    """
    Description:
    ------------
    Given SN rate /propto t**s1/s2, this code creates a contour plot in the
    s1 vs s2 parameter space. The contour plot derived from the sSNR method is 
    always plotted, whereas the results derived from an analysis using VESPA is
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
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/Fig_grid_s1-s2.pdf
    
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

        #Initialize output file with the best fit.
        fpath = self._inputs.subdir_fullpath + 'likelihoods/Best_s1_s2.csv'
        header = 'Method,s1,s1_unc_low,s1_unc_high,s2,s2_unc_low,s2_unc_high'
        self.out = stats.Write_Outpars(fpath, header)
   
        self.run_plot()

    def set_fig_frame(self):        
        x_label = r'$s1$'
        y_label = r'$s2$'
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        #self.ax.set_xlim(-3.,0.)
        #self.ax.set_ylim(-3.,0.)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')             
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
                        
    def plot_contours(self):
        fpath = self._inputs.subdir_fullpath + 'likelihoods/sSNRL_s1_s2.csv'
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
        nx, ny = len(np.unique(s1)), len(np.unique(s2))
        X, Y, XErr, YErr = stats.plot_contour(
          self.ax, s1, s2, ln_L, c[0], nx, ny, r'$sSNR_L$')          
        self.out.add_line('sSNRL', X, Y, XErr, YErr)

        try:
            fpath = self._inputs.subdir_fullpath + 'likelihoods/vespa_s1_s2.csv'
            N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
            nx, ny = len(np.unique(s1)), len(np.unique(s2))
            X, Y, XErr, YErr = stats.plot_contour(
              self.ax, s1, s2, ln_L, c[1], nx, ny, r'$\tt{vespa}$')
            self.out.add_line('VESPA', X, Y, XErr, YErr)

            
            self.ax.legend(
              frameon=False, fontsize=fs, numpoints=1, ncol=1,
              loc=1, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)   
        except:
            pass

    def manage_output(self):
        plt.tight_layout()
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/Fig_grid_s1-s2.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.set_fig_frame()
        self.plot_contours()
        self.manage_output()             

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Plot_s1s2(class_input(case='H17'))

