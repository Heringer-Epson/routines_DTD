#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from lib import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

contour_list = [0.95, 0.68, 0.] 
fs = 24.
c = ['slateblue', 'orangered', 'limegreen']

def plot_contour(ax, x, y, z, color, label):
    _x, _y = np.unique(x), np.unique(y)
    
    x_min, x_max = min(_x), max(_x)
    y_min, y_max = min(_y), max(_y)

    X, Y = np.meshgrid(_x, _y)		
    qtty = np.reshape(z, (len(_x), len(_y)))
    levels = [stats.get_contour_levels(z, contour) for contour in contour_list]

    ax.contourf(
      X, Y, qtty, levels[0:2], colors=color, alpha=0.4, 
      extent=[x_min, x_max, y_min, y_max], zorder=5)	
    ax.contourf(
      X, Y, qtty, levels[1:3], colors=color, alpha=0.75, 
      extent=[x_min, x_max, y_min, y_max], zorder=5)	     
    ax.plot(
      x[np.argmax(z)], y[np.argmax(z)], ls='None', marker='+', color=color,
      markersize=30.)
    ax.plot([np.nan], [np.nan], color=color, ls='-', lw=15., marker='None',
      label=label)

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
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/grid_s1-s2.pdf
    
    References:
    -----------
    Maoz+ 2012: http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    """       
    def __init__(self, _inputs):

        self._inputs = _inputs
        self.s1 = None
        self.s2 = None

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.add_vespa = 'M12' in self._inputs.case.split('_')       
   
        self.run_plot()

    def read_sSNRL_data(self):
        fpath = self._inputs.subdir_fullpath + 'likelihood_s1_s2.csv'
        self.s1, self.s2, self.sSNRL_ln_L = np.loadtxt(
          fpath, delimiter=',', skiprows=7, usecols=(0,1,3), unpack=True)          

        self.s1 = self.s1[::-1]
        self.s2 = self.s2[::-1]   
        self.s1[abs(self.s1 + 1.) < 1.e-5] = 1.e-5
        self.s2[abs(self.s2 + 1.) < 1.e-5] = 1.e-5
        self.sSNRL_ln_L = stats.clean_array(self.sSNRL_ln_L[::-1])

    def set_fig_frame(self):
        x_label = r'$s_1$'
        y_label = r'$s_2$'
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xlim(min(self.s1), max(self.s1))
        self.ax.set_ylim(min(self.s2), max(self.s2))
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
        plot_contour(
          self.ax, self.s1, self.s2, self.sSNRL_ln_L, c[0], r'$sSNR_L$ method')
        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, ncol=1,
          loc=2, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)            

    def manage_output(self):
        plt.tight_layout()
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/grid_s1-s2.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.read_sSNRL_data()
        self.set_fig_frame()
        self.plot_contours()
        self.manage_output()             

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Plot_s1s2(class_input(case='SDSS_gr_Maoz'))
