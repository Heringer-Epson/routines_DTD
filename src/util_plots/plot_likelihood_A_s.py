#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from lib import stats
from lib import survey_efficiency

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

contour_list = [0.95, 0.68, 0.] 
fs = 24.
c = ['slateblue', 'orangered', 'limegreen']

def plot_contour(ax, x, y, z, color, label):
    _x, _y = np.log10(np.unique(x)), np.unique(y)
    
    x_min, x_max = min(_x), np.log10(max(_x))
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
      np.log10(x[np.argmax(z)]), y[np.argmax(z)], ls='None', marker='+', color=color,
      markersize=30.)
    ax.plot([np.nan], [np.nan], color=color, ls='-', lw=15., marker='None',
      label=label)

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
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/grid_A-s.pdf
    
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

    def read_sSNRL_data(self):
        fpath = self._inputs.subdir_fullpath + 'likelihood_A_s.csv'
        self.A, self.s, self.sSNRL_ln_L = np.loadtxt(
          fpath, delimiter=',', skiprows=7, usecols=(0,1,3), unpack=True)          

        self.A = self.A[::-1]
        self.s = self.s[::-1]   
        self.s[abs(self.s + 1.) < 1.e-5] = 1.e-5
        self.sSNRL_ln_L = stats.clean_array(self.sSNRL_ln_L[::-1])

    def set_fig_frame(self):
        x_label = r'$\mathrm{log}\, A$'
        y_label = r'$s$'
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xlim(min(np.log10(self.A)), max(np.log10(self.A)))
        self.ax.set_ylim(min(self.s), max(self.s))
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')
        
    def retrieve_vespa_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0)
        f1, f2 = self._inputs.filter_1, self._inputs.filter_2
        photo1 = self.df['abs_' + f1]
        photo2 = self.df['abs_' + f2]
        Dcolor = self.df['Dcolor_' + f2 + f1]

        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
        Dcolor_cond = ((Dcolor >= self._inputs.Dcolor_min)
                       & (Dcolor <= 2. * RS_std))
        self.df_Dcolor = self.df[Dcolor_cond]

    def determine_most_likely_DTD(self, df):
        host_cond = df['is_host'].values
        vistime = survey_efficiency.visibility_time(df['z'].values)
        eff_corr = np.vectorize(survey_efficiency.detection_eff)(df['z'].values)
        eff_vistime = np.multiply(vistime,eff_corr)
        mass1 = (df['vespa1'].values + df['vespa2'].values) * .55
        mass2 = df['vespa3'].values * .55
        mass3 = df['vespa4'].values * .55
        t0 = self._inputs.t_onset.to(u.Gyr).value       
        
        stacked_grid = np.vstack(np.meshgrid(
          np.unique(self.A),np.unique(self.s))).reshape(2,-1).T
        _L = []
        for psis in stacked_grid:
            _L.append(stats.compute_L_from_DTDs(psis[0], psis[1], t0, mass1,
              mass2, mass3, eff_vistime, host_cond))
        _L = np.asarray(_L)        
        _L = stats.clean_array(_L)                
        opt_DTD = stacked_grid[_L.argmax()]  
        return _L  

    def plot_Maoz_result(self):
        self.ax.axhspan(-1.07 + 0.07, -1.07 - 0.07, alpha=0.5, color='gray')
        self.ax.plot([np.nan], [np.nan], color='gray', ls='-', lw=15., alpha=0.5,
                     marker='None', label=r'Maoz ${\it \, et\, al}$ (2012)')
                
    def plot_contours(self):
        plot_contour(
          self.ax, self.A, self.s, self.sSNRL_ln_L, c[0], r'$sSNR_L$ method')
        if self.add_vespa:
            #Not subselected by Dcolor.
            L = self.determine_most_likely_DTD(self.df)
            plot_contour(self.ax, self.A, self.s, L, c[1], r'Vespa method: all')

            #Subselected by Dcolor.
            L = self.determine_most_likely_DTD(self.df_Dcolor)
            plot_contour(self.ax, self.A, self.s, L, c[2], r'Vespa method: $\Delta$')
        
        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, ncol=1,
          loc=2, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)            

    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/grid_A-s.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.read_sSNRL_data()
        self.set_fig_frame()
        if self.add_vespa:
            self.retrieve_vespa_data()
        self.plot_Maoz_result()
        self.plot_contours()
        self.manage_output()             

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Plot_As(class_input(case='SDSS_gr_Maoz'))

