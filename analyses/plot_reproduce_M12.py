#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from scipy.optimize import curve_fit

from input_params import Input_Parameters as class_input
from SN_rate import Model_Rates
from lib import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fs = 20.

bin1 = np.array([0.04, 0.42])
bin2 = np.array([0.42, 2.4])
bin3 = np.array([2.4, 14.])
x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
xErr = [
  [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
  [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]] 


x_fit = np.array([0.03, 14.])
def power_law(x,_A,_s):
    return _A + _s * x

class Plot_M12(object):
    """
    Description:
    ------------
    This code attempts to reproduce Fig. 1 in Maoz+ 2012. The purpose is to
    demonstrate that we understand their method and are able to re-derive
    the same result when the same input is used.
    
    Parameters:
    -----------
    dirpath : ~str
        Absolute path to the upper directory of the simulation 'run' to be
        used. This runs are produced by the ./../src/master.py code and stored
        under ./../OUTPUT_FILES/RUNS/.
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_reproduce_M12.pdf
    
    References:
    --------
    Maoz+ 2012: http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    """

    
    def __init__(self, dirpath, show_fig, save_fig):

        self.dirpath = dirpath
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.masses, self.redshift, self.hosts = None, None, None
        self.most_likely_rates, self.rates_unc = None, None
        self.A, self.s, self.A_unc, self.s_unc = None, None, None, None
        self.y_fit = None
        
        self.make_plot()

    def set_fig_frame(self):
        x_label = r'Delay Time [Gyr]'
        y_label = r'$\mathrm{SN\ yr^{-1}\ (10^{10}\ M_\odot)^{-1}}$'
        
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlim(1.e-2, 2.e1)
        self.ax.set_ylim(3.e-5, 4.e-2)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')    
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')

    def read_vespa_masses(self):
        df = pd.read_csv(
          self.dirpath + 'data_merged.csv', header=0, low_memory=False)

        mass1 = (df['vespa1'].values + df['vespa2'].values) * .55
        mass2 = df['vespa3'].values * .55
        mass3 = df['vespa4'].values * .55
        
        self.masses = np.transpose(zip(mass1,mass2,mass3))
        self.redshift = df['z'].values
        self.hosts = df['is_host'].values        

    def get_likelihoods(self):
        n_cells = 100
        r1 = np.logspace(-12, -11, n_cells)
        r2 = np.logspace(-13, -12, n_cells)
        r3 = np.logspace(-14, -13, n_cells)
        rate_space = np.asarray(
          [(_r1,_r2,_r3) for _r1 in r1 for _r2 in r2 for _r3 in r3])
        
        ln_L = []
        for _r in rate_space:
            ln_L.append(stats.compute_rates_using_L(
              _r, self.masses, self.redshift, self.hosts, True))
        self.most_likely_rates = rate_space[np.asarray(ln_L).argmax()]       

        #Compute uncertainty.
        alpha = np.zeros((3,3))
        for j in range(3):
            for k in range(3):
                alpha[j,k] = stats.compute_curvature(
                  j, k, self.most_likely_rates, self.masses, self.redshift,
                  self.hosts, True)

        self.rates_unc = np.sqrt(np.diag(np.linalg.inv(alpha)))

    def fit_DTD_to_rates(self):
        #Fit the derived values in log scale.
        
        #Fit not taking into account the rate's uncertainty.
        #popt, pcov = curve_fit(
        #  power_law, np.log10(x * 1.e9), np.log10(self.most_likely_rates))
        
        #Fit taking into account the rate's uncertainty.
        rates_unc_in_log = np.divide(
          self.rates_unc,self.most_likely_rates) * np.log10(np.e)
        popt, pcov = curve_fit(
          power_law, np.log10(x * 1.e9), np.log10(self.most_likely_rates),
          sigma=1. / rates_unc_in_log**2.)        
        
        self.A, self.s = popt[0], popt[1]
        self.y_fit = 10.**power_law(np.log10(x_fit * 1.e9), self.A, self.s)
        self.A_unc, self.s_unc = np.abs(np.sqrt(np.diag(pcov)))

    def plot_quantities(self):
        self.ax.errorbar(
          x, self.most_likely_rates * 1.e10, xerr=xErr,
          yerr = self.rates_unc * 1.e10, ls='None', marker='o', color='r',
          markersize=14., capsize=0., label='Most likely rates')
        self.ax.plot(x_fit, self.y_fit * 1.e10, ls='-', color='r' )        

        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, labelspacing=0.3,
          handletextpad=1., loc=3)

    def write_outfile(self):
        if self.save_fig:
            with open(
              './../OUTPUT_FILES/ANALYSES_FILES/reproduce_M12.dat', 'w') as out:
                out.write('Created on ' + str(datetime.datetime.now()) + '\n\n')
                out.write('log A = ' + str(self.A) + ' += ' + str(self.A_unc) + '\n')
                out.write('s = ' + str(self.s) + ' += ' + str(self.s_unc) + '\n')
                out.write('\nRates in units of 10^-14 yr^-1 Msun^-1\n')
                out.write(
                  'In bin 1 (0.04 - 0.42 Gyr): ' + str(self.most_likely_rates[0]
                  * 1.e14) + ' += ' + str(self.rates_unc[0] * 1.e14) + '\n')
                out.write(
                  'In bin 2 (0.42 - 2.4 Gyr): ' + str(self.most_likely_rates[1]
                  * 1.e14) + ' += ' + str(self.rates_unc[1] * 1.e14) + '\n')
                out.write(
                  'In bin 3 (2.4 - 14 Gyr): ' + str(self.most_likely_rates[2]
                  * 1.e14) + ' += ' + str(self.rates_unc[2] * 1.e14) + '\n')
                  
    def manage_output(self):
        print self.s, self.s_unc
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_reproduce_M12.pdf'
            plt.savefig(fpath + '.pdf', format='pdf')
        if self.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.read_vespa_masses()
        self.get_likelihoods()             
        self.fit_DTD_to_rates()
        self.write_outfile()
        self.plot_quantities()
        self.manage_output()

if __name__ == '__main__':
    dirpath = './../OUTPUT_FILES/RUNS/M12/'
    Plot_M12(dirpath=dirpath, show_fig=True, save_fig=True)
