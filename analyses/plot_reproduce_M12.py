#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from scipy.optimize import curve_fit
import stats


sys.path.append(os.path.join(os.environ['PATH_ssnarl'], 'src'))
from SN_rate import Model_Rates

from scipy.odr import *

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fs = 20.

bin1 = np.array([0.04, 0.42])
#bin1 = np.array([0.0, 0.42])
bin2 = np.array([0.42, 2.4])
bin3 = np.array([2.4, 14.])
x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
#x = 10.**np.array([np.mean(np.log10(bin1)), np.mean(np.log10(bin2)), np.mean(np.log10(bin3))])
xErr = [
  [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
  [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]] 
x_fit = np.array([0.03, 14.])
x_ons = 0.04e9
def power_law(x,_A,_s):
    return _A + _s * (x - 9.)

def power_law2(p,x):
    _A, _s = p
    return _A + _s * (x - 9.)

def write_pars(ax, A, A_unc, s, s_unc, pre, loc_x, loc_y):

    _A = str(format(A, '.2f'))
    _A_unc = str(format(A_unc, '.2f'))
    _s = str(format(s, '.2f'))
    _s_unc = str(format(s_unc, '.2f'))

    ax.text(loc_x, loc_y, r'' + pre + ' $s1=' + _s + '\pm' + _s_unc + '$ | $A=('
            + _A + '\pm' + _A_unc + ')\, 10^{-13}'
            + '[\mathrm{SN\, M_\odot ^{-1}\, yr^{-1}]}$', fontsize=fs - 4.,
            transform=ax.transAxes) 

def write_rate(ax, r, r_unc, pre, loc_x, loc_y):
    _r = str(format(r, '.2f'))
    _r_unc = str(format(r_unc, '.2f'))    
    ax.text(loc_x, loc_y, r'$' + pre + '(' + _r + '\pm' + _r_unc + ')\, 10^{-14}'
            + '\,[\mathrm{SN\, M_\odot ^{-1}\, yr^{-1}]}$', fontsize=fs - 4.,
            transform=ax.transAxes)

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
        
        self.make_plot()

    def set_fig_frame(self):
        x_label = r'Delay Time [Gyr]'
        y_label = r'$\mathrm{SN\ yr^{-1}\ (10^{10}\ M_\odot)^{-1}}$'
        
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlim(1.e-2, 2.e1)
        self.ax.set_ylim(1.e-4, 4.e-2)
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
        n_cells = 20
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

        self.ax.errorbar(
          x, self.most_likely_rates * 1.e10, xerr=xErr,
          yerr = self.rates_unc * 1.e10, ls='None', marker='o', color='r',
          markersize=14., capsize=0., label='Most likely rates')

        write_rate(
          self.ax, self.most_likely_rates[0] * 1.e14, self.rates_unc[0] * 1.e14,
          'r_1=', 0.02, 0.25)
        write_rate(
          self.ax, self.most_likely_rates[1] * 1.e14, self.rates_unc[1] * 1.e14,
          'r_2=', 0.02, 0.2)
        write_rate(
          self.ax, self.most_likely_rates[2] * 1.e14, self.rates_unc[2] * 1.e14,
          'r_2=', 0.02, 0.15)

    def fit_DTD_to_rates_no_unc(self):
        #Does not take into account the derived rate uncertainties.
        popt, pcov = curve_fit(
          power_law, np.log10(x * 1.e9), np.log10(self.most_likely_rates))
        A, s = popt[0], popt[1]
        print A, s
        y_fit = 10.**power_law(np.log10(x_fit * 1.e9), A, s)
        A_unc, s_unc = np.abs(np.sqrt(np.diag(pcov)))
        A = 10.**A
        A_unc = A * A_unc / np.log10(np.exp(1.))

        self.ax.plot(
          x_fit, y_fit * 1.e10, ls=':', color='r', alpha=0.6, lw=2.,
          label=r'Fit not inc. unc')
        write_pars(
          self.ax, 1.e13 * A, 1.e13 * A_unc, s, s_unc, 'No unc:', 0.02, 0.1)            

    def fit_DTD_to_rates_inc_unc(self):
        #Fit taking into account the rate's uncertainty.
        rates_unc_in_log = np.divide(
          self.rates_unc,self.most_likely_rates) * np.log10(np.e)
        popt, pcov = curve_fit(
          power_law, np.log10(x * 1.e9), np.log10(self.most_likely_rates),
          sigma=1. / rates_unc_in_log**2.)        
        
        A, s = popt[0], popt[1]
        print A, s
        y_fit = 10.**power_law(np.log10(x_fit * 1.e9), A, s)
        A_unc, s_unc = np.abs(np.sqrt(np.diag(pcov)))
        A = 10.**A
        A_unc = A * A_unc / np.log10(np.exp(1.))
        
        self.ax.plot(
          x_fit, y_fit * 1.e10, ls='-', color='r', lw=2., label=r'Fit including unc')
        write_pars(
          self.ax, 1.e13 * A, 1.e13 * A_unc, s, s_unc, 'Inc unc:', 0.02, 0.05)   

    def fit_DTD_to_rates_inc_xyunc(self):
        """Test where both the uncertainty in x and in y are taken into
        account when producing the best fit.
        See: https://stackoverflow.com/questions/22670057/linear-fitting-in-python-with-uncertainty-in-both-x-and-y-coordinates
        """
        
        #Fit taking into account the rate's uncertainty.
        rates_unc_in_log = np.divide(
          self.rates_unc,self.most_likely_rates) * np.log10(np.e)
        ages_unc_in_log = np.divide(
          xErr[0],x) * np.log10(np.e)

        quad_model = Model(power_law2)

        # Create a RealData object using our initiated data from above.
        data = RealData(
          np.log10(x * 1.e9), np.log10(self.most_likely_rates),
          sx=ages_unc_in_log, sy=rates_unc_in_log)

        odr = ODR(data, quad_model, beta0=[-12.5, -1.25])

        # Run the regression.
        out = odr.run()

        # Use the in-built pprint method to give us results.
        out.pprint()
        
    def add_legend(self):
        self.ax.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1, loc=1)  
              
    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_reproduce_M12.pdf'
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.read_vespa_masses()
        self.get_likelihoods()
        self.fit_DTD_to_rates_no_unc()             
        self.fit_DTD_to_rates_inc_unc()
        #self.fit_DTD_to_rates_inc_xyunc()
        self.add_legend()
        self.manage_output()

if __name__ == '__main__':
    dirpath = './../OUTPUT_FILES/RUNS/M12/'
    Plot_M12(dirpath=dirpath, show_fig=False, save_fig=True)
