#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from scipy.optimize import curve_fit

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

Nsb_list = [1,2,3]

ls = ['--', '-.', ':']
c = ['#66c2a5','#fc8d62','#8da0cb']
fs = 32.

yoff = [0.95, 1., 1.05]
t0 = 0.04

x_fit = np.array([0.03, 14.])
def power_law(x,_A,_s):
    return _A + _s * x

def get_bins(Nsb):
    bin1_edges = np.linspace(t0, 0.42, Nsb + 1)
    bin2_edges = np.linspace(0.42, 2.4, Nsb + 1)[1:]
    bin3_edges = np.linspace(2.4, 14., Nsb + 1)[1:]
    bins = np.concatenate((bin1_edges, bin2_edges, bin3_edges), axis=0)
    bin_means = np.array([(a + b) / 2. for (a,b) in zip(bins,bins[1:])])
    bin_Err = [list(bin_means - bins[0:-1]), list(bins[1:] - bin_means)]
    return bins, bin_means, bin_Err

def mean_DTD(s, A, t0, ti, tf):
    ##Below, I have tested that if s=1. or s=-1.01 converge to the same result.
    if abs(s + 1.) > 1.e-3:
        return A * (tf**(s + 1.) - ti**(s + 1.)) / (s + 1.) / (tf - ti)
    else:
        return A * np.log(tf / ti) / (tf - ti)

class DTD_Approximations(object):
    """
    Description:
    ------------
    This code attempts to illustrate how good the assumptions made in
    Maoz+ 2012 are for fitting the DTD based on the the supernova rates per
    unit of mass (sSNRm).
    
    The following is not true - check! In the original paper, the rates are
    binned in three age bins. To fit these
    rates, Maoz et al (arbitrarily) attribute an age to those bins, which
    correspond to the simple age average in each bin. Note that for the
    youngest bin, despite the original plot showing otherwise, the bin starts
    at t=0. 
    
    Parameters:
    -----------
    A : ~float
        DTD scale factor. Suggested is 1.e -12. Implied units are SN / Msun/ yr.
    s : ~float
        DTD continuous slope. Sugegsted is -1.
    tau : ~astropy float (time unit)
        Timescale to be used in the star formation rate.
        
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_binning_test.pdf

    References:
    --------
    Maoz+ 2012: http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    """
    
    def __init__(self, A, s, tau, show_fig, save_fig):
        self.A = A
        self.s = s
        self.tau = tau
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.fig, (self.ax1, self.ax2) = plt.subplots(
          2,1, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(10,12))
        self.make_plot()

    def set_fig_frame(self):

        x_label = r'Delay Time [Gyr]'
        y_label = r'$\langle \psi \rangle \,\,\,\, \mathrm{[10^{-10} \, SN\ yr^{-1}\ M_\odot^{-1}]}$'
        
        self.ax1.set_xlabel(x_label, fontsize=fs)
        self.ax1.set_ylabel(y_label, fontsize=fs)
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.set_xlim(1.e-2, 2.e1)
        self.ax1.set_ylim(5.e-4, 5.e-1)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params(
          'both', length=12, width=2., which='major', direction='in')
        self.ax1.tick_params(
          'both', length=6, width=2., which='minor', direction='in')    
        self.ax1.xaxis.set_ticks_position('both')
        self.ax1.yaxis.set_ticks_position('both')

        x_label = r'$\mathrm{log}\, A\, [\rm{SN\ yr^{-1}\ M_\odot^{-1}}]$'
        y_label = r'$s$'
        self.ax2.set_xlabel(x_label, fontsize=fs)
        self.ax2.set_ylabel(y_label, fontsize=fs)
        self.ax2.set_xlim(-12.05, -11.8)
        self.ax2.set_ylim(-1.1, -.9)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params(
          'both', length=12, width=2., which='major', direction='in')
        self.ax2.tick_params(
          'both', length=6, width=2., which='minor', direction='in')    
        self.ax2.xaxis.set_ticks_position('both')
        self.ax2.yaxis.set_ticks_position('both')
        self.ax2.xaxis.set_minor_locator(MultipleLocator(.05))
        self.ax2.xaxis.set_major_locator(MultipleLocator(.1))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(.05))
        self.ax2.yaxis.set_major_locator(MultipleLocator(.1))  

        plt.axvline(x=np.log10(self.A), ls=':', lw=3., c='k')
        plt.axhline(y=self.s, ls=':', lw=3., c='k')

    def do_fitting(self):

        for i, Nsb in enumerate(Nsb_list):

            bins, b_Mean, b_Err = get_bins(Nsb)
            Nb = len(b_Mean)
            psi_values = np.zeros(Nb)
            for j in range(Nb):
                psi_values[j] = mean_DTD(self.s, self.A, t0, bins[j], bins[j + 1])
                
            popt, pcov = curve_fit(
              power_law, np.log10(b_Mean), np.log10(psi_values))
            y_fit = 10.**power_law(np.log10(x_fit), popt[0], popt[1])

            self.ax1.errorbar(
              b_Mean, np.array(psi_values) * 1.e10, xerr=b_Err, ls='None',
              marker='o', color=c[i], markersize=10., elinewidth=2., capsize=0)
            
            self.ax1.plot(x_fit, y_fit * 1.e10, ls=ls[i], color=c[i], lw=2.,
              label=r'# bins = ' + str(Nsb * 3))
            
            self.ax2.errorbar(
              popt[0], popt[1], xerr=np.sqrt(np.diag(pcov))[0],
              yerr=np.sqrt(np.diag(pcov))[1], ls='None', marker='None', color=c[i],
              markersize=10., elinewidth=2., capsize=0)  

    def add_legend(self):
        self.ax1.legend(
          frameon=False, fontsize=fs, numpoints=1, labelspacing=0.2,
          handletextpad=.5,loc=3)

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_binning_test.pdf'
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.do_fitting()
        self.add_legend()
        self.manage_output()             

if __name__ == '__main__':
    DTD_Approximations(
      A=1.e-12, s=-1., tau=1. * u.Gyr, show_fig=False, save_fig=True)
