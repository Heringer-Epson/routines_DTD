#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from scipy.optimize import curve_fit

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

t_onset = ['40', '60', '100']
ls = ['--', '-.', ':']
c = ['#e41a1c', '#377eb8', '#4daf4a']
fs = 32.
yoff = [0.95, 1., 1.05] #Offset for plotting bins. For clarity purposes only.

x_fit = np.array([0.03, 14.])
def power_law(x,_A,_s):
    return _A + _s * x

def leftbin1_to_xpos(_lb1):
    """Given a choice of onset time, produce bin mean age for fitting purposes.
    """
    bin1 = np.array([_lb1, 0.42])
    bin2 = np.array([0.42, 2.4])
    bin3 = np.array([2.4, 14.])
    _x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
    _xErr = [
      [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
      [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]] 
    return bin1, bin2, bin3, _x, _xErr

def mean_DTD(s, A, t0, ti, tf):
    #Below, I have tested that if s=1. or s=-1.01 converge to the same result.
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
    
    Parameters:
    -----------
    s, t_onset, t_break, sfh_type, tau

    A : ~float
        DTD scale factor. Suggested value is 1.e-12.
        Implied physical unit is SN / M_sun / yr.
    s : ~float
        DTD continuous slope. Suggested value is -1.
    tau : ~astropy float (time unit)
        Timescale for an assumed exponential star formation rate.
        Suggested values is 1. * u.Gyr
        
    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_M12-method_test.pdf
    
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
        y_label = (r'$\langle \psi \rangle \,\,\,\, \mathrm{[10^{-10}'
                   + ' \, yr^{-1}\ M_\odot^{-1}]}$')
        
        self.ax1.set_xlabel(x_label, fontsize=fs)
        self.ax1.set_ylabel(y_label, fontsize=fs)
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.set_xlim(6.e-3, 2.e1)
        self.ax1.set_ylim(1.e-4, 1.)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params(
          'both', length=12, width=2., which='major', direction='in')
        self.ax1.tick_params(
          'both', length=6, width=2., which='minor', direction='in')    
        self.ax1.xaxis.set_ticks_position('both')
        self.ax1.yaxis.set_ticks_position('both')

        x_label = (r'$\mathrm{log}\, A\,\,\,\, \mathrm{[yr^{-1}'
                   + '\ M_\odot^{-1}]}$')
        y_label = r'$s$'
        self.ax2.set_xlabel(x_label, fontsize=fs)
        self.ax2.set_ylabel(y_label, fontsize=fs)
        self.ax2.set_xlim(-12.05, -11.8)
        self.ax2.set_ylim(-1.1, -0.9)
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
        
    def loop_t_onset(self):

        #Loop through t_onset and make plot and write table simultaneously.
        for i, t_ons in enumerate(t_onset):            
            lb1 = float(t_onset[i]) / 1.e3 #t_onset in Myr -> Gyr.
            t0 = float(t_ons) / 1.e3
            bin1, bin2, bin3, x, xErr = leftbin1_to_xpos(lb1)

            psi_values = np.zeros(3)
            for j, _bin in enumerate([bin1, bin2, bin3]):
                psi_values[j] = mean_DTD(self.s, self.A, t0, _bin[0], _bin[1])
            
            #Fit the derived values in log scale.
            popt, pcov = curve_fit(
              power_law, np.log10(x), np.log10(psi_values))
            y_fit = 10.**power_law(np.log10(x_fit), popt[0], popt[1])

            label = r'$t_{\mathrm{WD}}=' + t_onset[i] + '\, \mathrm{Myr}$'
            self.ax1.errorbar(
              x, np.array(psi_values) * 1.e10 * yoff[i], xerr=xErr, ls='None',
              marker='o', color=c[i], markersize=10., elinewidth=2., capsize=0)
            self.ax1.plot(
              x_fit, y_fit * 1.e10, ls=ls[i], color=c[i], lw=2., label=label)
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
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_M12-method_test.pdf'
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.loop_t_onset()
        self.add_legend()
        self.manage_output()             

if __name__ == '__main__':
    DTD_Approximations(
      A=1.e-12, s=-1., tau=1. * u.Gyr, show_fig=False, save_fig=True)
