#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz

from input_params import Input_Parameters as class_input
from SN_rate import Model_Rates
from generic_input_pars import Generic_Pars
from build_fsps_model import Build_Fsps

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

t_onset = ['10', '40', '100']
ls = ['--', '-.', ':']
c = ['#e41a1c', '#377eb8', '#4daf4a']
fs = 28.

yoff = [0.95, 1., 1.05]
#yoff = [1., 1., 1.]

x_fit = np.array([0.03, 14.])
def get_power_law(x_ons):
    def power_law(x,_A,_s):
        return _A + _s * (x - x_ons)
    return power_law

def leftbin1_to_xpos(_lb1):

    bin1 = np.array([_lb1, 0.42])
    bin2 = np.array([0.42, 2.4])
    bin3 = np.array([2.4, 14.])
    
    _x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
    _xErr = [
      [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
      [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]] 

    return bin1, bin2, bin3, _x, _xErr

def sfr_mass(t, t0, tf):
    #t in Gyr.
    return (np.exp(-t0) - np.exp(-t)) / (np.exp(-t0) - np.exp(-tf))

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
    s, t_onset, t_break, sfh_type, tau

    s : ~float
        DTD continuous slope.
    t_onset : ~astropy float (time unit)
        Time prior to which the supernova rate is null. (WDs not formed yet.)
    t_onset : ~astropy float (time unit)
        Inconsequential for this routine since it assumes a continuous DTD.
    sfh_type : ~str
        'exponential' or 'delayed-exponential'. Sets the input star formation
        rate for computing the 'true' supernova rate.
    tau : ~astropy float (time unit)
        Timescale to be used in the star formation rate.
        
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_DTD-fit-test_X_Y.pdf
      where X is the input slope and Y is the input t_onset.
      In the output files, 'd' stands for 'dot'.
    
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

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)

        self.TS = '1.0'
        self.D = {}
        self.make_plot()

    def set_fig_frame(self):

        plt.subplots_adjust(wspace=0.01)

        x_label = r'Delay Time [Gyr]'
        y_label = r'$\psi\,\,\,\, \mathrm{[10^{-10} \, SN\ yr^{-1}\ M_\odot^{-1}]}$'
        
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlim(6.e-3, 2.e1)
        self.ax.set_ylim(1.e-3, 1.e3)
        #self.ax.set_ylim(.01, .1)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')    
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')

    def loop_t_onset(self):

        #Loop through t_onset and make plot and write table simultaneously.
        _inputs = Generic_Pars('exponential')
        _F = Build_Fsps(_inputs).D
        for i, t_ons in enumerate(t_onset):            
            model = Model_Rates(_inputs, _F, self.TS, self.s, self.s)
            self.mean_rates(model, _F, i)

    def mean_rates(self, model, F, idx):
        """Compute delta(m) and integrated sSNR in each bin. Then divide. This
        would be closer to what M12 did, but should yoield the same results as
        the function above. Sanity check.
        """

        lb1 = float(t_onset[idx]) / 1.e3 #t_onset in Myr -> Gyr.
        bin1, bin2, bin3, x, xErr = leftbin1_to_xpos(lb1)

        y_values = []
        for _bin in [bin1, bin2, bin3]:
            #Assumes is observed at t=10Gyr.
            cond_r = ((F['age_' + self.TS] > _bin[0]) & (F['age_' + self.TS] <= _bin[1]))
            int_rate = cumtrapz(model.sSNR[cond_r], F['age_' + self.TS][cond_r])
            
            bin_age_at10_l = 10. - _bin[0]
            bin_age_at10_r = 10. - _bin[-1]
            if bin_age_at10_r < 0:
                #Do not integrate until 14 Gyr. Instead, in the oldest bin,
                #the age of that population is 10 - 10. = 0 ~ the age at which the sfr starts.
                bin_age_at10_r = F['age_' + self.TS][0]
            
            mass_l = sfr_mass(bin_age_at10_l, F['age_' + self.TS][0], F['age_' + self.TS][-1])
            mass_r = sfr_mass(bin_age_at10_r, F['age_' + self.TS][0], F['age_' + self.TS][-1])
            mass_in_bin_at10 =  mass_l - mass_r            
            print mass_in_bin_at10
            mean_rate = int_rate[-1] / mass_in_bin_at10 / (_bin[1] - _bin[0])
           
            y_values.append(mean_rate)
        y_values = np.array(y_values) * self.A 
        
        #Fit the derived values in log scale.
        power_law = get_power_law(np.log10(lb1 * 1.e9))
        popt, pcov = curve_fit(
          power_law, np.log10(x * 1.e9), np.log10(y_values))
        y_fit = 10.**power_law(np.log10(x_fit * 1.e9), popt[0], popt[1])

        label = r'$t_{\mathrm{WD}}=' + t_onset[idx] + '\, \mathrm{Myr}$ : '\
                + '$s = ' + str(format(popt[0], '.2f')) + '\pm'\
                + str(format(np.sqrt(np.diag(pcov))[0], '.2f')) + '$'
        
        self.ax.errorbar(
          x, np.array(y_values) * 1.e10 * yoff[idx], xerr=xErr, ls='None',
          marker='o', color=c[idx], markersize=10., elinewidth=2., capsize=0)
        self.ax.plot(
          x_fit, y_fit * 1.e10, ls=ls[idx], color=c[idx], lw=2., label=label)

    def add_legend(self):
        self.ax.legend(
          frameon=False, fontsize=fs - 4, numpoints=1, labelspacing=0.2,
          handletextpad=.5,loc=3)

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fname = 'Fig_M12-method_test.pdf'
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/' + fname
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
      A=1.e-11, s=-1., tau=1. * u.Gyr, show_fig=True, save_fig=True)
