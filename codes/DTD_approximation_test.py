#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz

from SN_rate_gen import Model_Rates

bin1 = np.array([0., 0.42])
bin2 = np.array([0.42, 2.4])
bin3 = np.array([2.4, 14.])

x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
xErr = [
  [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
  [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]] 
x_fit = np.array([0.03, 14.])

fs = 20.
    
def power_law(x,_A,_s):
    return _A + _s * x

class DTD_Approximations(object):
    """
    Description:
    ------------
    This code attempts to illustrate how good the assumptions made in
    Maoz+ 2012 are for fitting the DTD based on the the supernova rates per
    unit of mass (sSNRm).
    
    In the original paper, the rates are binned in three age bins. To fit these
    rates, the authors (arbitrarily) attribute an age to those bins, which
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

    
    def __init__(self, s, t_onset, t_break, sfh_type, tau,
                 show_fig, save_fig):
                
        self.s = s
        self.t_onset = t_onset
        self.t_break = t_break
        self.sfh_type = sfh_type
        self.tau = tau
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.model = None
        self.sSNRm_bin_at_tmean = []
        self.sSNRm_bin_Maoz = []
        
        self.fitpars_tmean = None
        self.fitpars_Maoz = None
        
        self.make_plot()

    def set_fig_frame(self):
        x_label = r'Delay Time [Gyr]'
        y_label = r'$\mathrm{SN\ yr^{-1}\ (10^{10}\ M_\odot)^{-1}}$'
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlim(3.e-2, 2.e1)
        self.ax.set_ylim(3.e-5, 4.e-2)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')    
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')
        
        #Make title and figure name.
        slope_str = r' $s=' + str(format(self.s, '.1f')) + '$, '
        if self.sfh_type == 'exponential':
            sfh_str = (r'$S(t) \propto e^{\frac{-t}{'
                       + str(self.tau.value / 1.e9) + ' \mathrm{Gyr}}}$, ')
        tonset_str = (r'$t_{\mathrm{ons}}=' + str(format(
                      self.t_onset.value / 1.e9, '.2f')) + '$ Gyr')
        
        plt.title(r'Inputs:'  + slope_str + sfh_str + tonset_str, fontsize=fs)

    def retrieve_model_rates(self):
        tau_suffix = str(self.tau.to(u.yr).value / 1.e9)
        synpop_dir = './../INPUT_FILES/fsps_FILES/'
        synpop_fname = self.sfh_type + '_tau-' + tau_suffix + '.dat'
        
        self.model = Model_Rates(
          self.s, self.s, self.t_onset, self.t_break, 'sdss_r', 'sdss_g',
          'Chabrier', self.sfh_type, 0.0190, synpop_dir, synpop_fname)
        
    def compute_rates_in_bins(self):
        
        for _bin in [bin1, bin2, bin3]:
            age_l, age_u = _bin[0] * 1.e9 * u.yr, _bin[1] * 1.e9 * u.yr
            cond = ((self.model.age > age_l) & (self.model.age <= age_u))
   
            #Actual rate at mean age in each bin. This represents the case
            #Where Danny's approximation is the best possible. i.e., the
            #'guess' of age in the x-axis actually corresponds to the actual
            #sSNRm at that point.
            bin_mean_age = (age_l + age_u) / 2.
            rate_interp = interp1d(self.model.age,np.log10(self.model.sSNRm))
            rate_at_mean_age = 10.**rate_interp(bin_mean_age)
            self.sSNRm_bin_at_tmean.append(rate_at_mean_age)

            #Tries to compute the sSNRm that would represent the same rates
            #that Danny uses. Then a age is guessed for fitting the DTD.
            int_rate = cumtrapz(self.model.sSNRm[cond],
                                self.model.age.value[cond])
            mean_rate = int_rate[-1] / (age_u.value - age_l.value)
            self.sSNRm_bin_Maoz.append(mean_rate)

        self.sSNRm_bin_at_tmean = np.array(self.sSNRm_bin_at_tmean)
        self.sSNRm_bin_Maoz = np.array(self.sSNRm_bin_Maoz)

    def recover_DTD_Maoz_method(self):
        
        #Fit DTD using sSNRm at t mean.
        popt, pcov = curve_fit(power_law, np.log10(x * 1.e9),
                               np.log10(self.sSNRm_bin_at_tmean))
        self.fitpars_tmean = [
          popt[0], popt[1], np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1]]

        #Fit DTD using the mean sSNRm in each bin.
        popt, pcov = curve_fit(power_law, np.log10(x * 1.e9),
                               np.log10(self.sSNRm_bin_Maoz))
        self.fitpars_Maoz = [
          popt[0], popt[1], np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1]]

    def plot_quantities(self):
        y_fit_tmean = 10.**power_law(
          np.log10(x_fit * 1.e9), self.fitpars_tmean[0], self.fitpars_tmean[1])

        y_fit_Maoz = 10.**power_law(
          np.log10(x_fit * 1.e9), self.fitpars_Maoz[0], self.fitpars_Maoz[1])
        
        
        self.ax.plot(
          self.model.age * 1.e-9 ,self.model.sSNRm * 1.e10, ls=':', color='k',
          label=r'True $r(t)$')

        #Plot Maoz method where rates are actual rates at t average.
        self.ax.errorbar(
          x, self.sSNRm_bin_at_tmean * 1.e10, xerr=xErr, ls='None',
          marker='o', color='k', markersize=10., label=r'Mock Rates $r(\bar{t})$')

        self.ax.plot(
          x_fit, y_fit_tmean * 1.e10, ls='--', color='m', label=r'Best fit of '\
          + r'$r(\bar{t})$: s=' + str(format(self.fitpars_tmean[1], '.2f'))\
          + r'$\pm$' + str(format(self.fitpars_tmean[3], '.2f')))

        #Plot Maoz method where rates derived from the mean sSNRm. This is
        #more likely to represent the rates derived in Maoz+ 2012 by
        #optimizing the likelihood.
        self.ax.errorbar(
          x, self.sSNRm_bin_Maoz * 1.e10, xerr=xErr, ls='None', marker='o',
          color='b', markersize=10., label=r'Mock Rates $\langle r \rangle$')
        
        self.ax.plot(
          x_fit, y_fit_Maoz * 1.e10, ls='--', color='b', label=r'Best fit of '\
          + r'$\langle r \rangle$: s=' + str(format(self.fitpars_Maoz[1], '.2f'))\
          + r'$\pm$' + str(format(self.fitpars_Maoz[3], '.2f')))
        
        self.ax.legend(frameon=False, fontsize=20., numpoints=1, loc=4)

    def manage_output(self):
        if self.save_fig:
            fname = ('Fig_DTD-fit-test_' + str(format(self.s, '.1f')) + '_'
                     + str(format(self.t_onset.value / 1.e9, '.2f')))
            fname = fname.replace('.', 'd')
            fpath = './../OUTPUT_FILES/FIGURES/' + fname
            plt.savefig(fpath + '.pdf', format='pdf')
        if self.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_model_rates()
        self.compute_rates_in_bins()
        self.recover_DTD_Maoz_method()
        self.plot_quantities()
        self.manage_output()             

if __name__ == '__main__':
    DTD_Approximations(-1., 0.1e9 * u.yr, 1.e9 * u.yr, 'exponential',
                       1.e9 * u.yr, show_fig=True, save_fig=True)
