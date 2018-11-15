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

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

t_onset = ['10', '40', '100']
ls = ['--', '-.', ':']
c = ['#e41a1c', '#377eb8', '#4daf4a']
fs = 24.

yoff = [0.97, 1., 1.04]

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

def write_line(t_ons, A, A_unc, s, s_unc):

    _A = str(format(10.**A * 1.e11, '.2f'))
    _A_unc = str(format(np.log(10.) * 10.**A * A_unc * 1.e11, '.2f'))
    _s = str(format(s, '.2f'))
    _s_unc = str(format(s_unc, '.2f'))

    _label = (' & $' + t_ons + '$ & $ ' + _s + '\pm' + _s_unc + '$ & $ ' +  _A
              + '\pm' + _A_unc + '$\\\\\n')

    return _label

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

        self.fig = plt.figure(figsize=(14,6))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122, sharex=self.ax1)

        self.D = {}
        self.make_plot()

    def set_fig_frame(self):

        plt.subplots_adjust(wspace=0.01)

        x_label = r'Delay Time [Gyr]'
        y_label = r'$\mathrm{SN\ yr^{-1}\ (10^{10}\ M_\odot)^{-1}}$'
        
        self.ax1.set_xlabel(x_label, fontsize=fs)
        self.ax1.set_ylabel(y_label, fontsize=fs)
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.set_xlim(6.e-3, 2.e1)
        self.ax1.set_ylim(6.e-4, 8.e-2)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax1.tick_params(
          'both', length=4, width=1., which='minor', direction='in')    
        self.ax1.xaxis.set_ticks_position('both')
        self.ax1.yaxis.set_ticks_position('both')

        self.ax2.set_xlabel(x_label, fontsize=fs)
        self.ax2.set_yscale('log')
        self.ax2.set_ylim(6.e-4, 8.e-2)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax2.tick_params(
          'both', length=4, width=1., which='minor', direction='in')    
        self.ax2.xaxis.set_ticks_position('both')
        self.ax2.yaxis.set_ticks_position('both')
        self.ax2.set_yticklabels([])

        x, y = 0.8, .9
        self.ax1.text(x, y, r'$\mathbf{a:\, \langle r \rangle}$',
          ha='left', va='center', transform=self.ax1.transAxes, fontsize=fs)
        self.ax2.text(x, y, r'$\mathbf{b:\, r(\, \bar{t} \,)}$',
          ha='left', va='center', transform=self.ax2.transAxes, fontsize=fs)

    def loop_t_onset(self):

        #Loop through t_onset and make plot and write table simultaneously.
        for i, t_ons in enumerate(t_onset):
            custom_pars=('M12', 'S18', ['SNIa', 'zSNIa'], '0.2')
            inputs = class_input(case='custom', custom_pars=custom_pars)
            model = Model_Rates(inputs, self.s, self.s, self.tau)
            
            self.D['rm_' + str(i)] = self.mean_rates(model, i)
                      
            self.D['rt_' + str(i)] = self.rate_at_mean(model, i)

        directory = './../OUTPUT_FILES/TABLES/'
        with open(directory + 'tb_M12_test.txt', 'w') as out:

            #Header part.
            out.write('\\begin{deluxetable}{cccc}\n')
            out.write('\\tablecaption{M12 method test. \\label{tb:fitted_pars}}\n')
            out.write('\\tablecolumns{4}\n')
            out.write('\\tablehead{\\colhead{Rate type} & \\colhead{$t_{\\rm{WD}}$} '
                      + '& \\colhead{slope ($s$)} & \\colhead{scale factor ($A$)}\\\\\n')
            out.write('\\colhead{} & \\colhead{[Myr]} & \\colhead{} & '
                      + '\\colhead{$\\rm{[10^{-11}\, SN\, M_\\odot ^{-1}\, yr^{-1}]}$}}\n')
            out.write('\\startdata\n')
            
            out.write('\\multirow{3}{*}{$\\langle r \\rangle$} ')
            for i, t_ons in enumerate(t_onset):
                out.write(self.D['rm_' + str(i)])
            out.write('\\hline\n')
            out.write('\\multirow{3}{*}{$r(\\bar{t})$} ')
            for i, t_ons in enumerate(t_onset):
                out.write(self.D['rt_' + str(i)])
            out.write('\\hline\n')
           
            #Wrap up.
            out.write('\\enddata\n')
            out.write('\\end{deluxetable}')

    def mean_rates(self, model, idx):
        """This routine will compute the true mean rates in each bin and then
        fit a DTD to these values using M12's method. This represents a best
        case scenario for M12's analysis, where the y values in their Fig. 1
        are correct. Note that the respective ages that corresponds to the
        SN rate mean values in each bin are *assumed* to be the mean age of
        each bin.
        
        Importantly, in M12, the fitted rates are the actual DTD component,
        which was already approximately deconvolved from the SFH. That's was
        we use the SN per unit mass below (i.e sSNRm), rather than the
        expected rate sSNR.
        """

        lb1 = float(t_onset[idx]) / 1.e3 #t_onset in Myr -> Gyr.
        bin1, bin2, bin3, x, xErr = leftbin1_to_xpos(lb1)

        y_values = []
        for _bin in [bin1, bin2, bin3]:
            age_l, age_u = _bin[0] * 1.e9 * u.yr, _bin[1] * 1.e9 * u.yr
            cond = ((model.age > age_l) & (model.age <= age_u))
            int_rate = cumtrapz(model.sSNRm[cond], model.age.value[cond])
            mean_rate = int_rate[-1] / (age_u.value - age_l.value)
            y_values.append(mean_rate)
        y_values = np.array(y_values) * self.A 
        
        #Fit the derived values in log scale.
        power_law = get_power_law(np.log10(lb1 * 1.e9))
        popt, pcov = curve_fit(
          power_law, np.log10(x * 1.e9), np.log10(y_values))
        y_fit = 10.**power_law(np.log10(x_fit * 1.e9), popt[0], popt[1])

        label = r'$t_{\mathrm{WD}}=' + t_onset[idx] + '\, \mathrm{Myr}$'

        self.ax1.errorbar(
          x, np.array(y_values) * 1.e10 * yoff[idx], xerr=xErr, ls='None',
          marker='o', color=c[idx], markersize=10.)
        self.ax1.plot(
          x_fit, y_fit * 1.e10, ls=ls[idx], color=c[idx], lw=2., label=label)
          
        return write_line(
          t_onset[idx], popt[0], np.sqrt(np.diag(pcov))[0], popt[1],
          np.sqrt(np.diag(pcov))[1])
        
    def rate_at_mean(self, model, idx):
        """Calculate the SN rates at the mean age of each bin. This is not
        what was done by M12, but is, arguably, what is represented in their
        Fig. 1, in the sense that each y-value actually represents the rates
        at the assigned ages.
        """
        
        lb1 = float(t_onset[idx]) / 1.e3 #t_onset in Myr -> Gyr.
        bin1, bin2, bin3, x, xErr = leftbin1_to_xpos(lb1)
        
        y_values = []
        for _bin in [bin1, bin2, bin3]:
            age_l, age_u = _bin[0] * 1.e9 * u.yr, _bin[1] * 1.e9 * u.yr
            cond = ((model.age > age_l) & (model.age <= age_u))
            bin_mean_age = (age_l + age_u) / 2.
            rate_interp = interp1d(model.age,np.log10(model.sSNRm))
            rate_at_mean_age = 10.**rate_interp(bin_mean_age)
            y_values.append(rate_at_mean_age)
        y_values = np.array(y_values) * self.A

        #Fit the derived values in log scale.
        power_law = get_power_law(np.log10(lb1 * 1.e9))
        popt, pcov = curve_fit(
          power_law, np.log10(x * 1.e9), np.log10(y_values))
        y_fit = 10.**power_law(np.log10(x_fit * 1.e9), popt[0], popt[1])

        self.ax2.errorbar(
          x, y_values * 1.e10 * yoff[idx], xerr=xErr, ls='None', marker='s',
           color=c[idx], markersize=10.)
        self.ax2.plot(
          x_fit, y_fit * 1.e10, ls=ls[idx], color=c[idx], lw=2.)

        return write_line(
          t_onset[idx], popt[0], np.sqrt(np.diag(pcov))[0], popt[1],
          np.sqrt(np.diag(pcov))[1])

    def add_legend(self):
        self.ax1.legend(
          frameon=False, fontsize=fs, numpoints=1, labelspacing=0.2,
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
      A=1.55e-11, s=-1.23, tau=1. * u.Gyr, show_fig=False, save_fig=True)
