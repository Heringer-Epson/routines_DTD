#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from astropy import units as u
from scipy.optimize import curve_fit

A_min, A_max = -6., 2
s_min, s_max = -2., -.5
n_bins = 200
orders = 20
contour_list = [0.95, 0.68] 

t0 = 0.1e9

bin1 = np.array([0., 0.42])
bin2 = np.array([0.42, 2.4])
bin3 = np.array([2.4, 14.])

x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
xErr = [
  [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
  [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]] 
x_fit = np.array([0.03e9, 14.e9])

def visibility_time(redshift): 
    survey_duration = 269. * u.day
    survey_duration = survey_duration.to(u.year).value
    vistime = np.ones(len(redshift)) * survey_duration        
    vistime = np.divide(vistime,(1. + redshift)) #In the galaxy rest frame.
    return vistime

def detection_eff(redshift):
    if redshift < 0.175:
        detection_eff = 0.72
    else:
        detection_eff = -3.2 * redshift + 1.28
    return detection_eff 

def power_law(x,_A,_s):
    return _A + _s * x

def binned_DTD_rate(A, s):
    """Computes the average SN rate in each time bin."""
    psi1 = A / (s + 1.) * (0.42e9**(s + 1.) - t0**(s + 1.)) / (0.42e9 - t0)
    psi2 = A / (s + 1.) * (2.4e9**(s + 1.) - 0.42e9**(s + 1.)) / (2.4e9 - 0.42e9)
    psi3 = A / (s + 1.) * (14.e9**(s + 1.) - 2.4e9**(s + 1.)) / (14.e9 - 2.4e9)    
    return psi1, psi2, psi3

class Vespa_Rates(object):
    
    def __init__(self, _inputs):

        self._inputs = _inputs
        self.vistime = None
        self.eff_corr = None
        self.eff_vistime = None
        self.opt_rates = None
        self.opt_DTD = None
        self.ratesErr = None
        self.DTDErr = None
        self.A = None
        self.s = None
        self.s_unc = None
        self.L = None
        self.D = {}

        self.A_smp = np.logspace(A_min, A_max, n_bins)
        self.s_smp = np.linspace(s_min, s_max, n_bins)

        if 'Maoz' in self._inputs.case.split('_'):
            self.fig = plt.figure(figsize=(10,10))
            self.ax = self.fig.add_subplot(111)

            self.fig_grid = plt.figure(figsize=(10,10))
            self.ax_grid = self.fig_grid.add_subplot(111)       
       
            self.run_plot()
        else:
            print ('Nothing to do in analyse vespa. This routine requires pre'\
                    '-calculated vespa masses.')
        
    def set_fig_frame(self):
        
        f1 = self._inputs.filter_1
        f2 = self._inputs.filter_2
        x_label = r'Delay Time [Gyr]'
        y_label = r'$\mathrm{SN\ yr^{-1}\ (10^{10}\ M_\odot)^{-1}}$'
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlim(3.e-2, 2.e1)
        self.ax.set_ylim(3.e-5, 4.e-2)
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    

        x_label = r'$A$'
        y_label = r'$s$'
        self.ax_grid.set_xlabel(x_label, fontsize=24.)
        self.ax_grid.set_ylabel(y_label, fontsize=24.)
        self.ax_grid.tick_params(axis='y', which='major', labelsize=24., pad=8)      
        self.ax_grid.tick_params(axis='x', which='major', labelsize=24., pad=8)
        self.ax_grid.minorticks_off()
        self.ax_grid.tick_params('both', length=8, width=1., which='major')
        self.ax_grid.tick_params('both', length=4, width=1., which='minor') 

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0)
        self.f1, self.f2 = self._inputs.filter_1, self._inputs.filter_2
        photo1 = self.df['abs_' + self.f1]
        photo2 = self.df['abs_' + self.f2]
        Dcolor = self.df['Dcolor_' + self.f2 + self.f1]

        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
        #self.df = self.df[(Dcolor  >= -10. * RS_std)
        #                  & (Dcolor <= 2. * RS_std)]

        self.host_cond = self.df['n_SN'].values
        self.vistime = visibility_time(self.df['z'].values)
        self.eff_corr = np.vectorize(detection_eff)(self.df['z'].values)
        self.eff_vistime = np.multiply(self.vistime,self.eff_corr)
        
        self.mass1 = (self.df['vespa1'].values + self.df['vespa2'].values) * .55
        self.mass2 = self.df['vespa3'].values * .55
        self.mass3 = self.df['vespa4'].values * .55

    def compute_L_from_rates(self, psi1, psi2, psi3):
        rate = self.mass1 * psi1 + self.mass2 * psi2 + self.mass3 * psi3
        lmbd = np.multiply(rate,self.eff_vistime)
        ln_L = -np.sum(lmbd) + np.sum(np.log(lmbd[self.host_cond]))
        return ln_L
    
    def determine_most_likely_rates(self):
        r_smp = np.logspace(-16., -10., n_bins)
        stacked_grid = np.vstack(np.meshgrid(r_smp,r_smp,r_smp)).reshape(3,-1).T
        
        out_l = []
        for psis in stacked_grid:
            inp1, inp2, inp3 = psis[0], psis[1], psis[2]
            out_l.append(np.vectorize(self.compute_L_from_rates)(inp1, inp2, inp3))
        out_l = np.asarray(out_l)        
        
        self.opt_rates = stacked_grid[out_l.argmax()]        

    def compute_rate_uncertainty(self):
        alpha = np.zeros(9).reshape(3,3) #Format is C[row,col] or C[i,j]
        
        #Compute rates assuming most likely model.
        psi1, psi2, psi3 = self.opt_rates[0], self.opt_rates[1], self.opt_rates[2]
        rate = self.mass1 * psi1 + self.mass2 * psi2 + self.mass3 * psi3
        lmbd = np.multiply(rate,self.eff_vistime)
        masses = [self.mass1, self.mass2, self.mass3]       
                
        for j in range(3):
            for k in range(3):
                psi_j = self.opt_rates[j]
                psi_k = self.opt_rates[k]
                
                t = self.vistime
                n = self.host_cond.astype(float)
                
                
                alpha[j,k] = np.sum((np.power(np.divide(n,lmbd) - 1.,2)
                                     * np.power(self.eff_vistime,2)
                                     #* np.power(self.vistime,2) #as in Maoz+ 2012
                                     * np.multiply(masses[j],masses[k])))
                
        C = np.linalg.inv(alpha)
        self.ratesErr = np.sqrt(np.diag(C))

    def fit_most_likely_rates(self):
        """Fits the SN rates concomitantly in the three age bins. In other
        words, here we try to reproduce the rates derived by Maoz+ 2012."""
        
        yErr_log = np.log10(math.e) * np.divide(self.ratesErr,self.opt_rates) 
        popt, pcov = curve_fit(
          power_law,np.log10(x * 1.e9), np.log10(self.opt_rates), sigma=yErr_log)
        self.s, self.A = popt[1], popt[0] 
        self.s_unc = np.sqrt(np.diag(pcov))[1]
    
    def plot_most_likely_rates(self):
        """Plots the re-derived rates and their best DTD fit."""

        psi1, psi2, psi3 = self.opt_rates[0], self.opt_rates[1], self.opt_rates[2]
        y = np.array([psi1, psi2, psi3])
        
        self.ax.errorbar(
          x, self.opt_rates * 1.e10, xerr=xErr, yerr=self.ratesErr * 1.e10, ls='None',
          marker='o', color='b', markersize=10., label='re-calc Rates')

        y_fit = 10.**self.A * x_fit**self.s
        self.ax.plot(x_fit * 1.e-9, y_fit * 1.e10, ls='-', color='b', lw=2.,
                     label='Fit re-calc: s=' + str(format(self.s, '.2f'))\
                     + r'$\pm$' + str(format(self.s_unc, '.2f')))

    def compute_L_from_DTDs(self, _A, _s):
        psi1, psi2, psi3 = binned_DTD_rate(_A,_s)
        rate = self.mass1 * psi1 + self.mass2 * psi2 + self.mass3 * psi3
        lmbd = np.multiply(rate,self.eff_vistime)
        ln_L = -np.sum(lmbd) + np.sum(np.log(lmbd[self.host_cond]))
        return ln_L
    
    def determine_most_likely_DTD(self):
        stacked_grid = np.vstack(np.meshgrid(self.A_smp,self.s_smp)).reshape(2,-1).T
        
        _L = []
        for psis in stacked_grid:
            _L.append(np.vectorize(self.compute_L_from_DTDs)(psis[0], psis[1]))
        self.L = np.asarray(_L)        
        self.opt_DTD = stacked_grid[self.L.argmax()]
        
        #There can be a large range of likelihoods (even in log space: i.e.
        #>100 orders of mag). Set a maximum range of 20 orders of magnitude,
        #so that the values can be taken the exponent to be worked in linear
        #scale. This does not affect likelihood contours.
        self.L[self.L < max(self.L) - orders] = max(self.L) - orders      
        self.L = self.L - min(self.L) 
        self.L = np.exp(self.L)
        self.L = self.L / sum(self.L)
                
    def plot_most_likely_DTD(self):
        """Plots the rates and DTD obtained by maximizing the DTD likelihood."""

        _A = self.opt_DTD[0]
        _s = self.opt_DTD[1]
        y_rates = np.array(binned_DTD_rate(_A,_s))
        
        self.ax.errorbar(
          x, y_rates * 1.e10, xerr=xErr, ls='None',
          marker='s', color='g', markersize=10., label='Rates from most likely DTD')

        y_fit = 10.**_A * x_fit**_s
        y_fit = np.array([_A * (x_fit[0])**_s, _A * (x_fit[1])**_s])
        self.ax.plot(x_fit * 1.e-9, y_fit * 1.e10, ls='-', color='g', lw=2.,
                     label='Most likely DTD: s=' + str(format(_s, '.2f')))

    def make_grid_plot(self):

        cmap = plt.get_cmap('Greys')
        
        qtty_max = max(self.L)
        qtty_min = min(self.L)
        
        qtty = np.reshape(self.L, (n_bins, n_bins))
        
        _im = self.ax_grid.imshow(
          qtty, interpolation='nearest', aspect='auto',
          extent=[A_min, A_max, s_min, s_max],
          origin='lower', cmap=cmap,
          norm=colors.LogNorm(vmin=qtty_min,  vmax=qtty_max))                  

        #Add contours
        #Note that the cumulative histogram is normalized since self.L is norm.
        _L_hist = sorted(self.L, reverse=True)
        _L_hist_cum = np.cumsum(_L_hist)

        for contour in contour_list:
            _L_hist_diff = [abs(value - contour) for value in _L_hist_cum]
            diff_at_contour, idx_at_contour = min((val,idx) for (idx,val)
                                                  in enumerate(_L_hist_diff))
            self.D['prob_' + str(contour)] = _L_hist[idx_at_contour]
            
            #print list(self.L[self.L > 1.e-5]) 
            #print self.D['prob_' + str(contour)], diff_at_contour
            if diff_at_contour > 0.1:
                UserWarning(str(contour * 100.) + '% contour not constrained.')	

    def plot_contour(self):
        X, Y = np.meshgrid(self.A_smp, self.s_smp)		
        levels = [self.D['prob_' + str(contour)]
                  for contour in contour_list]	
        
        #print list(self.L), levels
        qtty = np.reshape(self.L, (n_bins, n_bins))

        self.ax_grid.contour(
          qtty, levels, origin='lower', 
          extent=[A_min, A_max, s_min, s_max], colors=['r', 'r'],
          linestyles=['--', '-'], linewidths=(2., 3.), zorder=5)	

    def plot_Maoz(self):
        """Plots and fits the data points provided in Maoz+ 2012."""
       
        y = np.array([140.e-14, 25.1e-14, 1.83e-14])
        yErr = np.array([30.e-14, 6.3e-14, 0.42e-14])
        self.ax.errorbar(x, y * 1.e10, xerr=xErr, yerr=yErr * 1.e10, ls='None',
                         marker='o', fillstyle='none', color='r',
                         markersize=10., label='Orig data points')
        
        yErr_log = np.log10(math.e) * np.divide(yErr,y) #1-sigma unc in log scale.        
        weigh = 1. / yErr_log
        
        #curve_fit
        popt, pcov = curve_fit(power_law,np.log10(x * 1.e9), np.log10(y),
                               sigma=yErr_log)
        pErr = np.sqrt(np.diag(pcov))

        y_fit = 10.**popt[0] * x_fit**popt[1]
        self.ax.plot(x_fit * 1.e-9, y_fit * 1.e10, ls='--', color='r', lw=2.,
                     label='Maoz (curve_fit): s=' + str(format(popt[1], '.2f'))\
                     + r'$\pm$' + str(format(pErr[1], '.2f')))
        
        #Polyfit. - Sanity check for when taking yErr into account.
        '''
        coeffs, var = np.polyfit(np.log10(x * 1.e9), np.log10(y), 1, w=weigh,
                            cov=True)
        pErr = np.sqrt(np.diag(np.abs(var)))
        s_m, A_m = coeffs[0], coeffs[1] 
        y_fit = 10.**A_m * x_fit**s_m
        self.ax.plot(x_fit * 1.e-9, y_fit * 1.e10, ls='--', color='g', lw=2.,
                     label='Maoz (polyfit): s=' + str(format(s_m, '.2f'))\
                     + r'$\pm$' + str(format(pErr[0], '.2f')))
        '''

    def make_legend(self):
        self.ax.legend(frameon=False, fontsize=20., numpoints=1, loc=3) 
        plt.tight_layout()

    def comparison_verbose(self):
        psi1, psi2, psi3 = self.opt_rates[0], self.opt_rates[1], self.opt_rates[2]
        print '\n\nIn units of 10^-14 Sne / yr / Msun'
        print 'Re-calc: psi1, psi2, psi3 = ', format(psi1 * 1.e14, '.2f'), ',',\
              format(psi2 * 1.e14, '.2f'), ',', format(psi3 * 1.e14, '.2f')
        print 'Expected: 140, 25.1, 1.83'
        print 'Rel. diff: ', format((psi1 * 1.e14 - 140.) / 140., '.2f'), ',',\
              format((psi2 * 1.e14 - 25.1) / 25.1, '.2f'), ',',\
              format((psi3 * 1.e14 - 1.83) / 1.83, '.2f')
    
    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/vespa_rates.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        
        #Reproduce calculation of most likely rates.
        '''
        self.determine_most_likely_rates()
        self.compute_rate_uncertainty()
        self.fit_most_likely_rates()
        self.plot_most_likely_rates()
        '''
        #Implement the calculation of most likely DTD.
        self.determine_most_likely_DTD()
        self.plot_most_likely_DTD()
        self.make_grid_plot()
        self.plot_contour()
        
        self.plot_Maoz()
        self.make_legend()
        #self.comparison_verbose()
        self.manage_output()             
