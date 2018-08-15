#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from astropy import units as u

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 24.

def binned_DTD_rate(A, s, t0):
    psi1 = A / (s + 1.) * (0.42e9**(s + 1.) - t0**(s + 1.)) / (0.42e9 - t0)
    psi2 = A / (s + 1.) * (2.4e9**(s + 1.) - 0.42e9**(s + 1.)) / (2.4e9 - 0.42e9)
    psi3 = A / (s + 1.) * (14.e9**(s + 1.) - 2.4e9**(s + 1.)) / (14.e9 - 2.4e9)    
    return psi1, psi2, psi3

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

def get_bin_qtty(bin1, bin2, bin3):
    bins = [bin1, bin2, bin3]
    bin_means = np.array([np.mean(_bin) for _bin in bins])
    bin_range_min = [np.mean(_bin) - _bin[0] for _bin in bins]  
    bin_range_max = [_bin[1] - np.mean(_bin) for _bin in bins]  
    return bin_means, [bin_range_min,bin_range_max]

class Plot_Vespa(object):
    """
    Description:
    ------------
    If VESPA rates are avaliable (provided by Maoz), this routine makes a
    figure similar to Fig. 1 in Maoz+ 2012. Axes are delay time vs SN rate.
    Original rates in Fig. 1 of Maoz+ 2012 are plotted alongside the rates
    calculated for the sample used here.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/vespa_fitted_DTD.pdf
    
    References:
    -----------
    Maoz+ 2012: http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    """    
    def __init__(self, _inputs):

        self._inputs = _inputs
        
        self.t0 = self._inputs.t_onset.to(u.yr).value
        
        self.n_bins = 20
        self.best_fit = None

        self.bin1 = np.array([self.t0*1.e-9, 0.42])
        self.bin2 = np.array([0.42, 2.4])
        self.bin3 = np.array([2.4, 14.])

        if 'M12' in self._inputs.case.split('_'):
            self.fig = plt.figure(figsize=(10,10))
            self.ax = self.fig.add_subplot(111)
            self.run_plot()
        else:
            print ('Nothing to do in analyse vespa. This routine requires pre'\
                    '-calculated vespa masses.')
        
    def set_fig_frame(self):
        
        f1 = self._inputs.filter_1
        f2 = self._inputs.filter_2

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
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    

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
        #self.df = self.df[(Dcolor  >= self._inputs.Dcolor_min)
        #                  & (Dcolor <= 2. * RS_std)]

        self.host_cond = self.df['is_host'].values
        self.vistime = visibility_time(self.df['z'].values)
        self.eff_corr = np.vectorize(detection_eff)(self.df['z'].values)
        self.eff_vistime = np.multiply(self.vistime,self.eff_corr)
        
        self.mass1 = (self.df['vespa1'].values + self.df['vespa2'].values) * .55
        self.mass2 = self.df['vespa3'].values * .55
        self.mass3 = self.df['vespa4'].values * .55

    def compute_L(self, A, s):
        psi1, psi2, psi3 = binned_DTD_rate(A,s,self.t0)
        
        rate = self.mass1 * psi1 + self.mass2 * psi2 + self.mass3 * psi3
        lmbd = np.multiply(rate,self.eff_vistime)
        ln_L = -np.sum(lmbd) + np.sum(np.log(lmbd[self.host_cond]))
        return ln_L
    
    def iterate_DTDs(self):
        A_smp = np.logspace(-5., -2., self.n_bins)
        s_smp = np.linspace(-4., 0., self.n_bins)
        
        stacked_grid = np.vstack(np.meshgrid(A_smp,s_smp)).reshape(2,-1).T
        
        out_l = []
        for psis in stacked_grid:
            #self.ln_likelihood.append(self.compute_L(psis))
            inp1, inp2 = psis[0], psis[1]
            out_l.append(np.vectorize(self.compute_L)(inp1, inp2))
        out_l = np.asarray(out_l)        
        
        self.best_fit = stacked_grid[out_l.argmax()]

    def plot_quantities(self):

        x, xErr =  get_bin_qtty(self.bin1, self.bin2, self.bin3)

        A = self.best_fit[0]
        s = self.best_fit[1]
        psi1, psi2, psi3 = binned_DTD_rate(A,s,self.t0)
        y = np.array([psi1, psi2, psi3])
    
        x_fit = np.array([0.03e9, 14.e9])
        y_fit = np.array([A * (0.03e9)**s, A * (14.e9)**s])
        self.ax.plot(x_fit * 1.e-9, y_fit * 1.e10, ls='--', color='b', lw=2.,
                     label=r'Best fit: $s=' + str(format(s, '.2f')) + '$')

        self.ax.errorbar(x, y * 1.e10, xerr=xErr, ls='None',
                         marker='o', ecolor='b', lw=2., fmt='none')

        print '\n\nIn units of 10^-14 Sne / yr / Msun'
        print 'Got: psi1, psi2, psi3 = ', format(psi1 * 1.e14, '.2f'), ',',\
              format(psi2 * 1.e14, '.2f'), ',', format(psi3 * 1.e14, '.2f')
        print 'Expected: 140, 25.1, 1.83'
        print 'Rel. diff: ', format((psi1 * 1.e14 - 140.) / 140., '.2f'), ',',\
              format((psi2 * 1.e14 - 25.1) / 25.1, '.2f'), ',',\
              format((psi3 * 1.e14 - 1.83) / 1.83, '.2f')

    def plot_Maoz(self):
        _bin1 = np.array([0.04, 0.42])
        x, xErr =  get_bin_qtty(_bin1, self.bin2, self.bin3)
        y = np.array([140.e-14, 25.1e-14, 1.83e-14])
        yErr = np.array([30.e-14, 6.3e-14, 0.42e-14])
        
        self.ax.errorbar(
          x, y * 1.e10, xerr=xErr, yerr=yErr * 1.e10,
          ls='None', marker='o', fillstyle='none', color='r',
          markersize=10., label=r'M12: $s=1.17 \pm 0.07$')

    def make_legend(self):
        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, ncol=1,
          loc=1, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)         
    
    def manage_output(self):
        plt.tight_layout()        
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/vespa_fitted_DTD.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.iterate_DTDs()
        self.plot_quantities()
        self.plot_Maoz()
        self.make_legend()
        self.manage_output()             
