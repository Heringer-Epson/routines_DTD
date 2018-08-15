#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from astropy import units as u

t0 = 0.1e9

bin1 = np.array([t0*1.e-9, 0.42])
bin2 = np.array([0.42, 2.4])
bin3 = np.array([2.4, 14.])

def binned_DTD_rate(A, s):
    psi1 = A / (s + 1.) * (0.42e9**(s + 1.) - t0**(s + 1.)) / (0.42e9 - t0)
    psi2 = A / (s + 1.) * (2.4e9**(s + 1.) - 0.42e9**(s + 1.)) / (2.4e9 - 0.42e9)
    psi3 = A / (s + 1.) * (14.e9**(s + 1.) - 2.4e9**(s + 1.)) / (14.e9 - 2.4e9)    
    return psi1, psi2, psi3

class Plot_Vespa_2(object):
    
    def __init__(self, _inputs):

        self._inputs = _inputs
        self.n_bins = 20
        self.best_fit = None

        if 'Maoz' in self._inputs.case.split('_'):
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

        self.host_cond = self.df['n_SN'].values
        self.vistime = self.visibility_time(self.df['z'].values)
        self.mass1 = (self.df['vespa1'].values + self.df['vespa2'].values) * .55
        self.mass2 = self.df['vespa3'].values * .55
        self.mass3 = self.df['vespa4'].values * .55

    def compute_L(self, A, s):
        psi1, psi2, psi3 = binned_DTD_rate(A,s)
        
        rate = self.mass1 * psi1 + self.mass2 * psi2 + self.mass3 * psi3
        lmbd = np.multiply(rate,self.vistime)
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
                
    def visibility_time(self, redshift):

        def detection_eff_func(_z):
            if _z < 0.175:
                detection_eff = 0.72
            else:
                detection_eff = -3.2 * _z + 1.28
            return detection_eff
        
        survey_duration = 269. * u.day
        survey_duration = survey_duration.to(u.year).value
        _time = np.ones(len(redshift)) * survey_duration        
        _time = np.divide(_time,(1. + redshift)) #In the galaxy rest frame.
        
        vec_func = np.vectorize(detection_eff_func)
        eff_correction = vec_func(redshift)
        _time = np.multiply(_time,eff_correction)
        return _time

    def plot_quantities(self):
        
        x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
        xErr = [
          [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
          [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]]

        A = self.best_fit[0]
        s = self.best_fit[1]
        
        psi1, psi2, psi3 = binned_DTD_rate(A,s)
 
        y = np.array([psi1, psi2, psi3])
        
        self.ax.errorbar(x, y * 1.e10, xerr=xErr, ls='None', marker='o',
                         ecolor='b', lw=2., fmt='none')
    
        x_fit = np.array([0.03e9, 14.e9])
        y_fit = np.array([A * (0.03e9)**s, A * (14.e9)**s])
        
        self.ax.plot(x_fit * 1.e-9, y_fit * 1.e10, ls='--', color='b', lw=2., label='Best fit')

        plt.title('Fit DTD', fontsize=20.)
        self.ax.text(2., 0.014, 'Slope = ' + str(format(s, '.2f')),
                     fontsize=20., color='b')
        self.ax.text(2, 0.02, r'$t_{\mathrm{onset}}\ =\ $' + str(t0 * 1.e-6)
                     + ' Myr', fontsize=20.)

        print '\n\nIn units of 10^-14 Sne / yr / Msun'
        print 'Got: psi1, psi2, psi3 = ', format(psi1 * 1.e14, '.2f'), ',',\
              format(psi2 * 1.e14, '.2f'), ',', format(psi3 * 1.e14, '.2f')
        print 'Expected: 140, 25.1, 1.83'
        print 'Rel. diff: ', format((psi1 * 1.e14 - 140.) / 140., '.2f'), ',',\
              format((psi2 * 1.e14 - 25.1) / 25.1, '.2f'), ',',\
              format((psi3 * 1.e14 - 1.83) / 1.83, '.2f')

    def plot_Maoz(self):
        x = np.array([np.mean(bin1), np.mean(bin2), np.mean(bin3)])
        xErr = [
          [np.mean(bin1) - bin1[0], np.mean(bin2) - bin2[0], np.mean(bin3) - bin3[0]],
          [bin1[1] - np.mean(bin1), bin2[1] - np.mean(bin2), bin3[1] - np.mean(bin3)]]        
        y = np.array([140.e-14, 25.1e-14, 1.83e-14])
        yErr = np.array([30.e-14, 6.3e-14, 0.42e-14])
        self.ax.errorbar(x, y * 1.e10, xerr=xErr, yerr=yErr * 1.e10, ls='None',
                         marker='o', fillstyle='none', color='r',
                         markersize=10., label='Orig data points')
    
    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/vespa_likelihood.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.iterate_DTDs()
        self.plot_quantities()
        self.plot_Maoz()
        self.manage_output()             
