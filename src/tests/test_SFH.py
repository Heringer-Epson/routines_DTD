#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

tau_list = ['1.0', '2.0', '5.0', '10.0']
c = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
fs = 24.

def retrieve_fsps_SFH(_sfh, _tau):
    fpath = './../../INPUT_FILES/fsps_FILES/' + _sfh + '_tau-' + _tau + '.dat'
    df = pd.read_csv(fpath, header=0, low_memory=False)
    return df['# log_age'].values, df['instantaneous_sfr'].values

def delayed_exponential(_t, _tau):
    A = 1. / (((-_tau * _t[-1] - _tau**2.) * np.exp(-_t[-1] / _tau)) -
              ((-_tau * _t[0] - _tau**2.) * np.exp(-_t[0] / _tau)))     
    return A * _t * np.exp(-_t / _tau)

def exponential(_t, _tau):
    A = -1. / (_tau * (np.exp(-_t[-1] / _tau) - np.exp(-_t[0] / _tau)))     
    return A * np.exp(-_t / _tau)    

class Test_SFH(object):
    """
    Description:
    ------------
    TBW.
    """        
    
    def __init__(self, sfh, show_fig, save_fig):
        self.sfh = sfh
        self.show_fig = show_fig
        self.save_fig = save_fig

        fig = plt.figure(figsize=(10,10))
        self.ax = fig.add_subplot(111)
                
        self.run_test()

    def set_fig_frame(self):
        x_label = r'$\mathrm{log}\, t$'
        y_label = r'SFR [$10^{-10}\, \mathrm{M_\odot}\, \mathrm{yr^{-1}}$]'
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xlabel(x_label, fontsize=fs)

        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=16)
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')  

        self.ax.set_xlim(5., 10.2)
        self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.xaxis.set_major_locator(MultipleLocator(.5))

        if self.sfh == 'delayed-exponential':
            self.ax.set_ylim(0., 4.)
            self.ax.yaxis.set_minor_locator(MultipleLocator(.25))
            self.ax.yaxis.set_major_locator(MultipleLocator(1.))
        elif self.sfh == 'exponential':
            self.ax.set_ylim(0., 10.)
            self.ax.yaxis.set_minor_locator(MultipleLocator(.5))
            self.ax.yaxis.set_major_locator(MultipleLocator(2.))
        
    def loop_taus(self):

        #For legend purposes.
        self.ax.plot([np.nan], [np.nan], color='k', lw=3., ls='-', label=r'FSPS')
        self.ax.plot([np.nan], [np.nan], color='k', lw=2., ls=':', label=r'analytical')

        for i, tau in enumerate(tau_list):
            x, y = retrieve_fsps_SFH(self.sfh, tau)
            self.ax.plot(
              x, 1.e10 * y, ls='-', lw=3., color=c[i], marker='None',
              label=r'$\tau = ' + str(int(float(tau))) + '\, \mathrm{Gyr}$')
            
            if self.sfh == 'delayed-exponential':
                self.ax.plot(
                  x, 1.e10 * delayed_exponential(10.**x, float(tau) * 1.e9),
                  ls=':', lw=2., color=c[i+1], marker='None')
            elif self.sfh == 'exponential':
                self.ax.plot(
                  x, 1.e10 * exponential(10.**x, float(tau) * 1.e9),
                  ls=':', lw=2., color=c[i+1], marker='None')

    def make_legend(self):
        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, labelspacing=0.3,
          handletextpad=1.,loc=2)  

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../../OUTPUT_FILES/TEST_FIGURES/Fig_' + self.sfh + '_test.pdf'
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 
        
    def run_test(self):
        self.set_fig_frame()
        self.loop_taus()
        self.make_legend()
        self.manage_output()
        
if __name__ == '__main__':
    #Test_SFH(sfh='delayed-exponential', show_fig=True, save_fig=True)
    Test_SFH(sfh='exponential', show_fig=True, save_fig=True)
