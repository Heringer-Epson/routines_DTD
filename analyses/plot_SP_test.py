#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from astropy import units as u
from lib import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
c = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999','c']
fs = 26.

def pars2fpath(imf,sfh,Z,isoc_lib,spec_lib,t_o,t_c):
    return (
      './../OUTPUT_FILES/RUNS/sys_' + imf + '_' + sfh + '_' + Z + '_' + isoc_lib\
      + '_' + spec_lib + '_' + t_o + '_' + t_c + '/likelihoods/sSNRL_s1_s2.csv') 

class Make_Fig(object):
    """
    Description:
    ------------
    Shows how the most likely DTD parameters change depending on model
    parameters, such as the IMF, metallicity, etc. Figure contains two panels,
    the right and left ones showing the A-s and s1-s2 space, respectively.
  
    Parameters:
    -----------
    add_contours : ~bool
        If True, also plot the 68% and 95% confidence contours for all tests.
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_SP_uncertainties.pdf
    """           
    def __init__(self, add_contours, show_fig, save_fig):
        self.add_contours = add_contours
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        fig = plt.figure(figsize=(16,8))
        self.ax1 = fig.add_subplot(121)
        self.ax2 = fig.add_subplot(122)
        self.A_max, self.s_max = None, None
        
        self.run_plot()

    def set_fig_frame(self):
        plt.subplots_adjust(wspace=0.3)

        self.ax1.set_ylabel(r'$s$', fontsize=fs)
        self.ax1.set_xlabel(r'$\mathrm{log}\, A$', fontsize=fs)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=16)
        self.ax1.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax1.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax1.xaxis.set_ticks_position('both')
        self.ax1.yaxis.set_ticks_position('both')  
        self.ax1.set_xlim(-12.8, -11.8)
        self.ax1.set_ylim(-2., -1.)
        self.ax1.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.xaxis.set_major_locator(MultipleLocator(.5))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.yaxis.set_major_locator(MultipleLocator(.5))

        self.ax2.set_ylabel(r'$s_2$', fontsize=fs)
        self.ax2.set_xlabel(r'$s_1$', fontsize=fs)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=16)
        self.ax2.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax2.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax2.xaxis.set_ticks_position('both')
        self.ax2.yaxis.set_ticks_position('both')  
        self.ax2.set_xlim(-3., 0.)
        self.ax2.set_ylim(-3., 0.)
        self.ax2.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax2.xaxis.set_major_locator(MultipleLocator(.5))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax2.yaxis.set_major_locator(MultipleLocator(.5))

    def plot_arrow(self, fpath, color, label):
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
                
        #A-s space
        x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
        x_max, y_max = np.log10(x[np.argmax(z)]), y[np.argmax(z)]
        if (abs(x_max - self.A_max) > 1.e-6) or (abs(y_max - self.s_max) > 1.e-6):
            self.ax1.arrow(
              self.A_max, self.s_max, x_max - self.A_max, y_max - self.s_max, 
              color=color, width=0.004, length_includes_head=True)
            if self.add_contours:
                stats.plot_contour(self.ax1, np.log10(x), y, z, color=color)
            self.ax1.plot([np.nan],[np.nan],color=color,lw=10.,ls='-',label=label)

        #s1-s2 space
        x_max, y_max = s1[np.argmax(ln_L)], s2[np.argmax(ln_L)]
        if (abs(x_max - self.s1_max) > 1.e-6) or (abs(y_max - self.s2_max) > 1.e-6):
            self.ax2.arrow(
              self.s1_max, self.s2_max, x_max - self.s1_max, y_max - self.s2_max,
              color=color, width=0.004, length_includes_head=True)
        
        if self.add_contours:
            stats.plot_contour(self.ax2, s1, s2, ln_L, c=color)

    def plot_default_contour(self):
        fpath = pars2fpath('Chabrier','exponential','0.0190','PADOVA','BASEL','100','1')        
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
        
        #A-s space
        x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
        stats.plot_contour(self.ax1, np.log10(x), y, z, c=c[0])
        self.A_max, self.s_max = np.log10(x[np.argmax(z)]), y[np.argmax(z)]
        self.ax1.plot([np.nan],[np.nan],color=c[0],lw=10.,ls='-',label=r'Default')
        #s1-s2 space
        stats.plot_contour(self.ax2, s1, s2, ln_L, c=c[0])
        self.s1_max, self.s2_max = s1[np.argmax(ln_L)], s2[np.argmax(ln_L)]


    def add_arrows(self):
        self.plot_arrow(pars2fpath('Kroupa','exponential','0.0190','PADOVA','BASEL',
                        '100','1'), c[1], r'IMF: Kroupa')
        self.plot_arrow(pars2fpath('Salpeter','exponential','0.0190','PADOVA','BASEL',
                        '100','1'), c[2], r'IMF: Salpeter')
        self.plot_arrow(pars2fpath('Chabrier','exponential','0.0190','PADOVA','BASEL',
                        '40','1'), c[3], r'$t_{\mathrm{onset}}=40\, \mathrm{Myr}$')
        self.plot_arrow(pars2fpath('Chabrier','exponential','0.0150','PADOVA','BASEL',
                        '100','1'), c[4], r'$Z=0.015$')
        self.plot_arrow(pars2fpath('Chabrier','exponential','0.0300','PADOVA','BASEL',
                        '100','1'), c[5], r'$Z=0.03$')
        self.plot_arrow(pars2fpath('Chabrier','exponential','0.0190','PADOVA','MILES',
                        '100','1'), c[6], r'spec_lib: MILES')
        self.plot_arrow(pars2fpath('Chabrier','exponential','0.0190','MIST','BASEL',
                        '100','1'), c[7], r'isoc_lib: MIST')
        self.plot_arrow(pars2fpath('Chabrier','delayed-exponential','0.0190','PADOVA','BASEL',
                        '100','1'), c[8], r'SFH: Delayed-exponential')

        self.ax1.legend(
          frameon=False,fontsize=20.,numpoints=1,labelspacing=0.3,handletextpad=1.,loc=3) 

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_SP_uncertainties.pdf'
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.set_fig_frame()
        self.plot_default_contour()
        self.add_arrows()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(add_contours=False, show_fig=True, save_fig=False)
