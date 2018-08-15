#!/usr/bin/env python

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from DTD_gen import make_DTD

t_onset = 1.e8 * u.yr
t_break = 1.e9 * u.yr
tau = 1.e9 * u.yr

class Plot_Test(object):
    
    def __init__(self, show_fig=True, save_fig=False):
        """Makes a figure where a set of DTDs is plotted as a function of age.
        """

        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax1 = plt.subplot(211)  
        self.ax2 = plt.subplot(212, sharex= self.ax1)  
        self.fs = 20.   
        
        self.make_plot()

    def compute_analytical_sfr(self, tau, upper_lim):
        _tau = tau.to(u.yr).value
        _upper_lim = upper_lim.to(u.yr).value
        norm = 1. / (_tau * (1. - np.exp(-_upper_lim / _tau)))
        def sfr_func(age):
            return norm * np.exp(-age / _tau)
        return sfr_func
        
    def set_fig_frame(self):
        
        x_label = r'log age [yr]'
        y_label1 = r'$\rm{log\ SFR}\ \rm{[M_\odot\ yr^{-1}]}$'
        y_label2 = r'$\rm{log\ DTD}\ \rm{[M_\odot ^{-1}\ yr^{-1}]}$'
        
        self.ax1.set_ylabel(y_label1,fontsize=self.fs)
        self.ax1.set_xlim(7.,10.5)      
        self.ax1.set_ylim(-10.,-8.)
        self.ax1.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax1.minorticks_on()
        self.ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax1.xaxis.set_major_locator(MultipleLocator(1.))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax1.yaxis.set_major_locator(MultipleLocator(0.5))  
        self.ax1.tick_params('both', length=8, width=1., which='major')
        self.ax1.tick_params('both', length=4, width=1., which='minor')
        self.ax1.tick_params(labelbottom='off') 

        self.ax2.set_xlabel(x_label,fontsize=self.fs)
        self.ax2.set_ylabel(y_label2,fontsize=self.fs)
        self.ax2.set_xlim(7.,10.5)      
        self.ax2.set_ylim(-14.5,-11.5)
        self.ax2.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax2.minorticks_on()
        self.ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax2.xaxis.set_major_locator(MultipleLocator(1.))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax2.yaxis.set_major_locator(MultipleLocator(0.5))  
        self.ax2.tick_params('both', length=8, width=1., which='major')
        self.ax2.tick_params('both', length=4, width=1., which='minor')

    def plot_models(self):

        t = 1.e9 * u.yr
        tprime = np.logspace(7., 10.2, 1000) * u.yr
        cond = ((tprime >= t_onset) & (tprime <= t))
        tprime = tprime[cond]
        
        t_minus_tprime = t - tprime
        
        self.sfr_func = self.compute_analytical_sfr(tau, 14.6e9 * u.yr)
        self.DTD_func = make_DTD(-1., -1., t_onset, t_break)
                
        self.ax2.plot(
          np.log10(t_minus_tprime.value), np.log10(self.DTD_func(t_minus_tprime)), marker='None',
          ls='-', lw=3., color='b')
        self.ax2.plot(
          np.log10(tprime.value), np.log10(self.DTD_func(tprime)), marker='None',
          ls='-', lw=3., color='r')

        self.ax1.plot(
          np.log10(t_minus_tprime.value), np.log10(self.sfr_func(t_minus_tprime.value)), marker='None',
          ls='-', lw=3., color='b')
        self.ax1.plot(
          np.log10(tprime.value), np.log10(self.sfr_func(tprime.value)), marker='None',
          ls='-', lw=3., color='r')
                
        
        #self.ax.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1,
        #               labelspacing=-0.2, loc=2)            

              
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_conv-test' + '.' + extension,
                        format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def make_plot(self):
        self.set_fig_frame()
        self.plot_models()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        self.save_figure()
        self.show_figure()  

if __name__ == '__main__':
    Plot_Test(show_fig=True, save_fig=False)
