#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
c = ['#1b9e77','#377eb8','#e41a1c','#984ea3','#ff7f00','#a65628','#f781bf','#999999','c','m', 'r']
fs = 30.

def pars2fpath(imf,sfh,Z,fhbh,dust,isoc_lib,spec_lib,t_o,t_c,Q,f2,f1,f0):
    return (
      './../OUTPUT_FILES/RUNS/sys/' + imf + '_' + sfh + '_' + Z + '_'
      + fhbh + '_' + dust + '_' + isoc_lib + '_' + spec_lib + '_' + t_o
      + '_' + t_c + '_' + Q + '_' + f2 + f1 + f0 + '/likelihoods/sSNRL_s1_s2.csv') 

def plot_arrow(ax, fpath, A_max, s_max, color, label, add_contours):
    N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
            
    x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
    x_max, y_max = np.log10(x[np.argmax(z)]), y[np.argmax(z)]
    
    if abs(A_max - x_max) > 0.05 or abs(s_max - y_max) > 0.03:
        #Only plot cases for which 'A' or 's' change significantly.
        print A_max - x_max, s_max - y_max
        ax.plot([A_max, x_max], [s_max, y_max], ls='-', lw=4.,
                      marker='None', color=color)
        ax.plot([np.nan],[np.nan],color=color,lw=4.,ls='-',label=label)        
        if add_contours:
            stats.plot_contour(ax, np.log10(x), y, z, c=color)
            ax.plot([np.nan],[np.nan],color=color,lw=10.,ls='-',label=label)

class Make_Fig(object):
    """
    Description:
    ------------
    Makes Fig. 6 in the DTD paper, which shows how the most likely DTD
    parameters change depending on model parameters, such as the IMF,
    metallicity, etc. Only the (A,s) parameter space is ploted, because it
    behaves we;; (i.e. most likely parameters are not at the edge of the 
    allowed parameter space) and the changes are easier to understand.
  
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
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_systematics.pdf
    """           
    def __init__(self, add_contours, show_fig, save_fig):
        self.add_contours = add_contours
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        fig = plt.figure(figsize=(10,10))
        self.ax = fig.add_subplot(111)
        self.A_max, self.s_max = None, None
        
        self.run_plot()

    def set_fig_frame(self):

        xlabel = r'$\mathrm{log}\, A\,\,\,\, \mathrm{[yr^{-1}\ M_\odot^{-1}]}$'
        self.ax.set_ylabel(r'$s$', fontsize=fs)
        self.ax.set_xlabel(xlabel, fontsize=fs)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=16)
        self.ax.tick_params(
          'both', length=12, width=2., which='major', direction='in')
        self.ax.tick_params(
          'both', length=6, width=2., which='minor', direction='in')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')  
        self.ax.set_xlim(-12.7,-11.9)
        self.ax.set_ylim(-1.7,-0.9)
        self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.xaxis.set_major_locator(MultipleLocator(.2))
        self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.yaxis.set_major_locator(MultipleLocator(.2))

    def plot_default_contour(self):
        #Either path below works, as expected.
        #fpath = pars2fpath(
        #  'Kroupa','exponential','0.0190','0.0','PADOVA','BASEL','100','1','1.6')        
        fpath = './../OUTPUT_FILES/RUNS/default/standard/likelihoods/sSNRL_s1_s2.csv'
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
        
        x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
        nx, ny = len(np.unique(x)), len(np.unique(y))        
        stats.plot_contour(self.ax, np.log10(x), y, z, '#1b9e77', nx, ny)
        self.A_max, self.s_max = np.log10(x[np.argmax(z)]), y[np.argmax(z)]


    def add_arrows(self):
       
        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','0','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'i', 'r'), self.A_max, self.s_max, c[1], 
          r'$\Delta(g-i),L_{r}$', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0150','0.0','0','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[3], 
          r'$Z=0.015$', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0300','0.0','0','PADOVA',
          'BASEL', '100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[5], 
          r'$Z=0.03$', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','0','PADOVA',
          'BASEL', '40','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[2],
          r'$t_{\mathrm{onset}}=40\, \mathrm{Myr}$', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Salpeter','exponential','0.0190','0.0','0','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[4],
           r'IMF: Salpeter', self.add_contours)

        #Actually gives the same A and s, with small diffs in the uncertainties!
        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','0','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'i', 'i'), self.A_max, self.s_max, c[6], 
          r'$\Delta(g-i),L_{i}$', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','0','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[6], 
          r'Dust=0', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','1','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[7], 
          r'Dust=1', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','2','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[8], 
          r'Dust=2', self.add_contours)
        
        plot_arrow(
          self.ax, pars2fpath('Chabrier','exponential','0.0190','0.0','0','PADOVA',
          'BASEL', '100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[1], 
          r'IMF: Chabrier', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','0','PADOVA',
          'MILES', '100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[6], 
          r'spec_lib: MILES', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','0','MIST',
          'BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[7], 
          r'isoc_lib: MIST', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','delayed-exponential','0.0190','0.0','0',
          'PADOVA','BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[8],
          r'SFH: Delayed-exponential', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.2','0','PADOVA',
          'BASEL','100','1', '1.6', 'g', 'r', 'r'), self.A_max, self.s_max, c[9],
          r'fhbh$=0.2$', self.add_contours)

        plot_arrow(
          self.ax, pars2fpath('Kroupa','exponential','0.0190','0.0','0','PADOVA',
          'BASEL','100','1', '0.0', 'g', 'r', 'r'), self.A_max, self.s_max, c[10], r'Q=0',
          self.add_contours)

        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, labelspacing=.5, loc=3) 

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_systematics.pdf'
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.set_fig_frame()
        self.plot_default_contour()
        self.add_arrows()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(add_contours=False, show_fig=False, save_fig=True)
