#!/usr/bin/env python

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 24.
c = ['k', 'darkgray', 'dodgerblue']
a = [.9, .8, 1.]

class Plot_CMD(object):
    """
    Description:
    ------------
    Makes Fig. 1 in the paper, displaying the control and host galaxies in the 
    parameter space of absolute r vs absolute color. A bottom panel shows a 
    histogram of colors. Data will be retrieved from a 'default' run.

    Parameters:
    -----------
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.

    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_CMD.pdf
    """        
    def __init__(self, x_range, show_fig, save_fig):
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.x_range = x_range

        fig, (self.ax1, self.ax2) = plt.subplots(
          2,1, figsize=(10,10), gridspec_kw = {'height_ratios':[2, 1]},
          sharex=True)
        self.df = None
                
        self.make_plot()
        
    def set_fig_frame(self):

        plt.subplots_adjust(hspace=0.02)
    
        x_label = r'$M_g - M_r$'
        y1_label = r'$M_r$'
        y2_label = r'Count'
        
        self.ax1.set_ylabel(y1_label, fontsize=fs)
        self.ax1.set_xlim(self.x_range[0], self.x_range[1])
        self.ax1.set_ylim(-24., -16.)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params('both', length=8, width=1., which='major',
                             direction='in', right=True, top=True)
        self.ax1.tick_params('both', length=4, width=1., which='minor',
                             direction='in', right=True, top=True) 
        self.ax1.xaxis.set_minor_locator(MultipleLocator(.05))
        self.ax1.xaxis.set_major_locator(MultipleLocator(.1))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax1.yaxis.set_major_locator(MultipleLocator(2.))  
        plt.setp(self.ax1.get_xticklabels(), visible=False)
    
        self.ax2.set_xlabel(x_label, fontsize=fs)
        self.ax2.set_ylabel(y2_label, fontsize=fs)
        self.ax2.set_ylim(0., 1100.)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params('both', length=8, width=1., which='major',
                             direction='in', right=True, top=True)
        self.ax2.tick_params('both', length=4, width=1., which='minor',
                             direction='in', right=True, top=True)   
        self.ax2.yaxis.set_minor_locator(MultipleLocator(100.))
        self.ax2.yaxis.set_major_locator(MultipleLocator(500.))  

        plt.gca().invert_xaxis()

    def retrieve_data(self):
        fpath = './../OUTPUT_FILES/RUNS/default/data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
    
    def plot_quantities(self):

        #Remove objects outside shown range.
        cond = ((self.df['Dcolor_gr'].values >= self.x_range[0])
                & (self.df['Dcolor_gr'].values <= self.x_range[1]))
        Dcolor = self.df['Dcolor_gr'].values[cond]
        r_abs = self.df['abs_r'].values[cond]
        hosts = self.df['is_host'].values[cond]
        
        r_err = self.df['petroMagErr_g'].values[cond]
        Dcolor_err = np.sqrt(self.df['petroMagErr_g'].values[cond]**2.
                             + self.df['petroMagErr_r'].values[cond]**2.)
                                     
        #Distinguish between objects that are used in sSNRL or not.
        cond_acc = ((Dcolor >= -0.4) & (Dcolor <= .08))
        cond_rej = np.logical_not(cond_acc)
        hosts_acc = (hosts & (Dcolor >= -0.4) & (Dcolor <= .08))
        hosts_rej = (hosts & ((Dcolor < -0.4) | (Dcolor > .08)))

        #Draw CMD (on self.ax1).
        self.ax1.plot(Dcolor[cond_acc], r_abs[cond_acc], ls='None', marker='o',
                      markersize=2., color=c[0], alpha=a[0], zorder=1.)
        self.ax1.plot(Dcolor[cond_rej], r_abs[cond_rej], ls='None', marker='o',
                      markersize=2., color=c[1], alpha=a[1], zorder=1.)

        self.ax1.errorbar(
          Dcolor[hosts_acc], r_abs[hosts_acc], xerr=r_err[hosts_acc],
          capsize=0., elinewidth=1., zorder=2.,
          ls='None', marker='*', markersize=10., color=c[2], alpha=a[2])
        self.ax1.errorbar(
          Dcolor[hosts_rej], r_abs[hosts_rej], xerr=r_err[hosts_rej],
          capsize=0., elinewidth=1., zorder=2., fillstyle='none',
          ls='None', marker='*', markersize=10., color=c[2], alpha=a[2])        
        
        #Draw hostograms (on self.ax2).
        bins = np.arange(self.x_range[0], self.x_range[1] + 1.e-5, 0.01)
        Dcolor_hosts = np.repeat(Dcolor[hosts_acc], 100)
        
        self.ax2.hist(
          Dcolor[cond_acc], bins=bins, align='mid', color=c[0], alpha=a[0])
        self.ax2.hist(
          Dcolor[cond_rej], bins=bins, align='mid', color=c[1], alpha=a[1])
        self.ax2.hist(
          Dcolor_hosts, bins=bins, align='mid', color=c[2], alpha=a[2])

    def make_legend(self):
        self.ax2.plot(
          [np.nan], [np.nan], ls='-', marker='None', lw=15., color=c[0],
          alpha=a[0], label=r'Control Galaxies')
        self.ax2.plot(
          [np.nan], [np.nan], ls='-', marker='None', lw=15., color=c[2],
          alpha=a[2], label=r'Hosts $(\times\, 100)$')
        self.ax2.plot(
          [np.nan], [np.nan], ls='-', marker='None', lw=15., color=c[1],
          alpha=a[1], label=r'Rejected Galaxies')
                      
        self.ax2.legend(frameon=False, fontsize=fs, numpoints=1, loc=1)

    def manage_output(self):
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_CMD.pdf'
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show()
        plt.close(self.fig)    

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.plot_quantities()
        self.make_legend()
        self.manage_output()             

if __name__ == '__main__':
    Plot_CMD(x_range=(-0.5,.1), show_fig=True, save_fig=True)
