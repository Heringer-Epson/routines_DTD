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
fs = 28.
#c = ['k', 'darkgray', 'dodgerblue']
#c = ['#4daf4a', '#878787', '#377eb8']
c = ['#fdae61', '#3288bd']

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

class Plot_CMD(object):
    """
    Description:
    ------------
    Makes the Fig. 1 of the DTD paper, displaying the control and host galaxies 
    in the  parameter space of absolute r vs absolute color. A bottom panel 
    shows a histogram of colors. Data will be retrieved from a 'default' run.

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
    def __init__(self, x_range, y_range, show_fig, save_fig):
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.x_range = x_range
        self.y_range = y_range

        self.fig = plt.figure(1, figsize=(14, 12))
        self.ax = plt.axes(rect_scatter)
        self.axx = plt.axes(rect_histx, sharex=self.ax)
        self.axy = plt.axes(rect_histy, sharey=self.ax)
                
        self.make_plot()
        
    def set_fig_frame(self):
    
        x_label = r'$\Delta(g - r)$'
        y1_label = r'$M_r$'
        hist_label = r'Count'
        
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y1_label, fontsize=fs)
        self.ax.set_xlim(self.x_range[0], self.x_range[1])
        self.ax.set_ylim(self.y_range[0], self.y_range[1])
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params('both', length=8, width=1., which='major',
                             direction='in', right=True, top=True)
        self.ax.tick_params('both', length=4, width=1., which='minor',
                             direction='in', right=True, top=True) 
        self.ax.xaxis.set_minor_locator(MultipleLocator(.05))
        self.ax.xaxis.set_major_locator(MultipleLocator(.1))
        self.ax.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(1.))  
    
        self.axx.set_ylabel(hist_label, fontsize=fs)
        self.axx.set_ylim(0., 1100.)
        self.axx.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.axx.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.axx.tick_params('both', length=8, width=1., which='major',
                             direction='in', right=True, top=True)
        self.axx.tick_params('both', length=4, width=1., which='minor',
                             direction='in', right=True, top=True)   
        self.axx.yaxis.set_minor_locator(MultipleLocator(100.))
        self.axx.yaxis.set_major_locator(MultipleLocator(500.))  
        plt.setp(self.axx.get_xticklabels(), visible=False)

        self.axx.invert_xaxis()

        self.axy.set_xlabel(hist_label, fontsize=fs)
        self.axy.set_xlim(0., 1100.)
        self.axy.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.axy.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.axy.tick_params('both', length=8, width=1., which='major',
                             direction='in', right=True, top=True)
        self.axy.tick_params('both', length=4, width=1., which='minor',
                             direction='in', right=True, top=True)   
        self.axy.xaxis.set_minor_locator(MultipleLocator(100.))
        self.axy.xaxis.set_major_locator(MultipleLocator(500.))  
        plt.setp(self.axy.get_yticklabels(), visible=False)

    def retrieve_data(self):
        fpath = './../OUTPUT_FILES/RUNS/default/data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
    
    def plot_quantities(self):

        #Remove objects outside shown range.
        Dcolor = self.df['Dcolor_gr'].values
        r_abs = self.df['abs_r'].values
        hosts = self.df['is_host'].values
                
        r_err = self.df['petroMagErr_g'].values
        Dcolor_err = np.sqrt(self.df['petroMagErr_g'].values**2.
                             + self.df['petroMagErr_r'].values**2.)

        #Draw CMD (on self.ax).
        self.ax.plot(Dcolor, r_abs, ls='None', marker='o', markersize=2.,
                     color=c[0], zorder=1.)

        self.ax.errorbar(
          Dcolor[hosts], r_abs[hosts], xerr=r_err[hosts], capsize=0., elinewidth=1.,
          zorder=2., ls='None', marker='*', markersize=10., color=c[1])      
        
        #Draw histogram (on self.axx).
        xbins = np.arange(self.x_range[0], self.x_range[1] + 1.e-5, 0.01)
        Dcolor_hosts = np.repeat(Dcolor[hosts], 100)
        
        self.axx.hist(Dcolor, bins=xbins, align='mid', color=c[0])
        self.axx.hist(Dcolor_hosts, bins=xbins, align='mid', color=c[1])

        #Draw histogram (on self.axy).
        ybins = np.arange(self.y_range[0], self.y_range[1] + 1.e-5, 0.1)
        r_abs_hosts = np.repeat(r_abs[hosts], 100)

        self.axy.hist(
          r_abs, bins=ybins, align='mid', color=c[0],orientation='horizontal')
        self.axy.hist(
          r_abs_hosts, bins=ybins, align='mid', color=c[1], orientation='horizontal')

    def make_legend(self):
        self.axx.plot(
          [np.nan], [np.nan], ls='-', marker='None', lw=15., color=c[0],
          label=r'Control Galaxies')
        self.axx.plot(
          [np.nan], [np.nan], ls='-', marker='None', lw=15., color=c[1],
          label=r'Hosts $(\times\, 100)$')

        self.axx.legend(frameon=False, fontsize=fs, numpoints=1, loc=1)

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
    Plot_CMD(x_range=(-0.55,.2), y_range=(-23.5,-17.5), show_fig=False, save_fig=True)
