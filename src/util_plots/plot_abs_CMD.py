#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class Abs_CMD(object):
    """
    Description:
    ------------
    Makes a figure displaying the control and host galaxies in the parameter
    space of absolute r vs absolute color.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/CMD_abs.pdf
    """        
    def __init__(self, _inputs):
                
        self._inputs = _inputs

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.make_plot()
        
    def set_fig_frame(self):
        
        f1 = self._inputs.f1
        f2 = self._inputs.f2

        x_label = r'$M_' + f1 + '$'
        y_label = r'$M_' + f2 + '- M_' + f1 + '$'
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xlim(-24., -16.)
        self.ax.set_ylim(0.2, 0.9)
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        self.ax.xaxis.set_minor_locator(MultipleLocator(1.))
        self.ax.xaxis.set_major_locator(MultipleLocator(2.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
        self.ax.yaxis.set_major_locator(MultipleLocator(.1))  

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_absmag.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
        f1 = self.df['abs_' + self._inputs.f1].values
        f2 = self.df['abs_' + self._inputs.f2].values
        self.mag = f1
        self.color = f2 - f1  
        self.hosts = self.df['is_host'].values
    
    def plot_quantities(self):

        #For the legend.
        self.ax.plot(self.mag, self.color, ls='None', marker=',', color='k', 
                     label='Control')
        self.ax.plot(self.mag[self.hosts], self.color[self.hosts], ls='None',
                     marker='^', color='b', label='Hosts')        
        N_ctrl = str(len(self.mag))
        N_hosts = str(len(self.mag[self.hosts]))
        plt.title(r'K-corrected. $\mathrm{N_{ctrl}} = ' + N_ctrl
          + ', \mathrm{N_{host}} = ' + N_hosts + '/132$', fontsize=20.)
        
        self.ax.legend(frameon=True, fontsize=20., numpoints=1, loc=1)

    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/CMD_abs.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.plot_quantities()
        self.manage_output()             
        plt.close(self.fig)    
