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

class Ext_CMD(object):
    """
    Description:
    ------------
    Makes a figure displaying the control and host galaxies in the r vs color
    parameter space, after the bands are corrected for the MW extinction.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/CMD_ext.pdf
    """        
    def __init__(self, _inputs):
                
        self._inputs = _inputs

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.make_plot()
        
    def set_fig_frame(self):
        
        f1 = self._inputs.filter_1
        f2 = self._inputs.filter_2

        x_label = r'$' + f1 + '$'
        y_label = r'$' + f2 + '- ' + f1 + '$'
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xlim(14., 18.)
        self.ax.set_ylim(0., 1.6)
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        self.ax.xaxis.set_minor_locator(MultipleLocator(.5))
        self.ax.xaxis.set_major_locator(MultipleLocator(1.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
        self.ax.yaxis.set_major_locator(MultipleLocator(.2))  

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_absmag.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
        f1 = self.df['ext_' + self._inputs.filter_1].values
        f2 = self.df['ext_' + self._inputs.filter_2].values
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
        plt.title(r'Extinction-corrected. $\mathrm{N_{ctrl}} = ' + N_ctrl
          + ', \mathrm{N_{host}} = ' + N_hosts + '/132$', fontsize=20.)
        
        self.ax.legend(frameon=True, fontsize=20., numpoints=1, loc=2)

    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/CMD_ext.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.plot_quantities()
        self.manage_output()             
        plt.close(self.fig)    
