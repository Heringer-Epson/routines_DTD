#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm

class Plot_Dcolor(object):
    
    def __init__(self, _inputs):

        self._inputs = _inputs

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.run_calculation()

    def set_fig_frame(self):
        
        f1 = self._inputs.filter_1
        f2 = self._inputs.filter_2

        x_label = r'$\Delta (' + f2 + '-' +f1 + ')$'
        y_label = r'Count'
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xlim(self._inputs.bin_range[0], self._inputs.bin_range[1])
        #self.ax.set_ylim(-.2, 1.2)
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        self.ax.yaxis.set_minor_locator(MultipleLocator(100))
        self.ax.yaxis.set_major_locator(MultipleLocator(500))  

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        df = pd.read_csv(fpath, header=0)
        f1, f2 = self._inputs.filter_1, self._inputs.filter_2
        self.Dcolor = df['Dcolor_' + f2 + f1].values           
        self.hosts = df['n_SN'].values
        
        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        self.RS_mu, self.RS_std = np.loadtxt(fpath, delimiter=',', skiprows=1,
                                             usecols=(0,1), unpack=True)                
        self.RS_mu, self.RS_std = float(self.RS_mu), float(abs(self.RS_std))

    def make_histogram(self):

        acc_cond = ((self.Dcolor >= self._inputs.Dcolor_range[0]) &
                    (self.Dcolor <= self._inputs.Dcolor_range[1]))

        Dcolor_acc = self.Dcolor[acc_cond]
        Dcolor_rej = self.Dcolor[np.logical_not(acc_cond)]
        Dcolor_hosts = self.Dcolor[self.hosts]

        #Further remove spurious objects which are outside the plotting range.
        rej_cond = ((Dcolor_rej >= self._inputs.bin_range[0])
                    & (Dcolor_rej <= self._inputs.bin_range[1]))
        Dcolor_rej = Dcolor_rej[rej_cond]
        
        bins = np.arange(self._inputs.bin_range[0], self._inputs.bin_range[1]
                         + 1.e-5, self._inputs.bin_size)
        
        amp = self._inputs.bin_size * float(len(Dcolor_acc))
        p = norm.pdf(bins, self.RS_mu, self.RS_std) 

        #For the legend.
        self.ax.plot([np.nan], [np.nan], lw=10., color='dimgray', ls='-',
                     label='Accepted data')
        self.ax.plot([np.nan], [np.nan], lw=10., color='lightgray', ls='-',
                     label='Rejected data')  
        self.ax.plot([np.nan], [np.nan], lw=10., color='dodgerblue', ls='-',
                     label=r'Hosts ($\times 100$)') 
                           
        hist, bins, patches = plt.hist(Dcolor_acc, bins=bins, 
                                       facecolor='dimgray')  
        hist, bins, patches = plt.hist(Dcolor_rej, bins=bins, 
                                       facecolor='lightgray')
        hist, bins, patches = plt.hist(list(Dcolor_hosts) * 100, bins=bins, 
                                       facecolor='dodgerblue')         
        
        plt.plot(bins, amp * p, 'r', lw=2., label='Gaussian fit')
        
        N_ctrl = str(len(Dcolor_acc) + len(Dcolor_rej))
        N_hosts = str(len(Dcolor_hosts))
        plt.title(
          r'$\mu = ' + str(format(self.RS_mu, '.3f')) + ', \sigma = '
          + str(format(self.RS_std, '.3f')) + ', \mathrm{N_{ctrl}} = ' + N_ctrl
          + ', \mathrm{N_{host}} = ' + N_hosts + '$', fontsize=20.)
        
        self.ax.legend(frameon=True, fontsize=20., numpoints=1, loc=2) 

    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/fit_Dcolor.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_calculation(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.make_histogram()
        self.manage_output()             
