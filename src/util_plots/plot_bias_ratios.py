#!/usr/bin/env python

import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from stats import mag2lum

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fs=24.
c= ['#1b9e77','#d95f02','#7570b3']
m = ['s', 'o', '^']
ls = ['-', '-.', ':']
z_steps = [0.05, 0.1, 0.2]

def bin_data(z, qtty, hosts):
    """Bin a quantity (color or magnitude) in two redshift ranges.
    Only the hosts are binned. Returns the median of the qtty in each bin.
    """
    z_h, qtty_h = z[hosts], qtty[hosts]
    cond1 = ((0.01 <= z_h) & (z_h < 0.2))
    cond2 = ((0.2 <= z_h) & (z_h < 0.4))
    z_1, qtty_1 = np.median(z_h[cond1]), np.median(qtty_h[cond1])
    z_2, qtty_2 = np.median(z_h[cond2]), np.median(qtty_h[cond2])
    return np.array([z_1,qtty_1]), np.array([z_2,qtty_2])

def bin_data_fine(z, qtty, hosts, z_step, sample):
    """Bin a quantity (color or magnitude) in two redshift ranges.
    Only the hosts are binned. Returns the median of the qtty in each bin.
    """
    z_out, qtty_out = [], []
    z_edge = np.arange(0.,0.4001,z_step)
    
    if sample == 'hosts':
        Z, Q = z[hosts], qtty[hosts]
    elif sample == 'control':
        Z, Q = np.copy(z), np.copy(qtty)
    
    for z_l, z_r in zip(z_edge,z_edge[1:]):
        cond = ((Z >= z_l) & (Z < z_r))
        z_out.append(np.median(Z[cond]))
        qtty_out.append(np.median(Q[cond]))
    return z_edge, np.array(qtty_out)
    

class Bias_Ratios(object):
    """
    Description:
    ------------
    Makes a figure displaying the the median color of the hosts in two
    redshift bins: 0.01 < z < 0.2 and 0.2 < z < 0.4. This is a simple test
    of potential bias where SNe are detected preferentially in redder/brigther
    hosts at higher redshift. 

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../../OUTPUT_FILES/RUNS/$RUN_DIR/FIGURES/Fig_z_bias.pdf
    """        
    def __init__(self, _inputs):
                
        self._inputs = _inputs

        self.fig, (self.ax1, self.ax2) = plt.subplots(
          1,2, figsize=(16,8), sharey=False)
        
        self.z, self.color, self.mag, self.hosts = None, None, None, None
        self.df = None
        
        self.make_plot()
        
    def set_fig_frame(self):

        plt.subplots_adjust(bottom=0.14, left=0.1, right=.95, wspace=0.3)
        
        f1 = self._inputs.filter_1
        f2 = self._inputs.filter_2

        self.ax1.set_xlabel(r'$z$', fontsize=fs)
        aux = '(M_' + f2 + '- M_' + f1 + ')'
        self.ax1.set_ylabel(
          r'$' + aux + '_{\mathrm{SN}} - ' + aux + '_{\mathrm{all}}$', fontsize=fs)
        self.ax1.set_xlim(0., 0.4)
        self.ax1.set_ylim(-0.2, 0.35)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.minorticks_off()
        self.ax1.tick_params('both', length=8, width=2., which='major')
        self.ax1.tick_params('both', length=4, width=2., which='minor')    
        self.ax1.xaxis.set_minor_locator(MultipleLocator(.05))
        self.ax1.xaxis.set_major_locator(MultipleLocator(.1))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.05))
        self.ax1.yaxis.set_major_locator(MultipleLocator(.1))  
        self.ax1.axhline(y=0., ls=':', lw=1., color='gray')

        self.ax2.set_xlabel(r'$z$', fontsize=20.)
        self.ax2.set_ylabel(r'$L_{' + f1 + ',\mathrm{SN}}\,/\,L_{' + f1
                            + ',\mathrm{all}}$', fontsize=fs)
        self.ax2.set_xlim(0., 0.4)
        self.ax2.set_ylim(0., 7.)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.minorticks_off()
        self.ax2.tick_params('both', length=8, width=2., which='major')
        self.ax2.tick_params('both', length=4, width=2., which='minor')    
        self.ax2.xaxis.set_minor_locator(MultipleLocator(.05))
        self.ax2.xaxis.set_major_locator(MultipleLocator(.1))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax2.yaxis.set_major_locator(MultipleLocator(2.))  
        self.ax2.axhline(y=1., ls=':', lw=1., color='gray')

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
        f1 = self.df['abs_' + self._inputs.filter_1].values
        f2 = self.df['abs_' + self._inputs.filter_2].values
        self.z = self.df['z'].values
        self.mag = f1
        self.color = f2 - f1  
        self.hosts = self.df['is_host'].values
    
    def plot_quantities(self):

        for i, z_step in enumerate(z_steps):
            zErr = z_step / 2.

            #Left panel: color.
            z_edge, color_fine_h = bin_data_fine(
              self.z, self.color, self.hosts, z_step, 'hosts')
            z_edge, color_fine_c = bin_data_fine(
              self.z, self.color, self.hosts, z_step, 'control')            
            
            z_plot = (z_edge[0:-1] + z_edge[1:]) / 2.
            self.ax1.errorbar(
              z_plot, color_fine_h - color_fine_c, xerr=zErr, ls='--',
              marker='None', color=c[i], capsize=0., lw=0.5, elinewidth=4.,
              label=r'$\Delta z=' + str(z_step) + '$')

            #Right panel: flux.
            z_edge, mag_fine_h = bin_data_fine(
              self.z, self.mag, self.hosts, z_step, 'hosts')
            z_edge, mag_fine_c = bin_data_fine(
              self.z, self.mag, self.hosts, z_step, 'control')
            
            L_h, L_c = mag2lum(mag_fine_h), mag2lum(mag_fine_c)

            z_plot = (z_edge[0:-1] + z_edge[1:]) / 2.
            self.ax2.errorbar(
              z_plot, L_h / L_c, xerr=zErr, ls='--', marker='None',
              color=c[i], capsize=0., lw=0.5, elinewidth=4.)

        self.ax1.legend(frameon=False, fontsize=fs, numpoints=1, loc='best')

    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/Fig_bias_ratios.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.plot_quantities()
        self.manage_output()             
        plt.close(self.fig)    
