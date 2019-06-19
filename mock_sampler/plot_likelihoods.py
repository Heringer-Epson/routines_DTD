#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 24.
c = ['#1b9e77','#d95f02','#7570b3']

class Plot_Likelihoods(object):
    """
    Code Description
    ----------    
    TBW.

    Parameters:
    -----------
    A : ~float
        DTD normalization.
    s : ~float
        DTD slope.
        
    Outputs:
    --------
    TBW
    """
    
    def __init__(self, inputs, A, s1, s2, survey_t):
        self.inputs, self.A, self.s1, self.s2 = inputs, A, s1, s2
        self.survey_t = survey_t

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.make_plot()

    def set_fig_frame(self):        
        x_label = r'$\mathrm{log}\, A$'
        y_label = r'$s$'
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.xaxis.set_major_locator(MultipleLocator(.5)) 
        self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.yaxis.set_major_locator(MultipleLocator(.5)) 
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')    
        
        self.ax.axhline(y=self.s1, ls=':', lw=2., c='gray')      
        self.ax.axvline(x=np.log10(self.A), ls=':', lw=2., c='gray')      

    def plot_contours(self):
                       
        #CL
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/sSNRL_s1_s2.csv'
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)           
        x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
        nx, ny = len(np.unique(x)), len(np.unique(y))
        
        stats.plot_contour(self.ax, np.log10(x), y, z, c[0], nx, ny, r'$\tt{CL}$')

        print s1[ln_L.argmax()], s2[ln_L.argmax()]
        print x[z.argmax()], y[z.argmax()]

        #X, Y, XErr, YErr = stats.plot_contour(
        #  self.ax, np.log10(x), y, z, c[1], nx, ny, r'$\tt{vespa}$')
        #self.out.add_line('VESPA', X, Y, XErr, YErr)

        #Vespa
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/vespa_s1_s2.csv'
        N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)           
        x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
        nx, ny = len(np.unique(x)), len(np.unique(y))
        
        print s1[ln_L.argmax()], s2[ln_L.argmax()]
        print x[z.argmax()], y[z.argmax()]
        
        stats.plot_contour(self.ax, np.log10(x), y, z, c[1], nx, ny, r'$\tt{SFHR}$')

        #X, Y, XErr, YErr = stats.plot_contour(
        #  self.ax, np.log10(x), y, z, c[1], nx, ny, r'$\tt{vespa}$')
        #self.out.add_line('VESPA', X, Y, XErr, YErr)
              
        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, ncol=1,
          loc=2, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)    

    def manage_output(self):
        plt.tight_layout()
        if self.inputs.save_fig:
            fpath = './../OUTPUT_FILES/MOCK_SAMPLE/Fig_mock_contours.pdf'
            plt.savefig(fpath, format='pdf')
        if self.inputs.show_fig:
            plt.show() 

    def make_plot(self):
        self.set_fig_frame()
        self.plot_contours()
        self.manage_output()

