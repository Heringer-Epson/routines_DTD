#!/usr/bin/env python

import os
import sys
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from generic_input_pars import Generic_Pars
from build_fsps_model import Build_Fsps

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from Dcolor2sSNRL_gen import Generate_Curve
from SN_rate import Model_Rates

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'  
fs = 24.   
taus = [1., 1.5, 2., 3., 4., 5., 7., 10.]

s1s2 = zip([-0.5, -1., -1.5, -3., -1., -1.],[-1., -1., -1.5, -1., -2.,-3.])
label = [r'-0.5/-1', r'-1/-1', r'-1.5/-1.5', r'-3/-1', r'-1/-2', r'-1/-3']
offset = [0.96, 0.9, 1.14, 1.51, 0.2, 0.17]

class Plot_sSNRL(object):
    """
    Description:
    ------------
    Makes the Fig. 1 of the DTD paper, displaying SN rate (per unit of
    luminosity) as a function of Dcolor. 4 panels are included, showing the
    impact of choosing different parameters, such as the IMF, the metallicity,
    the SFH and the time of onset of SNe Ia. Similar replicates Fig. 3
    in Heringer+ 2017.

    Parameters:
    -----------
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.

    Notes:
    ------
    The normalization used here is different than in Heringer+ 2017. In that
    paper the DTD is normalized at 0.5 Gyr, whereas here an arbitraty
    constant (A=10**-12) is given to the DTD of the form SN rate = A*t**s1
    where s1 is the slope prior to 1 Gyr. 
             
    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_sSNRL.pdf
    
    References:
    -----------
    Heringer+ 2017: http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    """         
    def __init__(self, show_fig=True, save_fig=False):
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.fig, (self.ax1, self.ax2) = plt.subplots(
          1,2, figsize=(16,10), sharey=True)
      
        self.make_plot()
                
    def set_fig_frame(self):

        plt.subplots_adjust(wspace=0.03)
        
        x_label = r'$\Delta (g-r)$'
        y_label = r'$\rm{log\ sSNR}_L\ \rm{[yr^{-1}\ L_\odot^{-1}]}$'
        
        self.ax1.set_xlabel(x_label, fontsize=fs)
        self.ax1.set_ylabel(y_label, fontsize=fs)
        self.ax1.set_xlim(-1.05,0.35)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params('both', length=8, width=1., which='major',
                                 direction='in', right=True, top=True)
        self.ax1.tick_params('both', length=4, width=1., which='minor',
                                 direction='in', right=True, top=True) 
        self.ax1.xaxis.set_minor_locator(MultipleLocator(.05))
        self.ax1.xaxis.set_major_locator(MultipleLocator(.2))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.25))
        self.ax1.yaxis.set_major_locator(MultipleLocator(.5))  

        self.ax2.set_xlabel(x_label, fontsize=fs)
        self.ax2.set_xlim(-1.05,0.35)
        self.ax2.set_ylim(-15.5,-11.5)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params('both', length=8, width=1., which='major',
                                 direction='in', right=True, top=True)
        self.ax2.tick_params('both', length=4, width=1., which='minor',
                                 direction='in', right=True, top=True) 
        self.ax2.xaxis.set_minor_locator(MultipleLocator(.05))
        self.ax2.xaxis.set_major_locator(MultipleLocator(.2))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(.25))
        self.ax2.yaxis.set_major_locator(MultipleLocator(.5))  
        plt.setp(self.ax2.get_yticklabels(), visible=False)

    def plot_models(self):

        #Add SFH text. 
        self.ax1.text(-0.98, -11.7, 'SFH: exponential', fontsize=fs )
        self.ax2.text(-0.98, -11.7, 'SFH: delayed exponential', fontsize=fs )
        
        for l, sfh in enumerate(['exponential', 'delayed-exponential']):
            _inputs = Generic_Pars(sfh)
            _D = Build_Fsps(_inputs).D

            if l == 0:
                ax = self.ax1
            elif l == 1:
                ax = self.ax2
            
            for i, (s1,s2) in enumerate(s1s2):
                Sgen = Generate_Curve(_inputs, _D, s1, s2)
                
                x = Sgen.Dcolor_at10Gyr
                y = np.log10(Sgen.sSNRL_at10Gyr * 1.e-12) + offset[i]                    
                ax.plot(x, y, ls='None', marker='s', markersize=12., color='b',
                        fillstyle='none', zorder=2)

                #Plot Dcolor-sSNRL for each tau.
                for tau in _inputs.tau_list:
                    TS = str(tau.to(u.yr).value / 1.e9)
                    model = Model_Rates(_inputs, _D, TS, s1, s2)
                    x = _D['Dcolor_' + TS]
                    y = np.log10(model.sSNRL * 1.e-12) + offset[i]
                    ax.plot(x, y, ls='-', marker='None', color='r',
                            linewidth=.8, alpha=0.7, zorder=1)                    

                #Add extended models.
                x = Sgen.Dcd_fine
                y = np.log10(Sgen.sSNRL_fine * 1.e-12) + offset[i]
                ax.plot(x, y, ls=':', marker='None', markersize=8.,
                        color='forestgreen', linewidth=4., zorder=3)                                

                ax.text(0.05, y[-1] + 0.05, label[i], color='k', fontsize=fs)
                        
            _inputs.clean_fsps_files()

    def manage_output(self):
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_sSNRL.pdf'
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show() 
        plt.close()
            
    def make_plot(self):
        self.set_fig_frame()
        self.plot_models()
        self.manage_output()

if __name__ == '__main__':
    Plot_sSNRL(show_fig=True, save_fig=False)
 
