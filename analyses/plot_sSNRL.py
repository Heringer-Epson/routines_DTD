#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from SN_rate import Model_Rates

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'  
fs = 20.   

class Input_Pars(object):
    """
    Description:
    ------------
    Define a set of input parameters to use to make the Dcolor vs SN rate plot
    in the class below. This is intended to replicate the ./../src/ code
    input_params.py, but only containing the relevant quantities for this plot.

    Parameters:
    -----------
    As described in ./../src/input_params.py
    """  
    def __init__(self):
        self.subdir_fullpath = './../../INPUT_FILES/'
        self.sfh_type = 'exponential'
        self.t_onset = 1.e8 * u.yr
        self.t_cutoff = 1.e9 * u.yr

class Plot_sSNRL(object):
    """
    Description:
    ------------
    Makes a plot of SN rate (per unit of luminosity) as a function of Dcolor.
    Replicates Fig. 3 in Heringer+ 2017.

    Parameters:
    -----------
    sfh_type : ~str
        'exponential' or 'delayed-exponential'. Which SFH to use.
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.

    Notes:
    ------
    The normalization used here is different than in Heringer+ 2017. In that
    paper the DTD is normalized at 0.5 Gyr, whereas here an arbitraty
    constant (A=10**-12) is given to the DTD of the form SN rate = A*t**s1
    where s1 is the slope prioir to 1Gyr. 
             
    Outputs:
    --------
    ./../../OUTPUT_FILES/ANALYSES_FIGURES/Fig_Dcolor-sSNRL_X.pdf
        where X denotes the SFH adopted, 'exponential' or 'delayed-exponential'.
    
    References:
    -----------
    Heringer+ 2017: http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    """         
    def __init__(self, show_fig=True, save_fig=False):
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
        
        self.inputs = Input_Pars()
        self.make_plot()
        
    def set_fig_frame(self):
        
        x_label = r'$\Delta (g-r)$'
        y_label = r'$\rm{log\ sSNR}_L\ \rm{[yr^{-1}\ L_\odot^{-1}]}$'
        
        self.ax.set_xlabel(x_label,fontsize=fs)
        self.ax.set_ylabel(y_label,fontsize=fs)
        self.ax.set_xlim(-0.6,0.1)      
        self.ax.set_ylim(-14.5,-11.5)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.minorticks_on()
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        self.ax.xaxis.set_major_locator(MultipleLocator(0.1))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.yaxis.set_major_locator(MultipleLocator(0.5))  
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')

    def plot_models(self):
        offset = [0.17, 0.25, 0., -0.1, -0.12, -0.2]
        for i, (s1,s2) in enumerate(zip([-0.5, -0.5, -1., -1.25, -1., -1.],
                                        [-0.5, -1., -1., -1.25, -2., -3.])):
        #for s1, s2 in zip([-1., -1., -3.],
        #                  [-1., -2., -1.]):
            
            x_10Gyr, y_10Gyr = [], []
            for tau in [1., 1.5, 2., 3., 4., 5., 7., 10.]:                
                model = Model_Rates(self.inputs, 1.e-12, s1, s2, tau * u.Gyr)

                age_cond = (model.age.to(u.yr).value <= 1.e10)
                
                x = model.Dcolor[age_cond]
                sSNRL = model.sSNRL[age_cond]
                sSNRL[sSNRL <= 0.] = 1.e-40
                y = np.log10(sSNRL)
            
                self.ax.plot(x, y + offset[i], color='r', ls='-', lw=1.5)
                
                #Get values at 10 Gyr for interpolation.
                marker_cond = (model.age.to(u.yr).value == 1.e10)
                x_marker = model.Dcolor[marker_cond]
                y_marker = np.log10(model.sSNRL[marker_cond])
            
                x_10Gyr.append(x_marker), y_10Gyr.append(y_marker)
            
            #Plot markers at 10Gyrs for each of the SFH.
            self.ax.plot(x_10Gyr, np.array(y_10Gyr) + offset[i], color='b',
                         ls='--', lw=1.5, marker='s')
            
            #Add label text for each DTD.
            if s1 != s2:
                label = str(s1) + '/' + str(s2)
            else:
                label = str(s1)                    
            self.ax.text(x_10Gyr[0] + 0.02, y_10Gyr[0] + offset[i] - 0.1,
                         label, color='k', fontsize=fs)

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = ('./../../OUTPUT_FILES/ANALYSES_FIGURES/Fig_Dcolor-sSNRL_'
                     + self.inputs.sfh_type + '.pdf')
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show() 
            
    def make_plot(self):
        self.set_fig_frame()
        self.plot_models()
        plt.tight_layout()
        self.manage_output()

if __name__ == '__main__':
    Plot_sSNRL(show_fig=True, save_fig=True)
 
