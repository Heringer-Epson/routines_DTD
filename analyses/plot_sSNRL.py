#!/usr/bin/env python

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u

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

def copy_fsps(_sfh_type):
    inppath = ('./../INPUT_FILES/fsps_FILES/Chabrier_' + _sfh_type
               + '_0.0190_BASEL_PADOVA/')
    tgtpath = './fsps_FILES/'
    if os.path.isdir(tgtpath):
        shutil.rmtree(tgtpath)
    shutil.copytree(inppath, tgtpath)

def retrieve_data(_inputs):
    """Anything that does not depend on s1 or s2, should be computed here
    to avoid wasting computational time.
    """
    D = {}

    D['Dcd_fine'] = np.arange(-1.1, 1.00001, 0.01)
    
    #General calculations. unit conversion.
    D['t_ons'] = _inputs.t_onset.to(u.Gyr).value
    D['t_bre'] = _inputs.t_cutoff.to(u.Gyr).value
    
    #Get SSP data and compute the theoretical color with respect to the RS.
    synpop_dir = _inputs.subdir_fullpath + 'fsps_FILES/'
    df = pd.read_csv(synpop_dir + 'SSP.dat', header=0, escapechar='#')
    logage_SSP = df[' log_age'].values
    mag_2_SSP = df[_inputs.filter_2].values
    mag_1_SSP = df[_inputs.filter_1].values
    
    #Retrieve RS color.
    RS_condition = (logage_SSP == 10.0)
    RS_color = mag_2_SSP[RS_condition] - mag_1_SSP[RS_condition]

    for i, tau in enumerate(_inputs.tau_list):
        TS = str(tau.to(u.yr).value / 1.e9)
     
        model = pd.read_csv(synpop_dir + 'tau-' + TS + '.dat', header=0)
        D['tau_' + TS] = tau.to(u.Gyr).value
        D['mag2_' + TS] = model[_inputs.filter_2].values
        D['mag1_' + TS] = model[_inputs.filter_1].values
        D['age_' + TS] = 10.**(model['# log_age'].values) / 1.e9 #Converted to Gyr.
        D['int_mass_' + TS] = model['integrated_formed_mass'].values
        D['Dcolor_' + TS] = (D['mag2_' + TS] - D['mag1_' + TS] - RS_color) 

        #Get analytical normalization for the SFH.
        if _inputs.sfh_type == 'exponential':
            D['sfr_norm_' + TS] = (
              -1. / (D['tau_' + TS] * (np.exp(-D['age_' + TS][-1] / D['tau_' + TS])
              - np.exp(-D['age_' + TS][0] / D['tau_' + TS]))))     
        elif _inputs.sfh_type == 'delayed-exponential':
            D['sfr_norm_' + TS] = (
              1. / (((-D['tau_' + TS] * D['age_' + TS][-1] - D['tau_' + TS]**2.)
              * np.exp(- D['age_' + TS][-1] / D['tau_' + TS])) -
              ((-D['tau_' + TS] * D['age_' + TS][0] - D['tau_' + TS]**2.)
              * np.exp(- D['age_' + TS][0] / D['tau_' + TS]))))
    return D

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
    def __init__(self, sfh_type):

        self.sfh_type = sfh_type

        self.filter_1 = 'r'
        self.filter_2 = 'g'
        self.spec_lib = 'BASEL'
        self.isoc_lib = 'PADOVA'
        self.imf_type = 'Chabrier'
        self.Z = '0.0190'
        self.t_cutoff = 1.e9 * u.yr
        self.t_onset = 1.e8 * u.yr     
        self.tau_list = np.array(
          [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr
        self.subdir_fullpath = './'

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
        
        #Create function to copy fsps file where the generator can find.
        
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
            copy_fsps(sfh)
            _inputs = Input_Pars(sfh)
            _D = retrieve_data(_inputs)

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

                ax.text(0.1, y[-1] + 0.05, label[i], color='k', fontsize=fs)
                        
            #Clean up copied fsps folder.
            #shutil.rmtree('./fsps_FILES/')

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
    Plot_sSNRL(show_fig=True, save_fig=True)
 
