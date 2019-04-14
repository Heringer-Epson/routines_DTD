#!/usr/bin/env python

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import stats

sys.path.append(os.path.join(os.environ['PATH_ssnarl'], 'src'))
from Dcolor2sSNRL_gen import Generate_Curve
from build_fsps_model import Build_Fsps
from generic_input_pars import Generic_Pars

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 28.

class Plot_Masses(object):
    """
    Description:
    ------------
    Makes Fig. 7 in tge DTD paper. This routine plots (for comparison purposes)
    ranges of masses derived with the color-luminosity (CL) method against VESPA
    masses (these are integrated masses) and against an independent source
    (Chang+ 2015). The latter are current stellar masses (i.e. mass in dead
    stars are removed). These sources adopt different IMFs, which is accounted
    for when computing masses in the CL method. To derive CL masses, one has
    to assume a SFH. Because of this, we only select galaxies that are likely
    to belong to the red sequence (via a color cut and by imposing VESPA
    masses in the early bins to be null). We then assume a SFH history with
    1 Gyr timescale and an age of 10Gyr, where the errorbars are produced by
    assuming ages of 2.4 and 14 Gyr.

    Parameters:
    -----------
    tau : ~astropy float
        Timescale of the assumed exponential SFH. 1. * u.Gyr is suggested.
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.

    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_mass.pdf
    
    References:
    -----------
    Tojeiro+ 2007: http://adsabs.harvard.edu/abs/2007MNRAS.381.1252T (VESPA 1)
    Tojeiro+ 2009: http://adsabs.harvard.edu/abs/2009ApJS..185....1T (VESPA 2)
    Chang+ 2015: http://adsabs.harvard.edu/abs/2015ApJS..219....8C
    """        
    def __init__(self, tau, show_fig, save_fig):
        self.tau = tau
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.fig = plt.figure(figsize=(12,6))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        self.MsL, self.MtL = None, None
        self.Ms_fsps_at10, self.Ms_fsps_unc = None, None
        self.Ms_C15, self.Ms_C15_unc = None, None

        self.Mt_fsps_at10, self.Mt_fsps_unc = None, None
        self.Mt_vespa, self.Mt_vespa_unc = None, None
                
        self.make_plot()
        
    def set_fig_frame(self):
    
        xlabel = r'$M_{\ast}\ [\rm{M_\odot}]$ (Chang et al)'
        ylabel = r'$M_{\ast}\ [\rm{M_\odot}]$ (FSPS)'
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.set_xlabel(xlabel, fontsize=fs)
        self.ax1.set_ylabel(ylabel, fontsize=fs)
        self.ax1.set_xlim(1.e9, 1.e12)
        self.ax1.set_ylim(1.e9, 1.e12)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params('both', length=12, width=2., which='major',
                             direction='in', right=True, top=True)
        self.ax1.tick_params('both', length=6, width=2., which='minor',
                             direction='in', right=True, top=True) 
        self.ax1.plot([1.e9,1.e12], [1.e9,1.e12], marker='None', ls='--',
                     lw=3., color='m', zorder=2.)

        xlabel = r'$M_{\rm{T}}\ [\rm{M_\odot}]$ (VESPA)'
        ylabel = r'$M_{\rm{T}}\ [\rm{M_\odot}]$ (FSPS)'
        self.ax2.set_xscale('log')
        self.ax2.set_yscale('log')
        self.ax2.set_xlabel(xlabel, fontsize=fs)
        self.ax2.set_ylabel(ylabel, fontsize=fs)
        self.ax2.set_xlim(1.e9, 1.e12)
        self.ax2.set_ylim(1.e9, 1.e12)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params('both', length=12, width=2., which='major',
                             direction='in', right=True, top=True)
        self.ax2.tick_params('both', length=6, width=2., which='minor',
                             direction='in', right=True, top=True) 
        self.ax2.plot([1.e9,1.e12], [1.e9,1.e12], marker='None', ls='--',
                     lw=3., color='m', zorder=2.)

    def retrieve_data(self):
        fpath = './../OUTPUT_FILES/RUNS/M12_comp/data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)

        #Calculate binned masses as in M12.
        mass_corr = .55
        self.df['mass1'] = (self.df['vespa1'].values + self.df['vespa2'].values) * mass_corr
        self.df['mass2'] = self.df['vespa3'].values * mass_corr
        self.df['mass3'] = self.df['vespa4'].values * mass_corr
        self.df['mass'] = self.df['mass1'] + self.df['mass2'] + self.df['mass3']

        #Condition. To approach a fair comparison, select only vespa galaxies
        #which belong to the RS, only have mass in the old bin, exhibit low
        #redshift and small magnitude uncertainties..
        cond = (
          (self.df['Dcolor_gr'].values > -0.1) & (self.df['Dcolor_gr'].values < 0.1)
          & (self.df['z'].values > 0.) & (self.df['z'].values < 0.2)
          & (self.df['petroMagErr_g'].values < 0.02) & (self.df['petroMagErr_r'].values < 0.02)
          & (self.df['mass1'].values < 1.e6) & (self.df['mass2'].values < 1.e6))
             
          
        self.df = self.df[cond]

    def compute_Ms_to_light(self):
        """Compute the mass (stellar - present day) to light ratio using FSPS.
        These assume a Chabrier IMF so that it can be compared to Chang+ 2015.
        """
        _inputs = Generic_Pars('exponential', 'Chabrier', '0.0190', 40. * u.Myr)
        _D = Build_Fsps(_inputs).D
        TS = str(self.tau.to(u.yr).value / 1.e9)
        age = _D['age_' + TS] * 1.e9
        cond = ((age >= 2.4e9) & (age <= 14.e9))
                
        Lr_fsps = stats.mag2lum(_D['mag1_' + TS])
        ML_aux = np.divide(_D['int_stellar_mass_' + TS],Lr_fsps)
        ML_at10 = ML_aux[(age == 10.e9)][0]
        ML_l, ML_u = (ML_aux[cond])[0], (ML_aux[cond])[-1]
        self.MsL = (ML_at10,ML_l,ML_u) 
 
    def compute_Mt_to_light(self):
        """Compute the mass (integrated) to light ratio using FSPS.
        These assume a Kroupa IMF so that it can be compared to M12.
        """
        _inputs = Generic_Pars('exponential', 'Kroupa', '0.0190', 40. * u.Myr)
        _D = Build_Fsps(_inputs).D
        TS = str(self.tau.to(u.yr).value / 1.e9)
        age = _D['age_' + TS] * 1.e9
        cond = ((age >= 2.4e9) & (age <= 14.e9))
                
        Lr_fsps = stats.mag2lum(_D['mag1_' + TS])
        ML_aux = np.divide(_D['int_mass_' + TS],Lr_fsps)
        ML_at10 = ML_aux[(age == 10.e9)][0]
        ML_l, ML_u = (ML_aux[cond])[0], (ML_aux[cond])[-1]
        self.MtL = (ML_at10,ML_l,ML_u) 
        
    def get_masses(self):
        """Get independent mass estimates from Chang+ 2015.
        """
        fpath = './../INPUT_FILES/C15_masses/masses.dat'
        plate, mjd, fiber, lmass_025, lmass_50, lmass_975 = np.loadtxt(
          fpath,skiprows=66, delimiter='|', usecols=(4,5,6,9,11,13),
          dtype=str, unpack=True)
        
        
        Ms_fsps_at10, Ms_fsps_l, Ms_fsps_u = [], [], []
        Mt_fsps_at10, Mt_fsps_l, Mt_fsps_u = [], [], []
        M_vespa, M_C15, M_C15_l, M_C15_u = [], [], [], []
        for k, (plate_V, mjd_V, fiber_V, Mr, Dcolor, M_v) in enumerate(zip(
          self.df['plateID'].values, self.df['mjd'].values,
          self.df['fiberID'].values, self.df['abs_r'].values,
          self.df['Dcolor_gr'].values, self.df['mass'].values)):
            
            print k, len(self.df['plateID'].values)
                                               
            for i, (plate_C, mjd_C, fiber_C) in enumerate(zip(plate, mjd, fiber)):
                
                if (
                  (int(plate_C.strip()) == int(plate_V))
                  & (int(mjd_C.strip()) == int(mjd_V))
                  & (int(fiber_C.strip()) == int(fiber_V))
                  & (Dcolor > -0.05) & (Dcolor < 0.05)):

                    L = stats.mag2lum(Mr) 
                    
                    #Compute stellar mass in sSNRL.
                    Ms_fsps_at10.append(self.MsL[0] * L)
                    Ms_fsps_l.append(self.MsL[1] * L)
                    Ms_fsps_u.append(self.MsL[2] * L)

                    #Compute integrated mass in sSNRL.
                    Mt_fsps_at10.append(self.MtL[0] * L)
                    Mt_fsps_l.append(self.MtL[1] * L)
                    Mt_fsps_u.append(self.MtL[2] * L)
                    
                    #Append mass from Chang+ 2015.
                    M_C15.append(10.**(float(lmass_50[i])))
                    M_C15_l.append(10.**(float(lmass_025[i])))
                    M_C15_u.append(10.**(float(lmass_975[i])))
                    
                    #Append vespa masses.
                    M_vespa.append(M_v)
                    
        self.Ms_fsps_at10 = np.asarray(Ms_fsps_at10)
        self.Ms_fsps_unc = (np.array(Ms_fsps_at10) - np.array(Ms_fsps_l),
                           np.array(Ms_fsps_u) - np.array(Ms_fsps_at10))

        self.Mt_fsps_at10 = np.asarray(Mt_fsps_at10)
        self.Mt_fsps_unc = (np.array(Mt_fsps_at10) - np.array(Mt_fsps_l),
                           np.array(Mt_fsps_u) - np.array(Mt_fsps_at10))

        self.Ms_C15 = np.asarray(M_C15)
        self.Ms_C15_unc = (np.array(M_C15) - np.array(M_C15_l),
                          np.array(M_C15_u) - np.array(M_C15))
        
        self.Mt_vespa = np.asarray(M_vespa)
        
    def plot_quantities(self):
        self.ax1.errorbar(self.Ms_C15, self.Ms_fsps_at10, xerr=self.Ms_C15_unc,
        yerr=self.Ms_fsps_unc, capsize=0., color='k', ls='None',
        marker='d', markersize=6., elinewidth=0.5)

        self.ax2.errorbar(self.Mt_vespa, self.Mt_fsps_at10,
        yerr=self.Mt_fsps_unc, capsize=0., color='k', ls='None',
        marker='d', markersize=6., elinewidth=0.5)

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_mass.pdf'
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show()
        plt.close(self.fig)    

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.compute_Ms_to_light()
        self.compute_Mt_to_light()
        self.get_masses()
        self.plot_quantities()
        self.manage_output()             

if __name__ == '__main__':

    Plot_Masses(tau=1. * u.Gyr, show_fig=False, save_fig=True)

