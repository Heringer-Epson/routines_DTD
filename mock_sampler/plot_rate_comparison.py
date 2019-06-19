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
import core_funcs

sys.path.append(os.path.join(os.environ['PATH_ssnarl'], 'src'))
from build_fsps_model import Build_Fsps
from Dcolor2sSNRL_gen import Generate_Curve

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 24.
c = ['#1b9e77','#d95f02','#7570b3']

def calculate_likelihood(mode, _inputs, _df, _D, _s1, _s2):
    if mode == 'sSNRL':
        Sgen = Generate_Curve(_inputs, _D, _s1, _s2)
        if _inputs.model_Drange == 'reduced':
            x, y = Sgen.Dcolor_at10Gyr[::-1], Sgen.sSNRL_at10Gyr[::-1]
        elif _inputs.model_Drange == 'extended':
            x, y = Sgen.Dcd_fine, Sgen.sSNRL_fine
        sSNRL = np.asarray(core_funcs.interp_nobound(x, y, _df['Dcolor']))
        rate = np.multiply(stats.mag2lum(_df['absmag']),sSNRL)
    return rate 

class Plot_Rates(object):
    """
    Code Description
    ----------    
    Given the known DTD, compare the rates predicted by the Cl and SFHR
    methods with the true one.

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

        self.D = Build_Fsps(self.inputs).D

        self.df, self.r_true, self.r_CL, self.r_SFHR = None, None, None, None
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.make_plot()

    def set_fig_frame(self):        
        x_label = r'$\rm{log}\,\, r_{\rm{true}}$'
        y_label = r'$\rm{log}\,\, r_{\rm{predicted}}$'
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
        
        self.ax.plot([-5.5,-0.5],[-5.5,-0.5], ls=':', lw=2., c='gray')      

    def retrieve_true_rates(self):
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/data_mock_final.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
        self.r_true = self.df['sSNR'].values

        m_fsps = self.df['vespa1'] + self.df['vespa2'] + self.df['vespa3'] + self.df['vespa4'] 
        self.df['m_fsps'] = m_fsps
        
    def compute_CL_rates(self):
        self.inputs.visibility_flag = False
        self.inputs.model_Drange = 'extended'
        m_fsps = self.df['vespa1'] + self.df['vespa2'] + self.df['vespa3'] + self.df['vespa4'] 
        abs_mag = (-2.5 * (self.df['logmass'].values - np.log10(m_fsps.values))
                   + self.df['petroMag_r'].values)

        CL_df = {
          'absmag': abs_mag, 'Dcolor': self.df['Dcolor_gr'].values,
          'z': self.df['z'], 'is_host': self.df['is_host'].values}        
           
        self.r_CL = self.A * calculate_likelihood(
          'sSNRL', self.inputs, CL_df, self.D, self.s1, self.s2)

    def compute_SFHR_rates(self):
        self.inputs.visibility_flag = False

        #m_factor = 10.**self.df['logmass'].values
        m_factor = np.divide(10.**self.df['logmass'].values,self.df['m_fsps'].values)
        
        SFHR_df = {
          'mass1': np.multiply(self.df['vespa2'].values,m_factor),
          'mass2': np.multiply(self.df['vespa3'].values,m_factor),
          'mass3': np.multiply(self.df['vespa4'].values,m_factor),
          'z': self.df['z'].values, 'is_host': self.df['is_host'].values} 
        
        psi1, psi2, psi3 = stats.binned_DTD_rate(
          self.s1, self.s2, self.D['t_ons'], self.D['t_bre'])
        
        self.r_SFHR = self.A * (
          np.multiply(psi1,SFHR_df['mass1']) + np.multiply(psi2,SFHR_df['mass2'])
          + np.multiply(psi3,SFHR_df['mass3']))

    def get_examples(self):
        ratio = np.divide(self.r_true,self.r_SFHR)
        cond = (ratio > 18.)
        print len(ratio[cond])
        print self.df['tau'].values[cond], self.df['logage'].values[cond], self.df['Dcolor_gr'].values[cond]
        #print [ratio[cond]]

    def plot_qtties(self):
        self.ax.plot(np.log10(self.r_true), np.log10(self.r_SFHR), ls='None',
                     marker='s', color=c[1], markersize=5.)

        self.ax.plot(np.log10(self.r_true), np.log10(self.r_CL), ls='None',
                     marker='s', color=c[0], markersize=5.)

        self.ax.plot([np.nan], [np.nan], color=c[0], ls='-', lw=15., marker='None',
                label=r'$\tt{CL}$')
        self.ax.plot([np.nan], [np.nan], color=c[1], ls='-', lw=15., marker='None',
                label=r'$\tt{SFHR}$')

        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, ncol=1,
          loc=2, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)   
        
    def manage_output(self):
        plt.tight_layout()
        if self.inputs.save_fig:
            fpath = './../OUTPUT_FILES/MOCK_SAMPLE/Fig_rates.pdf'
            plt.savefig(fpath, format='pdf')
        if self.inputs.show_fig:
            plt.show() 

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_true_rates()
        self.compute_CL_rates()
        self.compute_SFHR_rates()
        self.get_examples()
        self.plot_qtties()
        self.manage_output()

