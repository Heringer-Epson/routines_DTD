#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import stats

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fs = 24.
c = ['#1b9e77','#d95f02','#7570b3']

def fv(v):
    if isinstance(v,float):
        return str(format(v, '.2f'))
    else:
        return (str(format(v[0], '.2f')), str(format(v[1], '.2f')))

def draw(_ax1, _ax2, _fpath, _c):
    N_obs, s1, s2, A, ln_L = stats.read_lnL(_fpath)
    x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
    nx, ny = len(np.unique(x)), len(np.unique(y))
    if _ax1 is not None:
        X, Y, XErr, YErr = stats.plot_contour(
          _ax1, np.log10(x), y, z, _c, nx, ny, add_max=True)
        print '    ', fv(X), fv(Y), fv(XErr), fv(YErr)
    nx, ny = len(np.unique(s1)), len(np.unique(s2))
    stats.plot_contour(_ax2, s1, s2, ln_L, _c, nx, ny, add_max=True)
    
class Make_Fig(object):
    """
    Description:
    ------------
    Makes Fig. 4 of the DTD paper. This routine applies both the
    color-luminosity (CL) and the star formation recosntruction method to a 
    sample of galaxies and plots their respective confidence contours in the
    (A,s -- left panel) and (s1,s2 -- right panel) parameter spaces.
    The dataset of galaxies is retrieved from M12 (provided to us by D. Maoz),
    so that VESPA masses are available and uses photometry and redshifts cuts
    similar to H17, so that the CL method can be applied. Currently, this
    sample is produced through the 'M12_comp' option in input_params.py.
    The 68% uncertainty values are printed by this run, but not stored.
  
    Parameters:
    -----------
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_method_comp.pdf
    >> A, s, (A upper, A lower), (s upper, s lower)

    References:
    -----------
    Maoz+ 2012 (M12): http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    Heringer+ 2017 (H17): http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    """         
  
    def __init__(self, show_fig, save_fig):
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        fig = plt.figure(figsize=(20,8))
        self.ax1 = fig.add_subplot(131)
        self.ax2 = fig.add_subplot(132)
        self.ax3 = fig.add_subplot(133)
        self.run_plot()

    def set_fig_frame(self):
        
        xlabel = r'$\mathrm{log}\, A\,\,\,\, \mathrm{[yr^{-1}\ M_\odot^{-1}]}$'
        self.ax1.set_xlabel(xlabel, fontsize=fs + 4)
        self.ax1.set_ylabel(r'$s=s_1=s_2$', fontsize=fs + 4)
        self.ax1.set_xlim(-12.7,-11.9)
        self.ax1.set_ylim(-1.7,-0.9)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params('both', length=12, width=2., which='major',
                             direction='in', right=True, top=True)
        self.ax1.tick_params('both', length=6, width=2., which='minor',
                             direction='in', right=True, top=True) 
        self.ax1.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.xaxis.set_major_locator(MultipleLocator(.2))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.yaxis.set_major_locator(MultipleLocator(.2))  

        self.ax2.set_xlabel(r'$s_1$', fontsize=fs + 4)
        self.ax2.set_ylabel(r'$s_2$', fontsize=fs + 4)
        self.ax2.set_xlim(-3.,0.)
        self.ax2.set_ylim(-3.,0.)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params('both', length=12, width=2., which='major',
                                 direction='in', right=True, top=True)
        self.ax2.tick_params('both', length=6, width=2., which='minor',
                                 direction='in', right=True, top=True) 
        self.ax2.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax2.xaxis.set_major_locator(MultipleLocator(.5))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax2.yaxis.set_major_locator(MultipleLocator(.5))  
        self.ax2.plot([-3., 0.], [-3., 0.], ls='--', c='k', lw=1.)

        self.ax3.set_xlabel(r'$s_1$', fontsize=fs + 4)
        self.ax3.set_ylabel(r'$s_2$', fontsize=fs + 4)
        self.ax3.set_xlim(-3.,0.)
        self.ax3.set_ylim(-3.,0.)
        self.ax3.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax3.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax3.tick_params('both', length=12, width=2., which='major',
                                 direction='in', right=True, top=True)
        self.ax3.tick_params('both', length=6, width=2., which='minor',
                                 direction='in', right=True, top=True) 
        self.ax3.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax3.xaxis.set_major_locator(MultipleLocator(.5))
        self.ax3.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax3.yaxis.set_major_locator(MultipleLocator(.5)) 
        self.ax3.plot([-3., 0.], [-3., 0.], ls='--', c='k', lw=1.)

        plt.subplots_adjust(bottom=0.14, wspace=.3, left=0.075, right=.95)

    def add_contours(self):

        print 'Method: color-luminosity'
        fpath =  './../OUTPUT_FILES/RUNS/default/standard_40_042/likelihoods/sSNRL_s1_s2.csv'
        draw(None, self.ax2, fpath, c[0])
        print 'Method: SFH reconstruction'
        fpath =  './../OUTPUT_FILES/RUNS/default/standard_40_042/likelihoods/vespa_s1_s2.csv'
        draw(None, self.ax2, fpath, c[1])
        
        print 'Method: color-luminosity'
        fpath =  './../OUTPUT_FILES/RUNS/default/standard_40_24/likelihoods/sSNRL_s1_s2.csv'
        draw(self.ax1, self.ax3, fpath, c[0])
        print 'Method: SFH reconstruction'
        fpath =  './../OUTPUT_FILES/RUNS/default/standard_40_24/likelihoods/vespa_s1_s2.csv'
        draw(self.ax1, self.ax3, fpath, c[1])



    def add_legend(self):
        self.ax1.plot([np.nan], [np.nan], color=c[0], ls='-', lw=15., alpha=0.5,
                      marker='None', label=r'CL')
        self.ax1.plot([np.nan], [np.nan], color=c[1], ls='-', lw=15., alpha=0.5,
                      marker='None', label=r'SFHR')

        handles, labels = self.ax1.get_legend_handles_labels()                             
        self.ax1.legend(
          handles[::-1], labels[::-1], frameon=False, fontsize=fs, numpoints=1,
          ncol=1, loc=1)  

        self.ax2.plot([np.nan], [np.nan], color='w', label=r'$t_{\rm{c}}=0.42\,$Gyr')
        self.ax2.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1, loc=1)  

        self.ax3.plot([np.nan], [np.nan], color='w', label=r'$t_{\rm{c}}=2.4\,$Gyr')
        self.ax3.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1, loc=1)  

    def manage_output(self):
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_method_comp.pdf'
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.set_fig_frame()
        self.add_contours()
        self.add_legend()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(show_fig=False, save_fig=True)
