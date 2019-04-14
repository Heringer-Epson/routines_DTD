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

s2c = {'H17-sSNRL':'#377eb8', 'M12-sSNRL':'#4daf4a', 'M12-vespa':'#e41a1c'}
fs = 24.

def gpth(ctrl_samp, SN_samp, SN_type, z, method):
    return './../OUTPUT_FILES/RUNS/datasets/' + ctrl_samp + '_' + SN_samp + '_'\
       + SN_type + '_' + z + '/likelihoods/' + method + '_s1_s2.csv'
def pars2rec(ctrl_samp, SN_samp, SN_type, z):
    return './../OUTPUT_FILES/RUNS/datasets/' + ctrl_samp + '_' + SN_samp + '_'\
       + SN_type + '_' + z + '/record.dat'
def read_samplesize(fpath_rec):
    with open(fpath_rec, 'r') as rec:
        [rec.readline() for i in range(59)]
        N_all = rec.readline().split('= ')[-1].strip('\n')
        N_host = rec.readline().split('= ')[-1].strip('\n')
        return N_host, N_all

def draw(_ax,_fpath,_c):
    N_obs, s1, s2, A, ln_L = stats.read_lnL(_fpath)
    x, y, z = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
    nx, ny = len(np.unique(x)), len(np.unique(y))
    stats.plot_contour(_ax, np.log10(x), y, z,_c, nx, ny, add_max=True)

    
class Make_Fig(object):
    """
    Description:
    ------------
    Makes Fig. 8 in the DTD paper (appendix figure).
    Shows how the confidence contours compare between the color-luminosity
    and star formation history reconstruction methods.
    Datasamples from H17 and M12 are used.
  
    Parameters:
    -----------
    z : ~str
        Upper redshift cut for data inclusion. Recommended is '0.2' or '0.4'.
        H17 uses z_max = 0.2 wihle M14 used z_max = 0.4. 
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_dataset_02.pdf
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_dataset_04.pdf

    References:
    -----------
    Maoz+ 2012 (M12): http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    Heringer+ 2017 (H17): http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    """         
  
    def __init__(self, z, show_fig, save_fig):
        
        self.z = z
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        fig = plt.figure(figsize=(12,8))
        self.ax1 = fig.add_subplot(221)
        self.ax2 = fig.add_subplot(222)
        self.ax3 = fig.add_subplot(223)
        self.ax4 = fig.add_subplot(224)
        self.run_plot()

    def set_fig_frame(self):
        
        plt.subplots_adjust(left=0.125, bottom=0.125, right=0.9, top=0.9,
                            wspace=0.05, hspace=0.05)
        x_label = r'$\mathrm{log}\, A\,\,\,\, \mathrm{[yr^{-1}\ M_\odot^{-1}]}$'
        y_label = r'$s$'
        self.ax1.set_ylabel(y_label, fontsize=fs)
        self.ax3.set_xlabel(x_label, fontsize=fs)
        self.ax3.set_ylabel(y_label, fontsize=fs)        
        self.ax4.set_xlabel(x_label, fontsize=fs)

        self.ax1.text(
          0.5, 1.1, 'Hosts: as original', ha='center', va='center',
          transform=self.ax1.transAxes, fontsize=fs)        
        self.ax2.text(
          0.5, 1.1, 'Hosts: S18', ha='center', va='center',
          transform=self.ax2.transAxes, fontsize=fs) 
        self.ax2.text(
          1.1, 0.5, 'SNIa',  ha='center', va='center', rotation=-90,
          transform=self.ax2.transAxes, fontsize=fs)
        self.ax4.text(
          1.1, 0.5, 'zSNIa',  ha='center', va='center', rotation=-90,
          transform=self.ax4.transAxes, fontsize=fs)

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
            ax.tick_params(axis='x', which='major', labelsize=fs, pad=16)
            ax.tick_params(
              'both', length=10, width=2., which='major', direction='in')
            ax.tick_params(
              'both', length=5, width=2., which='minor', direction='in')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')  

            ax.set_xlim(-13., -11.8)
            ax.set_ylim(-2.1, -0.5)
            ax.xaxis.set_minor_locator(MultipleLocator(.1))
            ax.xaxis.set_major_locator(MultipleLocator(.5))
            ax.yaxis.set_minor_locator(MultipleLocator(.1))
            ax.yaxis.set_major_locator(MultipleLocator(.5))

        #Format labels.
        self.ax1.set_xticklabels([])
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax4.set_yticklabels([])
        
    def iterate_runs(self):

        draw(self.ax1,gpth('H17','H17','SNIa',self.z,'sSNRL'),s2c['H17-sSNRL'])
        draw(self.ax1,gpth('M12','M12','SNIa',self.z,'sSNRL'),s2c['M12-sSNRL'])
        draw(self.ax1,gpth('M12','M12','SNIa',self.z,'vespa'),s2c['M12-vespa'])


        draw(self.ax2,gpth('H17','S18','SNIa',self.z,'sSNRL'),s2c['H17-sSNRL'])
        draw(self.ax2,gpth('M12','S18','SNIa',self.z,'sSNRL'),s2c['M12-sSNRL'])
        draw(self.ax2,gpth('M12','S18','SNIa',self.z,'vespa'),s2c['M12-vespa'])

        #H17 does not have a native list with zSNIa.
        draw(self.ax3,gpth('M12','M12','zSNIa',self.z,'sSNRL'),s2c['M12-sSNRL'])
        draw(self.ax3,gpth('M12','M12','zSNIa',self.z,'vespa'),s2c['M12-vespa'])

        draw(self.ax4,gpth('H17','S18','zSNIa',self.z,'sSNRL'),s2c['H17-sSNRL'])
        draw(self.ax4,gpth('M12','S18','zSNIa',self.z,'sSNRL'),s2c['M12-sSNRL'])
        draw(self.ax4,gpth('M12','S18','zSNIa',self.z,'vespa'),s2c['M12-vespa'])

    def add_sample_sizes(self):
        
        x1, x2, y = 0.05, 0.52, 0.91
        N_h, N_a = read_samplesize(pars2rec('M12','M12','SNIa',self.z))
        self.ax1.text(
          x1, y, r'M12 (' + N_h + ',' + N_a + ')',
          ha='left', va='center', transform=self.ax1.transAxes, fontsize=fs-4)
        N_h, N_a = read_samplesize(pars2rec('H17','H17','SNIa',self.z))
        self.ax1.text(
          x2, y, r'H17 (' + N_h + ',' + N_a + ')',
          ha='left', va='center', transform=self.ax1.transAxes, fontsize=fs-4)

        N_h, N_a = read_samplesize(pars2rec('M12','S18','SNIa',self.z))
        self.ax2.text(
          x1, y, r'M12 (' + N_h + ',' + N_a + ')',
          ha='left', va='center', transform=self.ax2.transAxes, fontsize=fs-4)
        N_h, N_a = read_samplesize(pars2rec('H17','S18','SNIa',self.z))
        self.ax2.text(
          x2, y, r'H17 (' + N_h + ',' + N_a + ')',
          ha='left', va='center', transform=self.ax2.transAxes, fontsize=fs-4)       

        N_h, N_a = read_samplesize(pars2rec('M12','M12','zSNIa',self.z))
        self.ax3.text(
          x1, y, r'M12 (' + N_h + ',' + N_a + ')',
          ha='left', va='center', transform=self.ax3.transAxes, fontsize=fs-4)

        N_h, N_a = read_samplesize(pars2rec('M12','S18','zSNIa',self.z))
        self.ax4.text(
          x1, y, r'M12 (' + N_h + ',' + N_a + ')',
          ha='left', va='center', transform=self.ax4.transAxes, fontsize=fs-4)
        N_h, N_a = read_samplesize(pars2rec('H17','S18','zSNIa',self.z))
        self.ax4.text(
          x2, y, r'H17 (' + N_h + ',' + N_a + ')',
          ha='left', va='center', transform=self.ax4.transAxes, fontsize=fs-4)  

    def make_legend(self):
        self.ax2.plot(
          [np.nan], [np.nan], color=s2c['M12-vespa'], ls='-', lw=15., alpha=1.,
          marker='None', label=r'M12-SFHR')
        self.ax2.plot(
          [np.nan], [np.nan], color=s2c['M12-sSNRL'], ls='-', lw=15., alpha=1.,
          marker='None', label=r'M12-CL')
        self.ax2.plot(
          [np.nan], [np.nan], color=s2c['H17-sSNRL'], ls='-', lw=15., alpha=1.,
          marker='None', label=r'H17-CL')
        self.ax2.legend(
          frameon=False, fontsize=fs-4, numpoints=1, ncol=1, labelspacing=.0,
          handlelength=1.5, handletextpad=.8, loc=4, bbox_to_anchor=(.6, -0.05)) 

    def manage_output(self):
        self.ax1.set_rasterization_zorder(1)
        self.ax2.set_rasterization_zorder(1)
        self.ax3.set_rasterization_zorder(1)
        self.ax4.set_rasterization_zorder(1)
        if self.save_fig:
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/Fig_dataset_'\
            + self.z.replace('.', '') + '.pdf'
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.set_fig_frame()
        self.iterate_runs()
        self.add_sample_sizes()
        self.make_legend()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(z='0.2', show_fig=False, save_fig=True)
    Make_Fig(z='0.4', show_fig=False, save_fig=True)
