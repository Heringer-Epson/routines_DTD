#!/usr/bin/env python

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 28.
mks = 10.

#c = ['#7fc97f','#beaed4','#fdc086']
host2c = {'M12': '#7fc97f', 'H17':'#beaed4', 'S18':'#fdc086'}
host2os = {'M12': -0.02, 'H17':0., 'S18':+0.02}
type2m = {'SNIa':'o', 'zSNIa':'^'}

def add_points(_ax, matching, photo_cut, ctrl, host):
   
    run_dir = (
      'hosts/' + ctrl + '_' + host + '_' + matching + '_' + photo_cut + '/')
    
    fpath = ('./../OUTPUT_FILES/RUNS/' + run_dir + '/data_Dcolor.csv')
    
    df = pd.read_csv(fpath, header=0, low_memory=False)
    Nc = len(df['ra'].values)

    #Retrieve data and transform RA such that it's around zero degrees.
    ra, dec = df['ra'].values, df['dec'].values
    ra[(ra > 200.)] = ra[(ra > 200.)] - 360.

    #Filter data.
    os = host2os[host] #Offset ra and dec for clarity.
       
    #Plot spect. conf SN Ia.
    cond = ((df['is_host'].values == True)
            & (df['Classification'].values == 'SNIa'))
    _ax.plot(
      ra[cond], dec[cond] + os, ls='None', marker=type2m['SNIa'], 
      markersize=mks, color=host2c[host])
    Ns = len(ra[cond])

    #print ctrl, host, df['IAUName'].values[cond]

    
    #Plot photo SN Ia (zSNIa).
    cond = ((df['is_host'].values == True)
            & (df['Classification'].values == 'zSNIa'))
    _ax.plot(
      ra[cond], dec[cond] + os, ls='None', marker=type2m['zSNIa'], 
      markersize=mks, color=host2c[host])
    Np = len(ra[cond])
    
    return str(Nc), str(Ns), str(Np)

class Plot_Masses(object):
    """
    Description:
    ------------
    TBW.

    Parameters:
    -----------
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.

    Notes:
    ------
    Help to how to add two legends:
    https://riptutorial.com/matplotlib/example/32429/multiple-legends-on-the-same-axes    

    Outputs:
    --------
    ./../OUTPUT_FILES/ANALYSES_FIGURES/Fig_hosts.pdf
    
    """        
    def __init__(self, matching, photo_cut, show_fig, save_fig):
        self.matching = matching
        self.photo_cut = photo_cut
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.fig = plt.figure(figsize=(16,10))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
                
        self.df = None
                
        self.make_plot()
        
    def set_fig_frame(self):
    
        plt.subplots_adjust(wspace=.5)

        xlabel = r'RA [$\degree$]'
        ylabel = r'DEC [$\degree$]'

        #Axis 1
        self.ax1.set_xlabel(xlabel, fontsize=fs)
        self.ax1.set_ylabel(ylabel, fontsize=fs)
        self.ax1.set_xlim(-60., 60)
        self.ax1.set_ylim(-1.5, 2.)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax1.tick_params('both', length=12, width=2., which='major',
                             direction='in', right=True, top=True)
        self.ax1.tick_params('both', length=6, width=2., which='minor',
                             direction='in', right=True, top=True) 
        self.ax1.xaxis.set_minor_locator(MultipleLocator(10.))
        self.ax1.xaxis.set_major_locator(MultipleLocator(20.))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.25))
        self.ax1.yaxis.set_major_locator(MultipleLocator(.5))  
        
        #Add boundary line in dec.
        self.ax1.axhline(-1.25, ls='--', lw=2., color='gray')
        self.ax1.axhline(1.25, ls='--', lw=2., color='gray')

        #Axis 2
        self.ax2.set_xlabel(xlabel, fontsize=fs)
        self.ax2.set_xlim(-60., 60)
        self.ax2.set_ylim(-1.5, 2.)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax2.tick_params('both', length=12, width=2., which='major',
                             direction='in', right=True, top=True)
        self.ax2.tick_params('both', length=6, width=2., which='minor',
                             direction='in', right=True, top=True) 
        self.ax2.xaxis.set_minor_locator(MultipleLocator(10.))
        self.ax2.xaxis.set_major_locator(MultipleLocator(20.))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(.25))
        self.ax2.yaxis.set_major_locator(MultipleLocator(.5))  
        self.ax2.set_yticklabels([])
        
        #Add boundary line in dec.
        self.ax2.axhline(-1.25, ls='--', lw=2., color='gray')
        self.ax2.axhline(1.25, ls='--', lw=2., color='gray')
        
    def plot_quantities(self):
        
        #M12 control sample.
        for host in ['M12', 'H17', 'S18']:
            Nc, Ns, Np = add_points(
              self.ax1, self.matching, self.photo_cut, 'M12', host)
            self.ax1.plot(
              [np.nan],[np.nan],ls='-',lw=18.,c=host2c[host],
              label=r'Hosts from ' + host + ' (' + Ns + ',' + Np + ')')

        #Add Control sample text to bottom right corner.
        self.ax1.text(
          0.03, 0.03, 'Control sample: M12  (' + Nc + ')', fontsize=fs,
           horizontalalignment='left', transform=self.ax1.transAxes)
           
        #H17 control sample.
        for host in ['M12', 'H17', 'S18']:
            Nc, Ns, Np = add_points(
              self.ax2, self.matching, self.photo_cut, 'H17', host)
            self.ax2.plot(
              [np.nan],[np.nan],ls='-',lw=18.,c=host2c[host],
              label=r'Hosts from ' + host + ' (' + Ns + ',' + Np + ')')

        #Add Control sample text to bottom right corner.
        self.ax2.text(
          0.03, 0.03, 'Control sample: H17  (' + Nc + ')', fontsize=fs,
           horizontalalignment='left', transform=self.ax2.transAxes)

    def make_legend(self):

        for _ax in [self.ax1, self.ax2]:

            #Make entries for SN type legend.
            aux_SNIa, = _ax.plot(
              [np.nan],[np.nan],ls='None', marker=type2m['SNIa'], color='k',
              markersize=mks)
            aux_zSNIa, = _ax.plot(
              [np.nan],[np.nan],ls='None', marker=type2m['zSNIa'], color='k',
              markersize=mks)
            
            #Add color legend.
            leg_aux = _ax.legend(
              frameon=False,fontsize=fs,numpoints=1,ncol=1,labelspacing=0.2,
              handlelength=0.1, handletextpad=1., loc=2)
            
            #Overwrite original legend with SN subtype legend.
            _ax.legend(
              [aux_SNIa,aux_zSNIa], ['SNIa','zSNIa'], frameon=False, fontsize=fs,
              numpoints=1, ncol=1, labelspacing=0.2, handletextpad=-.2, loc=1)
            
            #Re-add original color legend.
            _ax.add_artist(leg_aux)

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fname = 'Fig_' + self.matching + '_' + self.photo_cut + '.pdf'
            fpath = './../OUTPUT_FILES/ANALYSES_FIGURES/hosts/' + fname
            plt.savefig(fpath, format='pdf')
        if self.show_fig:
            plt.show()
        plt.close(self.fig)    

    def make_plot(self):
        self.set_fig_frame()
        self.plot_quantities()
        self.make_legend()
        self.manage_output()             

if __name__ == '__main__':

    Plot_Masses(matching='Table', photo_cut='PC', show_fig=False, save_fig=True)
    Plot_Masses(matching='View', photo_cut='PC', show_fig=False, save_fig=True)
    Plot_Masses(matching='Table', photo_cut='noPC', show_fig=False, save_fig=True)
    Plot_Masses(matching='View', photo_cut='noPC', show_fig=False, save_fig=True)
