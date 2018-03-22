#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
                                                
taus = [1., 3., 7., 10.]

class Plot_AgeDcolor(object):
    
    def __init__(self, show_fig=True, save_fig=False):
        """Makes a figure where the color with respect to the red sequence (RS)
        (defined as color at 10 gyr) is plotted against age.
        """

        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
        self.fs = 20.   
        
        self.make_plot()
        
    def set_fig_frame(self):
        
        x_label = r'$\rm{log\ age\ [yr]}$'
        y_label = r'$\Delta (g-r)$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs)
        self.ax.set_ylabel(y_label,fontsize=self.fs)
        self.ax.set_xlim(6.,10.2)      
        self.ax.set_ylim(-1.4,0.1)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_on()
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.xaxis.set_major_locator(MultipleLocator(1.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        self.ax.yaxis.set_major_locator(MultipleLocator(0.4))  
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')

    def plot_data(self):
        directory = './../INPUT_FILES/STELLAR_POP/'
        dashes = [(4,4), (1,5), (4,2,1,2), (5,2,20,2)]
        colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
        
        #Get SSP data to compute RS color and for plotting.
        fpath = directory + 'SSP.dat'
        logage_SSP, sdss_g_SSP, sdss_r_SSP = np.loadtxt(
        fpath, delimiter=',', skiprows=1, usecols=(0, 4, 5), unpack=True)        

        RS_condition = (logage_SSP == 10.0)
        RS_color = sdss_g_SSP[RS_condition] - sdss_r_SSP[RS_condition]
        Dcolor_SSP = sdss_g_SSP - sdss_r_SSP - RS_color

        self.ax.plot(
          logage_SSP, Dcolor_SSP, color='k', ls='-', lw=3., label=r'SSP')

        for i, tau in enumerate(taus):
            fpath = directory + 'exponential_tau-' + str(tau) + '.dat'
        
            logage_exp, sdss_g_exp, sdss_r_exp = np.loadtxt(
            fpath, delimiter=',', skiprows=1, usecols=(0, 4, 5), unpack=True)
            Dcolor_exp = sdss_g_exp - sdss_r_exp - RS_color
            self.ax.plot(
              logage_exp, Dcolor_exp, color=colors[i], dashes=dashes[i], lw=3.,
              label=r'$\tau = ' + str(int(tau)) + '\ \mathrm{Gyr}$')

        self.ax.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1,
                       labelspacing=-0.2, loc=2)
                  
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_Age-Dcolor' + '.' + extension,
                        format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def make_plot(self):
        self.set_fig_frame()
        self.plot_data()
        plt.tight_layout()
        self.save_figure()
        self.show_figure()  

if __name__ == '__main__': 
    Plot_AgeDcolor(show_fig=True, save_fig=True)
