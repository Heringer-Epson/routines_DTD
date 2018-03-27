#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from SN_rate_gen import Model_Rates

t_onset = 1.e8 * u.yr
t_break = 1.e9 * u.yr

class Plot_sSNRL(object):
    
    def __init__(self, show_fig=True, save_fig=False):
        """Makes a figure where a set of DTDs is plotted as a function of age.
        """

        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
        self.fs = 20.   
        
        self.make_plot()
        
    def set_fig_frame(self):
        
        x_label = r'$\Delta (g-r)$'
        y_label = r'$\rm{log\ sSNR}_L\ \rm{[yr^{-1}\ L_\odot^{-1}]}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs)
        self.ax.set_ylabel(y_label,fontsize=self.fs)
        self.ax.set_xlim(-0.6,0.1)      
        self.ax.set_ylim(-14.5,-11.5)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_on()
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        self.ax.xaxis.set_major_locator(MultipleLocator(0.1))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.yaxis.set_major_locator(MultipleLocator(0.5))  
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')

    def plot_models(self):

        for s1, s2 in zip([-0.5, -0.5, -1., -1.25, -1., -1.],
                          [-0.5, -1., -1., -1.25, -2., -3.]):

        #for s1, s2 in zip([-1.], [-1.]):
            
            x_10Gyr, y_10Gyr = [], []
        
            for tau in [1., 1.5, 2., 3., 4., 5., 7., 10.]:
            #for tau in [1.]:
                fname = 'exponential_tau-' + str(tau) + '.dat'
                
                model = Model_Rates(s1, s2, t_onset, t_break, fname)

                age_cond = (model.age.to(u.yr).value <= 1.e10)
                
                x = model.Dcolor[age_cond]
                y = np.log10(model.sSNRL.value[age_cond])
            
                self.ax.plot(x, y, color='r', ls='-', lw=1.5)
                
                #Get values at 10 Gyr for interpolation.
                marker_cond = (model.age.to(u.yr).value == 1.e10)
                x_marker = model.Dcolor[marker_cond]
                y_marker = np.log10(model.sSNRL.value[marker_cond])
            
                x_10Gyr.append(x_marker), y_10Gyr.append(y_marker)
            
            self.ax.plot(x_10Gyr, y_10Gyr, color='b', ls='--', lw=1.5, marker='s')


                  
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_Dcolor-sSNRL' + '.' + extension,
                        format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def make_plot(self):
        self.set_fig_frame()
        self.plot_models()
        plt.tight_layout()
        self.save_figure()
        self.show_figure()  

if __name__ == '__main__':
    Plot_sSNRL(show_fig=True, save_fig=False)
 
