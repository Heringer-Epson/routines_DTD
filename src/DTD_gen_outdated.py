#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u

def make_DTD(const_s1, s1, s2, t_onset, t_break):
    """Generates a DTD function of the format:
    
    SNR = 0. if t < t_onset (suffix 'a').
    SNR = const_s1 * t^s1 if t >= t_onset and t < t_break (suffix 'b')
    SNR = const_s2 * t^s2 if t >= t_break (suffix 'c').
    
    It is imposed that the DTD is continuous everywhere except at t_onset
    and that it is normalized at t_norm according to a literature value.
    """
    #Because np.concatenate does not seem to preserve units and because
    #some of the power-law calculations require taking .value of the age
    #quantities, convert all ages here to yr and remove units.
    t_onset = t_onset.to(u.yr).value
    t_break = t_break.to(u.yr).value
    const_s2 = const_s1 * t_break**(s1 - s2)
    def func_DTD(age):      
        _age = np.asarray(age)
        DTD = np.vectorize(lambda t: 1.e-40 if t < t_onset
                           else const_s1 * t**s1 if t <= t_break
                           else const_s2 * t**s2, otypes=[np.float64])
        return DTD(_age)    
    return func_DTD       

class Plot_DTDs(object):
    
    def __init__(self, show_fig=True, save_fig=False):
    """
    Description:
    ------------
    This class will plot the selected DTDs as a way to (visiually) verify
    that the created DTDs make sense..

    Outputs:
    --------
    ./../OUTPUT_FILES/RUNS/$RUN_DIR/Fig_Age-DTD.pdf'
    """  

        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
        self.fs = 20.   
        
        self.make_plot()
        
    def set_fig_frame(self):
        
        x_label = r'$\rm{log\ age\ [yr]}$'
        y_label = r'$\rm{log\ sSNR}_m\ \rm{[M_\odot^{-1}\ yr^{-1}]}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs)
        self.ax.set_ylabel(y_label,fontsize=self.fs)
        self.ax.set_xlim(7.5,10.2)      
        self.ax.set_ylim(-14.,-6.5)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_on()
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.xaxis.set_major_locator(MultipleLocator(1.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(1.))  
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')

    def plot_models(self):

        inputs = [(1.e-3, -0.50, -0.50, 1.e8 * u.yr, 1.e9 * u.yr),
                  (1.e-3, -0.50, -1.00, 1.e8 * u.yr, 1.e9 * u.yr),
                  (1.e-3, -1.00, -1.00, 1.e8 * u.yr, 1.e9 * u.yr),
                  (1.e-3, -1.25, -1.25, 1.e8 * u.yr, 1.e9 * u.yr),
                  (1.e-3, -1.00, -2.00, 1.e8 * u.yr, 1.e9 * u.yr),
                  (1.e-3, -1.00, -3.00, 1.e8 * u.yr, 1.e9 * u.yr)]
        
        age_array = np.logspace(6., 10.2, 100) * u.yr
        dashes = [(), (4,4), (4, 2, 1, 2, 1, 2), (4,2,1,2), (1,1), (3,2,3,2)]
        colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f']
        labels = [r'-0.5', r'-0.5/-1', r'-1', r'-1.25', r'-1/-2', r'-1/-3']
        
        for i, inp in enumerate(inputs):
            A, s1, s2, t_onset, t_break = inp
            DTD_func = make_DTD(A, s1, s2, t_onset, t_break)
            self.ax.plot(
              np.log10(age_array.value), np.log10(DTD_func(age_array)),
              color=colors[i], dashes=dashes[i], lw=3., label=labels[i])
        
        self.ax.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1,
                       labelspacing=0.3, loc='best')
                  
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_Age-DTD' + '.' + extension,
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
    """If directly executed, then plot the selected DTDs.
    """            
    Plot_DTDs(show_fig=True, save_fig=False)
 
