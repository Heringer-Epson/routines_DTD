#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from DTD_gen import make_DTD

def get_model_rates(s1, s2, t_onset, t_break, 
                    age, int_stellar_mass, int_formed_mass, r_band):
    """
    """
    out_sSNR, out_sSNRm, out_sSNRL = [], [], []
    
    #Compute the delay time distribution.
    DTD_func = make_DTD(s1, s2, t_onset, t_break)
    
    #Compute the amount of mass formed at each age of the SFH. This is the mass
    #that physically matters when computing the supernova rate via convolution.
    mass_formed = np.append(int_stellar_mass[0].to(u.Msun).value,
                            np.diff(int_formed_mass.to(u.Msun).value)) * u.Msun
    
    for i, t in enumerate(age):
    
        t_minus_tprime = t - age
        integration_cond = ((age >= t_onset) & (age <= t))
        
        DTD_ages = t_minus_tprime[integration_cond]
        DTD_component = DTD_func(DTD_ages)
        mass_component = mass_formed[integration_cond]
        
        #Perform convolution. This will give the expected SNe / yr for the
        #population used as input.  
        sSNR = np.sum(np.multiply(DTD_component,mass_component))
        
        #To compute the expected SN rate per unit mass, one then ahs to divide
        #by the 'burst total mass', which for a complex SFH (i.e. exponential)
        #corresponds to the integrated formed mass up to the age being assessed. 
        sSNRm = sSNR / int_formed_mass[i]
        
        #To compute the SN rate per unit of luminosity, one can simply take
        #the sSNR and divide by the L derived from the synthetic stellar pop.
        L = 10.**(-0.4*(r_band[i] - 5.)) * u.Lsun
        sSNRL = sSNR / L
        
        out_sSNR.append(sSNR), out_sSNRm.append(sSNRm), out_sSNRL.append(sSNRL)
    
    #Convert output lists to arrays preserving the units. 
    out_sSNR = np.array([rate.value for rate in out_sSNR]) * out_sSNR[0].unit
    out_sSNRm = np.array([rate.value for rate in out_sSNRm]) * out_sSNRm[0].unit
    out_sSNRL = np.array([rate.value for rate in out_sSNRL]) * out_sSNRL[0].unit

    return out_sSNR, out_sSNRm, out_sSNRL       
         

class Plot_sSNRm(object):
    
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
        
        x_label = r'$\rm{log\ age\ [yr]}$'
        y_label = r'$\rm{log\ sSNR}_m\ \rm{[M_\odot^{-1}\ yr^{-1}]}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs)
        self.ax.set_ylabel(y_label,fontsize=self.fs)
        self.ax.set_xlim(7.5,10.2)      
        self.ax.set_ylim(-14.,-11.)
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

        inputs = [(-0.50, -0.50, 1.e8 * u.yr, 1.e9 * u.yr),
                  (-0.50, -1.00, 1.e8 * u.yr, 1.e9 * u.yr),
                  (-1.00, -1.00, 1.e8 * u.yr, 1.e9 * u.yr),
                  (-1.25, -1.25, 1.e8 * u.yr, 1.e9 * u.yr),
                  (-1.00, -2.00, 1.e8 * u.yr, 1.e9 * u.yr),
                  (-1.00, -3.00, 1.e8 * u.yr, 1.e9 * u.yr)]

        age_array = np.logspace(6., 10.2, 1000) * u.yr
        dashes = [(), (4,4), (4, 2, 1, 2, 1, 2), (4,2,1,2), (1,1), (3,2,3,2)]
        colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f']
        labels = [r'-0.5', r'-0.5/-1', r'-1', r'-1.25', r'-1/-2', r'-1/-3']
        
        for i, inp in enumerate(inputs):
            s1, s2, t_onset, t_break = inp
            DTD_func = make_DTD(s1, s2, t_onset, t_break)
            self.ax.plot(
              np.log10(age_array.value), np.log10(DTD_func(age_array).value),
              color=colors[i], dashes=dashes[i], lw=3., label=labels[i])

        self.ax.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1,
                       labelspacing=0.3, loc=1)
                  
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_Age-sSNRm' + '.' + extension,
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
    """If directly executed, then plot the selected SN rates.
    """

    #Get SSP data for test plot.
    directory = './../INPUT_FILES/STELLAR_POP/'
    #fpath = directory + 'SSP.dat'
    fpath = directory + 'exponential_tau-1.0.dat'
    logage, int_stellar_mass, int_formed_mass, sdss_r= np.loadtxt(
    fpath, delimiter=',', skiprows=1, usecols=(0,2,3,6), unpack=True)   
    
    age = 10.**logage * u.yr
    int_stellar_mass *= u.Msun
    int_formed_mass *= u.Msun
    
    a, b, c = get_model_rates(-1., -2., 1.e8 * u.yr, 1.e9 * u.yr,
                              age, int_stellar_mass, int_formed_mass, sdss_r)

    #Plot_sSNRm(show_fig=True, save_fig=False)
 
