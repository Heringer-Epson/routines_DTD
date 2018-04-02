#!/usr/bin/env python

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from DTD_gen import make_DTD

t_onset = 1.e8 * u.yr
t_break = 1.e9 * u.yr

class Model_Rates(object):
    """This code is intended to compare the sSNR predicted using different
    integration techniques (relevant because the discontinuity in the DTD is
    relevant) and routine profiling.
    """
    
    def __init__(self, s1, s2, t_onset, t_break, tau):

        self.s1 = s1
        self.s2 = s2
        self.t_onset = t_onset
        self.t_break = t_break
        self.tau = tau
        
        self.age = None
        self.ages = None
        self.int_stellar_mass = None
        self.int_formed_mass = None
        self.g_band = None
        self.r_band = None
        self.mass_formed = None
        self.RS_color = None
        self.Dcolor = None
        self.DTD_func = None
        self.sSNR = None
        self.sSNRm = None
        self.sSNRL = None

        self.get_synpop_data()

    def get_synpop_data(self):
        """Read output data from FSPS runs. Besides the file specified by
        in self.synpop_fpath, always read data from a SSP run so that colors
        can be computed with respect to the red sequence. 
        """
        directory = './../INPUT_FILES/STELLAR_POP/'
        #Get SSP data.
        fpath = directory + 'SSP.dat'
        logage_SSP, sdss_g_SSP, sdss_r_SSP = np.loadtxt(
        fpath, delimiter=',', skiprows=1, usecols=(0,5,6), unpack=True)         

        #Retrieve RS color.
        RS_condition = (logage_SSP == 10.0)
        self.RS_color = sdss_g_SSP[RS_condition] - sdss_r_SSP[RS_condition]
        
        #Get data for the complex SFH (i.e. exponential.)
        tau_suffix = str(self.tau.to(u.yr).value / 1.e9)
        fpath = directory + 'exponential_tau-' + tau_suffix + '.dat'
        logage, sfr, int_stellar_mass, int_formed_mass, g_band, r_band = np.loadtxt(
        fpath, delimiter=',', skiprows=1, usecols=(0,1,2,3,5,6), unpack=True)   
        
        self.age = 10.**logage * u.yr
        self.sfr = sfr
        self.int_formed_mass = int_formed_mass
        self.g_band = g_band
        self.r_band = r_band
        
        self.Dcolor = self.g_band - self.r_band - self.RS_color               

    def compute_analytical_sfr(self, tau, upper_lim):
        _tau = tau.to(u.yr).value
        _upper_lim = upper_lim.to(u.yr).value
        norm = 1. / (_tau * (1. - np.exp(-_upper_lim / _tau)))
        def sfr_func(age):
            return norm * np.exp(-age / _tau)
        return sfr_func

    def make_sfr_func(self, _t, _sfr):
        interp_func = interp1d(_t, _sfr)
        def sfr_func(age):
            if age <= _t[0]:
                return _sfr[0]
            elif age > _t[0] and age < _t[-1]:
                return interp_func(age)
            elif age >= _t[-1]:
                return _sfr[-1]
        return np.vectorize(sfr_func)

    def convolve_functions(self, func1, func2, x):
        def out_func(xprime):
            return func1(x - xprime) * func2(xprime)
        return out_func

    @profile
    def compute_model_rates_simple(self):

        sSNR = []
        min_age = self.age[0].to(u.yr).value
        max_age = self.age[-1].to(u.yr).value
        ages = np.logspace(np.log10(min_age), np.log10(max_age), 1000)
        self.ages = ages
        
        self.DTD_func = make_DTD(self.s1, self.s2, self.t_onset, self.t_break)
        self.sfr_func = self.compute_analytical_sfr(self.tau, max_age * u.yr)
        
        _t_ons = self.t_onset.to(u.yr).value
        _t_bre = self.t_break.to(u.yr).value

        for i, t in enumerate(ages):

            cond = ((ages >= _t_ons) & (ages <= t))
            tprime = ages[cond]
            t_minus_tprime = t - tprime
            dt = np.append(np.array([0.]), np.diff(tprime))
            
            sfh = self.sfr_func(t_minus_tprime)
            DTD = self.DTD_func(tprime)
            
            _sSNR = np.multiply(sfh,DTD)
            _sSNR = np.sum(np.multiply(_sSNR,dt))

            sSNR.append(_sSNR)
                
        #Convert output lists to arrays preserving the units. 
        self.sSNR = np.array(sSNR)

        
    @profile
    def compute_model_rates(self):
        """
        """
        sSNR = []
        
        self.DTD_func = make_DTD(self.s1, self.s2, self.t_onset, self.t_break)
        self.sfr_func = self.compute_analytical_sfr(self.tau, self.age[-1])
        #self.sfr_func = self.make_sfr_func(self.age.value, self.sfr)
        
        for i, t in enumerate(self.age.to(u.yr).value):
            
            t0 = self.age.to(u.yr).value[0]
            self.conv_func = self.convolve_functions(self.sfr_func, self.DTD_func, t)
            #self.conv_func = self.convolve_functions(self.DTD_func, self.sfr_func, t)
     
            #Since the DTD is discontinuous at t_onset, on has to be careful
            #with the integration and do it piece-wise. Otherwise an artificial
            #'saw-like' pattern disturbs the predictions. For instance,
            #computing it as _sSNR = quad(self.conv_func, t0, t)[0] is not
            #recommended.
            _t_ons = self.t_onset.to(u.yr).value
            _t_bre = self.t_break.to(u.yr).value
            
            if t >= _t_ons:
                _sSNR = quad(self.conv_func, _t_ons, t)[0]
            else:
                _sSNR = 0.
            
            sSNR.append(_sSNR)
        
        self.sSNR = np.array(sSNR)

    @profile
    def compute_model_rates_parts(self):

        sSNR = []
        
        self.DTD_func = make_DTD(self.s1, self.s2, self.t_onset, self.t_break)
        self.sfr_func = self.compute_analytical_sfr(self.tau, self.age[-1])
        #self.sfr_func = self.make_sfr_func(self.age.value, self.sfr)
        
        for i, t in enumerate(self.age.to(u.yr).value):
            
            t0 = self.age.to(u.yr).value[0]
            self.conv_func = self.convolve_functions(self.sfr_func, self.DTD_func, t)
            #self.conv_func = self.convolve_functions(self.DTD_func, self.sfr_func, t)
     
            #Since the DTD is discontinuous at t_onset, on has to be careful
            #with the integration and do it piece-wise. Otherwise an artificial
            #'saw-like' pattern disturbs the predictions. For instance,
            #computing it as _sSNR = quad(self.conv_func, t0, t)[0] is not
            #recommended.
            _t_ons = self.t_onset.to(u.yr).value
            _t_bre = self.t_break.to(u.yr).value
          
            if t < _t_ons:
                v1 = quad(self.conv_func, t0, t)[0]
                v2 = 0.
                v3 = 0.
            elif t >= _t_ons and t < _t_bre:
                v1 = quad(self.conv_func, t0, _t_ons)[0]
                v2 = quad(self.conv_func, _t_ons, t)[0]
                v3 = 0.            
            elif t >= _t_bre:
                v1 = quad(self.conv_func, t0, _t_ons)[0]
                v2 = quad(self.conv_func, _t_ons, _t_bre)[0]
                v3 = quad(self.conv_func, _t_bre, t)[0]  

            _sSNR = v1 + v2 + v3

            sSNR.append(_sSNR)
        
        self.sSNR = np.array(sSNR)

    @profile
    def compute_model_rates_invconv(self):

        sSNR = []
        
        self.DTD_func = make_DTD(self.s1, self.s2, self.t_onset, self.t_break)
        self.sfr_func = self.compute_analytical_sfr(self.tau, self.age[-1])
        #self.sfr_func = self.make_sfr_func(self.age.value, self.sfr)
        
        for i, t in enumerate(self.age.to(u.yr).value):
            
            t0 = self.age.to(u.yr).value[0]
            self.conv_func = self.convolve_functions(self.DTD_func, self.sfr_func, t)
     
            #Since the DTD is discontinuous at t_onset, on has to be careful
            #with the integration and do it piece-wise. Otherwise an artificial
            #'saw-like' pattern disturbs the predictions. For instance,
            #computing it as _sSNR = quad(self.conv_func, t0, t)[0] is not
            #recommended.
            _t_ons = self.t_onset.to(u.yr).value
            _t_bre = self.t_break.to(u.yr).value
          
            if t < _t_ons:
                v1 = quad(self.conv_func, t0, t)[0]
                v2 = 0.
                v3 = 0.
            elif t >= _t_ons and t < _t_bre:
                v1 = quad(self.conv_func, t0, _t_ons)[0]
                v2 = quad(self.conv_func, _t_ons, t)[0]
                v3 = 0.            
            elif t >= _t_bre:
                v1 = quad(self.conv_func, t0, _t_ons)[0]
                v2 = quad(self.conv_func, _t_ons, _t_bre)[0]
                v3 = quad(self.conv_func, _t_bre, t)[0]  

            _sSNR = v1 + v2 + v3

            sSNR.append(_sSNR)
        
        self.sSNR = np.array(sSNR)

class Plot_Test(object):
    
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
        
        x_label = r'log age [yr]'
        y_label = r'$\rm{log\ sSNR}\ \rm{[yr^{-1}}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs)
        self.ax.set_ylabel(y_label,fontsize=self.fs)
        self.ax.set_xlim(7.,10.5)      
        self.ax.set_ylim(-14.5,-11.5)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_on()
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.xaxis.set_major_locator(MultipleLocator(1.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.yaxis.set_major_locator(MultipleLocator(0.5))  
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')

    def plot_models(self):

        for s1, s2 in zip([-1.], [-3.]):
            
            x_10Gyr, y_10Gyr = [], []
        
            for tau in [1., 1.5, 2., 3., 4., 5., 7., 10.]:
                
                fname = 'exponential_tau-' + str(tau) + '.dat'
                
                model = Model_Rates(s1, s2, t_onset, t_break,
                                    tau * 1.e9 * u.yr)

                #Compute model simple case where integration is done directly
                #via numpy arrays multiplication.
                model.compute_model_rates_simple()
                t = model.ages
                age_cond = (t <= 1.e10)
                
                x = t[age_cond]
                sSNR = model.sSNR[age_cond]
                sSNR[sSNR == 0.] = 1.e-40
                y = np.log10(sSNR)
            
                self.ax.plot(np.log10(x), y, color='r', ls='-', lw=1.5)
                
                #Compute default model where integration is done via scipy quad
                #and the convolution is computed as SFH(t-t`)xDTD(t`).
                model.compute_model_rates()
                t = model.age.value
                age_cond = (t <= 1.e10)
                
                x = t[age_cond]
                sSNR = model.sSNR[age_cond]
                sSNR[sSNR == 0.] = 1.e-40
                y = np.log10(sSNR)
            
                self.ax.plot(np.log10(x), y, color='b', ls='-', lw=1.5)                                

                #Compute default model where integration is done via scipy quad.
                model.compute_model_rates_parts()
                t = model.age.value
                age_cond = (t <= 1.e10)
                
                x = t[age_cond]
                sSNR = model.sSNR[age_cond]
                sSNR[sSNR == 0.] = 1.e-40
                y = np.log10(sSNR)
            
                self.ax.plot(np.log10(x), y, color='g', ls='-', lw=1.5)                                

                #Compute default model where integration is done via scipy quad
                #and the convolution is computed as SFH(t`)xDTD(t-t`).
                model.compute_model_rates_invconv()
                t = model.age.value
                age_cond = (t <= 1.e10)
                
                x = t[age_cond]
                sSNR = model.sSNR[age_cond]
                sSNR[sSNR == 0.] = 1.e-40
                y = np.log10(sSNR)
            
                self.ax.plot(np.log10(x), y, color='m', ls='-', lw=1.5)   


            self.ax.plot(np.nan, np.nan, color='r', ls='-', lw=1.5, label='Array multp. 1000 bins')
            self.ax.plot(np.nan, np.nan, color='b', ls='-', lw=1.5, label='Quad integrat.')
            self.ax.plot(np.nan, np.nan, color='g', ls='-', lw=1.5, label='Segmented integrat SFH(t-t`)xDTD(t`).')
            self.ax.plot(np.nan, np.nan, color='m', ls='-', lw=1.5, label='Segmented integrat of SFH(t`)xDTD(t-t`).')
            self.ax.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1,
                           labelspacing=-0.2, loc=2)            
                                  
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_sSNR_integration-test' + '.' + extension,
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
    Plot_Test(show_fig=True, save_fig=True)
