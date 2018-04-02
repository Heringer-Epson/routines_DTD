#!/usr/bin/env python

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy import units as u
from DTD_gen import make_DTD

class Model_Rates(object):
    
    def __init__(self, s1, s2, t_onset, t_break, tau):

        self.s1 = s1
        self.s2 = s2
        self.t_onset = t_onset
        self.t_break = t_break
        self.tau = tau
        
        self.age = None
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

        self.make_model()

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

    #@profile
    def compute_model_rates(self):
        """
        """
        sSNR, sSNRm, sSNRL, self.L = [], [], [], []

        _t_ons = self.t_onset.to(u.yr).value
        _t_bre = self.t_break.to(u.yr).value
    
        self.DTD_func = make_DTD(self.s1, self.s2, self.t_onset, self.t_break)
        self.sfr_func = self.compute_analytical_sfr(self.tau, self.age[-1])
        #self.sfr_func = self.make_sfr_func(self.age.value, self.sfr)
        
        for i, t in enumerate(self.age.to(u.yr).value):
            
            t0 = self.age.to(u.yr).value[0]
            self.conv_func = self.convolve_functions(self.sfr_func, self.DTD_func, t)
            #self.conv_func = self.convolve_functions(self.DTD_func, self.sfr_func, t)
     
            #Since the DTD is discontinuous at t_onset, on has to be careful
            #with the integration and do it piece-wise. Here, since the DTD is
            #zero prior to t_ons, we start the integration at t_ons.            
            _sSNR = quad(self.conv_func, _t_ons, t)[0] 
            
            #To compute the expected SN rate per unit mass, one then has to divide
            #by the 'burst total mass', which for a complex SFH (i.e. exponential)
            #corresponds to the integrated formed mass up to the age being assessed. 
            _sSNRm = _sSNR / self.int_formed_mass[i]
            
            #To compute the SN rate per unit of luminosity, one can simply take
            #the sSNR and divide by the L derived from the synthetic stellar pop.
            L = 10.**(-0.4 * (self.r_band[i] - 5.))
            _sSNRL = _sSNR / L
            
            sSNR.append(_sSNR), sSNRm.append(_sSNRm), sSNRL.append(_sSNRL), self.L.append(L)
        
        #Convert output lists to arrays preserving the units. 
        self.sSNR = np.array(sSNR)
        self.sSNRm = np.array(sSNRm)
        self.sSNRL = np.array(sSNRL)

    def make_model(self):
        self.get_synpop_data()
        self.compute_model_rates()

