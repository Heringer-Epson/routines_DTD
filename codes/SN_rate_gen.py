#!/usr/bin/env python

import numpy as np
from astropy import units as u
from DTD_gen import make_DTD

class Model_Rates(object):
    
    def __init__(self, s1, s2, t_onset, t_break, synpop_fname):

        self.s1 = s1
        self.s2 = s2
        self.t_onset = t_onset
        self.t_break = t_break
        self.synpop_fname = synpop_fname
        
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
        self.test = None

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
        fpath = directory + self.synpop_fname
        logage, int_stellar_mass, int_formed_mass, g_band, r_band = np.loadtxt(
        fpath, delimiter=',', skiprows=1, usecols=(0,2,3,5,6), unpack=True)   
        
        self.age = 10.**logage * u.yr
        self.int_stellar_mass = int_stellar_mass * u.Msun
        self.int_formed_mass = int_formed_mass * u.Msun
        self.g_band = g_band
        self.r_band = r_band
        
        self.Dcolor = self.g_band - self.r_band - self.RS_color
            
        #Compute the amount of mass formed at each age of the SFH. This is the
        #mass that physically matters when computing the SN rate via convolution.
        self.mass_formed = np.append(self.int_stellar_mass[0].to(u.Msun).value,
          np.diff(self.int_formed_mass.to(u.Msun).value)) * u.Msun
        
        #Perform check if mass formed sums up to 1.
        if np.sum(self.mass_formed.to(u.Msun).value) - 1. > 1.e6:
             raise ValueError('Error calculating how much mass was formed at'\
                              'each age bin. Does not sum up to unity.')

    def compute_DTD(self):
        """Call the 'make_DTD' routine to compute the relevant DTD.
        """
        self.DTD_func = make_DTD(self.s1, self.s2, self.t_onset, self.t_break)        
        
    def compute_model_rates(self):
        """
        """
        sSNR, sSNRm, sSNRL = [], [], []
        
        for i, t in enumerate(self.age):
        
            t_minus_tprime = t - self.age
            integration_cond = ((self.age >= self.t_onset) & (self.age <= t))
            DTD_ages = t_minus_tprime[integration_cond]

            DTD_component = self.DTD_func(DTD_ages)
            mass_component = self.mass_formed[integration_cond]
            
            #Perform convolution. This will give the expected SNe / yr for the
            #population used as input.  
            _sSNR = np.sum(np.multiply(DTD_component,mass_component))
            if _sSNR.value == 0.:
                _sSNR = 1.e-40 * _sSNR.unit
            
            #To compute the expected SN rate per unit mass, one then ahs to divide
            #by the 'burst total mass', which for a complex SFH (i.e. exponential)
            #corresponds to the integrated formed mass up to the age being assessed. 
            _sSNRm = _sSNR / self.int_formed_mass[i]
            
            #To compute the SN rate per unit of luminosity, one can simply take
            #the sSNR and divide by the L derived from the synthetic stellar pop.
            L = 10.**(-0.4*(self.r_band[i] - 5.)) * u.Lsun
            _sSNRL = _sSNR / L
            
            sSNR.append(_sSNR), sSNRm.append(_sSNRm), sSNRL.append(_sSNRL)
        
        #Convert output lists to arrays preserving the units. 
        self.sSNR = np.array([rate.value for rate in sSNR]) * sSNR[0].unit
        self.sSNRm = np.array([rate.value for rate in sSNRm]) * sSNRm[0].unit
        self.sSNRL = np.array([rate.value for rate in sSNRL]) * sSNRL[0].unit

    def make_model(self):
        self.get_synpop_data()
        self.compute_DTD()
        self.compute_model_rates()
