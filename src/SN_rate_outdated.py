#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy import units as u
from DTD_gen import make_DTD

class Model_Rates(object):
    """
    Description:
    ------------
    This class will compute SN rate values for a given set of input parameters.
    These rates are mapped to Dcolors and masses.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
        
    Quantities:
    -----------
    self.sSNR: SN rate
    self.sSNRm: SN rate per unit mass
    self.sSNRL: SN rate per unit luminosity (in the r-band of SDSS).
    self.L = Luminosity

    Notes:
    ------
    This code is currently only used as a test, since a faster version that
    uses a C wrapper is available in SN_rate.py. However, SN_rate.py needs to
    use an analytical function for the SFH, whereas this code reads the SFH
    as created by FSPS. Therefore this code might still be relevant when
    computing models for SFH other than 'exponential'.
    """     
    def __init__(self, _inputs, A, s1, s2, tau):

        self._inputs = _inputs
        self.A = A * 1.e9
        self.s1 = s1
        self.s2 = s2
        self.tau = tau
        
        self.age = None
        self.int_stellar_mass = None
        self.int_formed_mass = None
        self.mag_1 = None
        self.mag_2 = None
        self.L = None
        self.mass_formed = None
        self.RS_color = None
        self.Dcolor = None
        self.DTD_func = None
        self.sSNR = None
        self.sSNRm = None
        self.sSNRL = None

        self.make_model()

    #@profile
    def get_synpop_data(self):
        """Read output data from FSPS runs. Besides the file specified by
        in self.synpop_fpath, always read data from a SSP run so that colors
        can be computed with respect to the red sequence. 
        """
        synpop_dir = self._inputs.subdir_fullpath + 'fsps_FILES/'

        #Get SSP data.
        fpath = synpop_dir + 'SSP.dat'
        df = pd.read_csv(fpath, header=0, escapechar='#')
        logage_SSP = df[' log_age'].values
        sdss_g_SSP = df['sdss_g'].values
        sdss_r_SSP = df['sdss_r'].values
        
        #Retrieve RS color.
        RS_condition = (logage_SSP == 10.0)
        self.RS_color = sdss_g_SSP[RS_condition] - sdss_r_SSP[RS_condition]

        #Get data for the complex SFH (i.e. exponential.)
        tau_suffix = str(self.tau.to(u.yr).value / 1.e9)
        synpop_fname = self._inputs.sfh_type + '_tau-' + tau_suffix + '.dat'
        fpath = synpop_dir + synpop_fname
        df = pd.read_csv(fpath, header=0)
        
        self.age = 10.**df['# log_age'].values * u.yr
        self.sfr = df['instantaneous_sfr'].values
        self.int_stellar_mass = df['integrated_stellar_mass'].values
        self.int_formed_mass = df['integrated_formed_mass'].values
        self.mag_1 = df['sdss_r'].values
        self.mag_2 = df['sdss_g'].values
        
        self.Dcolor = self.mag_2 - self.mag_1 - self.RS_color                

    #@profile
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

    #@profile
    def convolve_functions(self, func1, func2, x):
        def out_func(xprime):
            return func1(x - xprime) * func2(xprime)
        return out_func

    #@profile
    def compute_model_rates(self):
        sSNR, sSNRm, sSNRL, L = [], [], [], []

        _t_ons = self._inputs.t_onset.to(u.yr).value
        _t_cut = self._inputs.t_cutoff.to(u.yr).value
        _tau = self.tau.to(u.yr).value
    
        self.DTD_func = make_DTD(
          self.A, self.s1, self.s2, _t_ons * u.yr, _t_cut * u.yr)
        self.sfr_func = self.make_sfr_func(self.age.to(u.yr).value, self.sfr)
        
        t0 = self.age.to(u.yr).value[0]
        for i, t in enumerate(self.age.to(u.yr).value):
            
            self.conv_func = self.convolve_functions(self.sfr_func, self.DTD_func, t)
          
            #Here, since the DTD is zero prior to t_ons, we start the
            #integration at t_ons.            
            if t >= _t_ons:
                _sSNR = quad(self.conv_func, _t_ons, t)[0]
            else:
                _sSNR = 0.
            
            #To compute the expected SN rate per unit mass, one then has to divide
            #by the 'burst total mass', which for a complex SFH (i.e. exponential)
            #corresponds to the integrated formed mass up to the age being assessed. 
            _sSNRm = _sSNR / self.int_formed_mass[i]
            
            #To compute the SN rate per unit of luminosity, one can simply take
            #the sSNR and divide by the L derived from the synthetic stellar pop.
            _L = 10.**(-0.4 * (self.mag_1[i] - 5.))
            _sSNRL = _sSNR / _L
            
            sSNR.append(_sSNR), sSNRm.append(_sSNRm), sSNRL.append(_sSNRL), L.append(_L)
        
        #Convert output lists to arrays preserving the units. 
        self.sSNR = np.array(sSNR)
        self.sSNRm = np.array(sSNRm)
        self.sSNRL = np.array(sSNRL)
        self.L = np.array(L)

    #@profile
    def make_model(self):
        self.get_synpop_data()
        self.compute_model_rates()
        
if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Model_Rates(class_input(case='SDSS_gr_Maoz'), 1.e-3, -1., -2., 1. * u.Gyr)
    
    
    
    
