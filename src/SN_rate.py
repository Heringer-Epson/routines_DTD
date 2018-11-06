#!/usr/bin/env python

import numpy as np
import pandas as pd
import ctypes
from scipy.integrate import quad
from astropy import units as u

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
    """       
    def __init__(self, _inputs, s1, s2, tau):

        self._inputs = _inputs
        self.s1 = s1
        self.s2 = s2
        self.tau = tau
        
        self.A = 1. #Arbitrary constant that is later renormalized.
        self.age = None
        self.int_stellar_mass = None
        self.int_formed_mass = None
        self.mag_1 = None
        self.mag_2 = None
        self.L = None
        self.mass_formed = None
        self.RS_color = None
        self.Dcolor = None
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
        mag_2_SSP = df[self._inputs.filter_2].values
        mag_1_SSP = df[self._inputs.filter_1].values
        
        #Retrieve RS color.
        RS_condition = (logage_SSP == 10.0)
        self.RS_color = mag_2_SSP[RS_condition] - mag_1_SSP[RS_condition]

        #Get data for the complex SFH (i.e. exponential.)
        tau_suffix = str(self.tau.to(u.yr).value / 1.e9)
        synpop_fname = 'tau-' + tau_suffix + '.dat'
        fpath = synpop_dir + synpop_fname
        df = pd.read_csv(fpath, header=0)
        
        self.age = 10.**df['# log_age'].values * u.yr
        self.sfr = df['instantaneous_sfr'].values
        self.int_stellar_mass = df['integrated_stellar_mass'].values
        self.int_formed_mass = df['integrated_formed_mass'].values
        self.mag_2 = df[self._inputs.filter_2].values
        self.mag_1 = df[self._inputs.filter_1].values
        
        self.Dcolor = self.mag_2 - self.mag_1 - self.RS_color               
    
    #@profile
    def compute_model_rates(self):
        sSNR, sSNRm, sSNRL, L = [], [], [], []

        _t_ons = self._inputs.t_onset.to(u.Gyr).value
        _t_bre = self._inputs.t_cutoff.to(u.Gyr).value
        _tau = self.tau.to(u.Gyr).value
        _t = self.age.to(u.Gyr).value
        
        const_s2 = self.A * np.power(_t_bre, self.s1 - self.s2)

        try:
            lib = ctypes.CDLL('./DTD_gen.so')
        except:
            lib = ctypes.CDLL('./../src/DTD_gen.so')
            
        if self._inputs.sfh_type == 'exponential':
            sfr_norm = (
              -1. / (_tau * (np.exp(-_t[-1] / _tau) - np.exp(-_t[0] / _tau))))     
            int_f = lib.conv_exponential_sSNR

        elif self._inputs.sfh_type == 'delayed-exponential':
            sfr_norm = (
              1. / (((-_tau * _t[-1] - _tau**2.) * np.exp(-_t[-1] / _tau)) -
              ((-_tau * _t[0] - _tau**2.) * np.exp(-_t[0] / _tau))))
            int_f = lib.conv_delayed_exponential_sSNR        
        
        int_f.restype = ctypes.c_double
        int_f.argtypes = (ctypes.c_int, ctypes.c_double)
        
        self.sSNR = np.zeros(len(self.age))
        age_cond = (_t >= _t_ons)
        
        self.sSNR[age_cond] = [
          quad(int_f, _t_ons, t, (t, _tau, self.A, self.s1, self.s2, _t_ons,
          _t_bre, sfr_norm, const_s2))[0] for t in _t[age_cond]]
        
        #Remove zeros because of log calculations later on.
        self.sSNR[self.sSNR == 0] = 1.e-40
        
        self.sSNRm = np.divide(self.sSNR,self.int_formed_mass)
        self.L = 10.**(-0.4 * (self.mag_1 - 5.))
        self.sSNRL = np.divide(self.sSNR,self.L)
        
    #@profile
    def make_model(self):
        self.get_synpop_data()
        self.compute_model_rates()
        
if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Model_Rates(class_input(case='test-case'), -1., -2., 1. * u.Gyr)

