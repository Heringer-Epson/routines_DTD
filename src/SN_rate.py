#!/usr/bin/env python
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import numpy as np
import ctypes
from scipy.integrate import quad

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
    #@profile
    def __init__(self, _inputs, D, TS, s1, s2):

        self._inputs = _inputs
        self.D = D
        self.s1 = s1
        self.s2 = s2
                   
        const_s2 = np.power(self.D['t_bre'] / self.D['t_ons'], self.s1 - self.s2)
        
        lib = ctypes.CDLL('./../lib/DTD_gen.so')
        if self._inputs.sfh_type == 'exponential':
            int_f = lib.conv_exponential_sSNR
        elif self._inputs.sfh_type == 'delayed-exponential':
            int_f = lib.conv_delayed_exponential_sSNR        
        
        int_f.restype = ctypes.c_double
        int_f.argtypes = (ctypes.c_int, ctypes.c_double)
        
        self.sSNR = np.zeros(len(self.D['age_' + TS])) + 1.e-40
        age_cond = (self.D['age_' + TS] >= self.D['t_ons'])
                
        #This is the current bottleneck.
        self.sSNR[age_cond] = [
          quad(int_f, self.D['t_ons'], t, (t, self.D['tau_' + TS], self.s1, self.s2,
          self.D['t_ons'], self.D['t_bre'], self.D['sfr_norm_' + TS], const_s2))[0]
          for t in self.D['age_' + TS][age_cond]]
        
        #Remove zeros because of log calculations later on.
        self.sSNR[self.sSNR == 0] = 1.e-40
        
        self.sSNRm = np.divide(self.sSNR,self.D['int_mass_' + TS])
        self.L = 10.**(-0.4 * (self.D['mag1_' + TS] - 5.))
        self.sSNRL = np.divide(self.sSNR,self.L)

