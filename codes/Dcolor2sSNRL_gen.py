#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from SN_rate_gen import Model_Rates

class Generate_Curve(object):
    
    def __init__(self, _inputs, _s1, _s2):

        self._inputs = _inputs
        self._s1 = _s1
        self._s2 = _s2
        
        self.Dcolor_at10Gyr = []
        self.sSNRL_at10Gyr = []
        self.Dcolor_max = None
        self.log_sSNRL_max = None
        self.Dcolor2sSNRL = None

        self.run_generator()
        
    #@profile
    def get_values_at10Gyr(self):
        
        for tau in self._inputs.tau_list:
            tau_suffix = str(tau.to(u.yr).value / 1.e9)
            synpop_dir = self._inputs.subdir_fullpath + 'fsps_FILES/'
            synpop_fname = self._inputs.sfh_type + '_tau-' + tau_suffix + '.dat'
            
            model = Model_Rates(
              self._s1, self._s2, self._inputs.t_onset,
              self._inputs.t_cutoff, self._inputs.filter_1,
              self._inputs.filter_2, self._inputs.imf_type,
              self._inputs.sfh_type, self._inputs.Z,
              synpop_dir, synpop_fname)
            
            age_cond = (model.age.to(u.yr).value == 1.e10)
            self.Dcolor_at10Gyr.append(model.Dcolor[age_cond][0])
            self.sSNRL_at10Gyr.append(model.sSNRL[age_cond][0])
            
        #Convert lists to arrays.
        self.Dcolor_at10Gyr = np.array(self.Dcolor_at10Gyr)
        self.sSNRL_at10Gyr = np.array(self.sSNRL_at10Gyr)

    #@profile
    def get_Dcolor_max(self):
        synpop_dir = self._inputs.subdir_fullpath + 'fsps_FILES/'
        synpop_fname = self._inputs.sfh_type + '_tau-1.0.dat'
    
        if 1.e9 * u.yr in self._inputs.tau_list:
            model = Model_Rates(
              self._s1, self._s2, self._inputs.t_onset, self._inputs.t_cutoff,
              self._inputs.filter_1, self._inputs.filter_2,
              self._inputs.imf_type, self._inputs.sfh_type,
              self._inputs.Z, synpop_dir, synpop_fname)
            
            age_cond = (model.age.to(u.yr).value == 1.e10)
            self.Dcolor_max = model.Dcolor[age_cond][0]
            self.log_sSNRL_max = np.log10(model.sSNRL[age_cond][0])

        else:
            raise ValueError(
              'Maximum Dcolor is compute based on the SFH where tau = 1 Gyr, '\
              'which is not present in the tau_list passed. Note that'\
              'sSNRL(Dcolor >= Dcolor_max) = sSNRL(Dcolor_max).')        

    #@profile
    def build_function(self):
        """Interpolate Dcolor and sSNRL (in log space).
        """
        Dcolor2sSNRL_interp = interp1d(self.Dcolor_at10Gyr,
                                       np.log10(self.sSNRL_at10Gyr))
        
        def Dcolor2sSNRL_builder(Dcolor):
            if Dcolor <= self.Dcolor_max:
                log_sSNRL = Dcolor2sSNRL_interp(Dcolor)
            elif Dcolor > self.Dcolor_max:
                log_sSNRL = self.log_sSNRL_max
            _sSNRL = 10.**log_sSNRL
            return _sSNRL   
        
        self.Dcolor2sSNRL = np.vectorize(Dcolor2sSNRL_builder)
        
    def run_generator(self):
        self.get_values_at10Gyr()
        self.get_Dcolor_max()
        self.build_function()
        
if __name__ == '__main__':
    tau_list = [1., 1.5, 2., 3., 4., 5., 7., 10.]
    tau_list = [tau * 1.e9 * u.yr for tau in tau_list]
    generator = Generate_Curve(
      -1., -2., 1.e8 * u.yr, 1.e9 * u.yr, 'sdss_r', 'sdss_g', 'Chabrier',
      'exponential', 0.0190, tau_list)
