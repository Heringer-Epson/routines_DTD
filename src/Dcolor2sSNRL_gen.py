#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from SN_rate import Model_Rates

def make_fine_model(Dcd, Dcd_fine, log_sSNRL):
    Dcd2sSNRL_func = interp1d(Dcd,log_sSNRL,bounds_error=False)
                              #fill_value=(log_sSNRL[0],log_sSNRL[-1]))
    return Dcd2sSNRL_func(Dcd_fine)

class Generate_Curve(object):
    """
    Description:
    ------------
    This class calculates a function that maps Dcolor into a SN rate. Whether
    to use SN_rate.py (faster) or SN_rate_outdated.py (also accepts SFH
    other than 'exponential') can be determined by commenting/uncommeting
    the module to be imported above. 

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
    _s1 : ~float
        DTD slope prior to t_onset.
    _s2 : ~float
        DTD slope after t_onset.
    """      
    def __init__(self, _inputs, _s1, _s2):

        self._inputs = _inputs
        self._s1 = _s1
        self._s2 = _s2
        
        self.Dcolor_at10Gyr = []
        self.sSNRL_at10Gyr = []
        self.Dcolor_max = None
        self.log_sSNRL_max = None
        self.Dcolor2sSNRL = None
        self.Dcolor2sSNRL_ext = None
        self.Dcd_fine = np.arange(-1.1, 1.00001, 0.01)
        self.sSNRL_fine = None
        self.sSNRL_matrix = np.zeros(
          shape=(len(self._inputs.tau_list), len(self.Dcd_fine)))

        self.run_generator()
        
    #@profile
    def get_values_at10Gyr(self):
        
        for i, tau in enumerate(self._inputs.tau_list):
            tau_suffix = str(tau.to(u.yr).value / 1.e9)
            synpop_dir = self._inputs.subdir_fullpath + 'fsps_FILES/'
            synpop_fname = 'tau-' + tau_suffix + '.dat'
            
            model = Model_Rates(self._inputs, self._s1, self._s2, tau)
            
            age_cond = (model.age.to(u.yr).value == 1.e10)
            self.Dcolor_at10Gyr.append(model.Dcolor[age_cond][0])
            self.sSNRL_at10Gyr.append(model.sSNRL[age_cond][0])
                        
            self.sSNRL_matrix[i] = make_fine_model(model.Dcolor,self.Dcd_fine,
                                                   np.log10(model.sSNRL))
                        
        #Convert lists to arrays.
        self.Dcolor_at10Gyr = np.array(self.Dcolor_at10Gyr)
        self.sSNRL_at10Gyr = np.array(self.sSNRL_at10Gyr)

    #@profile
    def get_Dcolor_max(self):
        synpop_dir = self._inputs.subdir_fullpath + 'fsps_FILES/'
        synpop_fname = 'tau-1.0.dat'
    
        tau = 1. * u.Gyr
        if tau in self._inputs.tau_list:
            model = Model_Rates(self._inputs, self._s1, self._s2, tau)
            
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
        """Interpolate Dcolor and sSNRL (in log space)."""
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

    def average_over_models(self):
        """New method to extend Dcd range."""

        #Average models in linear space. This will raise a warning because of
        #NaN slices. This is not a problem and works as intented.
        np.warnings.filterwarnings('ignore')
        sSNRL_fine = 10.**np.nanmedian(self.sSNRL_matrix, axis=0)
        np.warnings.filterwarnings('default')

        #Assign sSNRL = 0 for galaxies bluer than the model can predict.
        cond = ((self.Dcd_fine < - 0.2) & np.isnan(sSNRL_fine))     
        sSNRL_fine[cond] = 1.e-40
                
        #Assign the sSNRL at the reddest color for galaxies redder than predicted.
        sSNRL_fine[np.isnan(sSNRL_fine)] = sSNRL_fine[~np.isnan(sSNRL_fine)][-1]
        self.sSNRL_fine = sSNRL_fine

        #Create an extended function, which accepts bluer colors than in H17.
        #In current versions of scipy, the fill_values bug has been correct.
        #Activate tardis_up. However, kcorrect will not work...
        log_sSNRL_fine = np.log10(self.sSNRL_fine)
        fine_func = interp1d(
          self.Dcd_fine,log_sSNRL_fine,bounds_error=False,
          fill_value=(log_sSNRL_fine[0],log_sSNRL_fine[-1]))

        def Dcolor2sSNRL_ext_builder(Dcolor):
            return 10.**fine_func(Dcolor)
        self.Dcolor2sSNRL_ext = np.vectorize(Dcolor2sSNRL_ext_builder)

    #@profile
    def run_generator(self):
        self.get_values_at10Gyr()
        self.get_Dcolor_max()
        self.build_function()
        self.average_over_models()
        
if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Generate_Curve(class_input(case='test-case'), -1., -1.)
