#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy import units as u
from SN_rate import Model_Rates
import core_funcs

class Generate_Curve(object):
    """
    Description:
    ------------
    This class calculates a function that maps Dcolor into a SN rate. Whether
    to use SN_rate.py (faster) or SN_rate_outdated.py (also accepts SFH
    other than 'exponential') can be determined by commenting/uncommeting
    the module to be imported above. 
    
    !!!Applying the interpolation is currently taking up 90% of the time.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
    _s1 : ~float
        DTD slope prior to t_onset.
    _s2 : ~float
        DTD slope after t_onset.
    """      
    def __init__(self, _inputs, _D, _s1, _s2):

        self._inputs = _inputs
        self._D = _D
        self._s1 = _s1
        self._s2 = _s2
        
        self.Dcolor_at10Gyr = []
        self.sSNRL_at10Gyr = []
        self.Dcolor_max = None
        self.log_sSNRL_max = None
        self.Dcolor2sSNRL = None
        self.Dcolor2sSNRL_ext = None
        
        self.sSNRL_fine = None
        self.sSNRL_matrix = np.zeros(
          shape=(len(self._inputs.tau_list), len(self._D['Dcd_fine'])))

        self.run_generator()       
        
    #@profile
    def get_values_at10Gyr(self):
        """Anything that does not depend on s1 or s2, should be computed here
        to avoid wasting computational time.
        """
        
        for i, tau in enumerate(self._inputs.tau_list):
            TS = str(tau.to(u.yr).value / 1.e9)         
            model = Model_Rates(self._inputs, self._D, TS, self._s1, self._s2)
            
            age_cond = (abs(self._D['age_' + TS] - 10.) < 1.e-6)
            self.Dcolor_at10Gyr.append(self._D['Dcolor_' + TS][age_cond][0])
            self.sSNRL_at10Gyr.append(model.sSNRL[age_cond][0])
       
            self.sSNRL_matrix[i] = np.asarray(core_funcs.interpolator(
              self._D['Dcolor_' + TS], model.sSNRL, self._D['Dcd_fine']))
                        
            #Get Dcolor max from tau = 1Gyr model.
            if TS == '1.0':
                self.Dcolor_max = self._D['Dcolor_' + TS][age_cond][0]
                self.log_sSNRL_max = np.log10(model.sSNRL[age_cond][0])

        #Convert lists to arrays.
        self.Dcolor_at10Gyr = np.array(self.Dcolor_at10Gyr)
        self.sSNRL_at10Gyr = np.array(self.sSNRL_at10Gyr)    
        
    #@profile
    def average_over_models(self):
        """New method to extend Dcd range."""

        #Average models in linear space. This will raise a warning because of
        #NaN slices. This is not a problem and works as intented.
        np.warnings.filterwarnings('ignore')
        sSNRL_fine = 10.**np.nanmedian(self.sSNRL_matrix, axis=0)
        np.warnings.filterwarnings('default')

        #Assign sSNRL = 0 for galaxies bluer than the model can predict.
        cond = ((self._D['Dcd_fine'] < - 0.2) & np.isnan(sSNRL_fine))     
        sSNRL_fine[cond] = 1.e-40
                
        #Assign the sSNRL at the reddest color for galaxies redder than predicted.
        sSNRL_fine[np.isnan(sSNRL_fine)] = sSNRL_fine[~np.isnan(sSNRL_fine)][-1]
        self.sSNRL_fine = sSNRL_fine

    #@profile
    def run_generator(self):
        self.get_values_at10Gyr()
        self.average_over_models()

