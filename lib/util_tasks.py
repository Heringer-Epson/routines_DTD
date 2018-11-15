#!/usr/bin/env python

import os
import sys
import warnings
import numpy as np
import pandas as pd
from astropy import units as u

class Utility_Routines(object):
    """
    Description:
    ------------
    This code will perform some basic tests to ensure the some of the input
    parameters are appropriate.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/RUNS/$RUN_DIR/record.dat'
    """ 
    def __init__(self, _inputs):
        self._inputs = _inputs
        self.run_utilities()

    def check_variables(self):
        """Perform variable type checks."""
        #Make sure tau_list includes 1 Gyr (used to compute a Dcolor limit). If
        #not, include it and sort list. Raise Warning.
        if 1.e9 * u.yr not in self._inputs.tau_list:
            self._inputs.tau_list = np.append(
              self._inputs.tau_list.to(u.yr).value, 1.e9) * u.yr
            self._inputs.tau_list = np.sort(self._inputs.tau_list)
            warning_msg = (
              'Variable tau_list must contain 1 Gyr for a Dcolor limit to be '
              'compute. This value was added to the passed list.')
            warnings.warn(warning_msg)
        if (self._inputs.filter_1 is not 'r' or
            self._inputs.filter_2 is not 'g'):
            warning_msg = (
              'The observational data is currently only available in the '\
              'g-r color and therefore likelihoods cannot be computed for the '\
              'choice of "%s"-"%s"'\
              %(self._inputs.filter_2, self._inputs.filter_1))            
            warnings.warn(warning_msg)

    def initialize_outfolder(self):
        """Create the directory if it doesn't already exist and dd a copy of
        this input file in that directory."""
        if not os.path.exists(self._inputs.subdir_fullpath):
            os.makedirs(self._inputs.subdir_fullpath)
        if not os.path.exists(self._inputs.subdir_fullpath + 'fsps_FILES/'):
            os.makedirs(self._inputs.subdir_fullpath + 'fsps_FILES/')        
        if not os.path.exists(self._inputs.subdir_fullpath + 'FIGURES/'):
            os.makedirs(self._inputs.subdir_fullpath + 'FIGURES/')
        if not os.path.exists(self._inputs.subdir_fullpath + 'likelihoods/'):
            os.makedirs(self._inputs.subdir_fullpath + 'likelihoods/')

    def run_utilities(self):
        self.check_variables()
        self.initialize_outfolder()
