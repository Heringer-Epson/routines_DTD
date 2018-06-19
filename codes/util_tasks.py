#!/usr/bin/env python

import os
import sys
import time
import warnings
import numpy as np
from astropy import units as u

class Utility_Routines(object):
    """
    TBW.
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
        if (self._inputs.filter_1 is not 'sdss_r' or
            self._inputs.filter_2 is not 'sdss_g'):
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

    def make_record(self):
        fpath = self._inputs.subdir_fullpath + 'record.dat'
        with open(fpath, 'w') as rec:
            rec.write('run date: ' + time.strftime('%d/%m/%Y') + '\n')
            rec.write('---Model params---\n')
            rec.write('Colour: ' + self._inputs.filter_2 + '-'\
                      + self._inputs.filter_1 + '\n')
            rec.write('IMF: ' + self._inputs.imf_type + '\n')
            rec.write('SFH: ' + self._inputs.sfh_type + '\n')
            rec.write('Metallicity: ' + str(self._inputs.Z) + '\n')
            rec.write('t_onset: ' + str(format(
              self._inputs.t_onset.to(u.yr).value / 1.e9, '.2f')) + ' Gyr \n')
            rec.write('t_cutoff: ' + str(format(
              self._inputs.t_cutoff.to(u.yr).value / 1.e9, '.2f')) + ' Gyr \n')
            rec.write('Dcolor_min: ' + str(self._inputs.Dcolor_min) + ' \n')
            rec.write('Dcolor_max: ' + str(self._inputs.Dcolor_max) + ' \n\n')
            rec.write('---Dataset---\n')
            #rec.write('Control galaxies: ' + self._inputs.ctrl_fpath + ' \n')
            #rec.write('Host galaxies: ' + self._inputs.host_fpath)

    def run_utilities(self):
        self.check_variables()
        self.initialize_outfolder()
        self.make_record()        
