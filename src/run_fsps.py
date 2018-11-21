#!/usr/bin/env python

import os
import sys
import time
import warnings
import numpy as np
from astropy import units as u
from shutil import copyfile

sfh_type2sfh_key = {'exponential': 1, 'delayed-exponential': 4}
imf_type2imf_key = {'Salpeter': 0, 'Chabrier': 1, 'Kroupa': 2}
Z2Z_key_PADOVA = {
  0.0002: 1, 0.0003: 2, 0.0004: 3, 0.0005: 4, 0.0006: 5, 0.0008: 6,
  0.0010: 7, 0.0012: 8, 0.0016: 9, 0.0020: 10, 0.0025: 11, 0.0031: 12,
  0.0039: 13, 0.0049: 14, 0.0061: 15, 0.0077: 16, 0.0096: 17,
  0.0120: 18, 0.0150: 19, 0.0190: 20, 0.0240: 21, 0.0300: 22}
Z2Z_key_BASTI = {
  0.0003: 1, 0.0006: 2, 0.0010: 3, 0.0020: 4, 0.0040: 5, 0.0080: 6,
  0.0100: 7, 0.0200: 8, 0.0300: 9, 0.0400: 10}
Z2Z_key_GENEVA = {
  0.0010: 1, 0.0040: 2, 0.0080: 3, 0.0200: 4, 0.0400: 5}
Z2Z_key_PARSEC = {
  0.0001: 1, 0.0002: 2, 0.0005: 3, 0.0010: 4, 0.0020: 5, 0.0040: 6,
  0.0060: 7, 0.0080: 8, 0.0100: 9, 0.0140: 10, 0.0170: 11, 0.0200: 12,
  0.0300: 13, 0.0400: 14, 0.0600: 15}

class Make_FSPS(object):
    """
    Description:
    ------------
    Imports PythonFSPS (which makes use of FSPS v3.0) to compute a series
    synthetic stellar population and creates output files containing the
    relevant magnitudes, ages and masses. This creates (copies) a series of
    FSPS files in the RUN directory if the 'run_fsps_flag' is set to True
    (False) in the input_params.py file.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/fsps_FILES/X_Y.dat
        where X is the chosen SFH ('exponential' or 'delayed-exponential')
        and Y in the timescale.
    """        
    
    def __init__(self, _inputs):
        self._inputs = _inputs

    def make_output(self, sp, fname):
        """Given a FSPS object (sp), write an output file."""
        directory = self._inputs.subdir_fullpath + 'fsps_FILES/'
        header = ('log_age,instantaneous_sfr,integrated_stellar_mass,'\
                  + 'integrated_formed_mass,' + self._inputs.filter_2 + ','\
                  + self._inputs.filter_1)

        #Data needs to be transposed, otherwise arrays are written as
        #lines rather than columns.
        mag_1, mag_2 = zip(*sp.get_mags(bands=['sdss_' + self._inputs.filter_1,
                                               'sdss_' + self._inputs.filter_2]))
        out_data = (sp.log_age, sp.sfr, sp.stellar_mass, sp.formed_mass,
          mag_2, mag_1)
        np.savetxt(directory + fname, np.transpose(out_data), header=header,
          delimiter=',')          

    def run_fsps(self):
        print '\n\n>RUNNING FSPS... (requires fsps and python-fsps installed)\n'
        sys.path.append(os.environ['PY_FSPS_DIR'])
        import fsps
        
        imf_key = imf_type2imf_key[self._inputs.imf_type]
        sfh_key = sfh_type2sfh_key[self._inputs.sfh_type]
        if self._inputs.isoc_lib == 'PADOVA':
            Z_key = Z2Z_key_PADOVA[self._inputs.Z]
        elif self._inputs.isoc_lib == 'BASTI':
            Z_key = Z2Z_key_BASTI[self._inputs.Z]
        elif self._inputs.isoc_lib == 'GENEVA':
            Z_key = Z2Z_key_GENEVA[self._inputs.Z]
        elif self._inputs.isoc_lib == 'PARSEC':
            Z_key = Z2Z_key_PARSEC[self._inputs.Z]
            
        #Make Simple Stellar Population (SSP).
        print '  *Calculating a SSP.'
        sp = fsps.StellarPopulation(
          compute_vega_mags=False, zcontinuous=0, zmet=Z_key,
          add_agb_dust_model=False, add_dust_emission=False,
          add_stellar_remnants=True, fbhb=self._inputs.fbhb, pagb=0., zred=0.,
          imf_type=imf_key, sfh=0, dust_type=0., tage=14.96)        

        #Write output files.
        fname = 'SSP.dat'
        self.make_output(sp, fname)
        
        #Make complex SFH models:        
        print '  *Calculating ' + self._inputs.sfh_type + ' models.'
        for tau in self._inputs.tau_list:
            tau = tau.to(u.yr).value / 1.e9
            print '    Tau=' + str(tau) + ' Gyr.'
            #Call FSPS to compute synthetic stellar populations 
            sp = fsps.StellarPopulation(
              compute_vega_mags=False, zcontinuous=0, zmet=Z_key,
              add_agb_dust_model=False, add_dust_emission=False,
              add_stellar_remnants=True, fbhb=self._inputs.fbhb, pagb=0., zred=0., imf_type=imf_key,
              sfh=sfh_key, tau=tau, const=0., fburst=0., dust_type=0., tage=14.96)

            #Write output files.
            fname = 'tau-' + str(tau) + '.dat'
            self.make_output(sp, fname)

        #Make record file.
        directory = self._inputs.subdir_fullpath + 'fsps_FILES/'
        with open(directory + 'fsps_info.txt', 'w') as out:
            out.write('Files created on: ' + time.strftime('%d/%m/%Y') + '\n')
            out.write('FSPS version: ' + str(fsps.__version__) + '\n')
            out.write('Isochrones: ' + sp.libraries[0] + '\n')
            out.write('Spectral library: ' + sp.libraries[1])
            
    def copy_premade_files(self):
        fsps_dir = (
          self._inputs.imf_type + '_' + self._inputs.sfh_type + '_'
          + self._inputs.Z + '_' + str(self._inputs.fbhb) + '_' + self._inputs.spec_lib + '_'
          + self._inputs.isoc_lib + '/')
        for fname in os.listdir('./../INPUT_FILES/fsps_FILES/' + fsps_dir):
            copyfile('./../INPUT_FILES/fsps_FILES/' + fsps_dir + fname,
                     self._inputs.subdir_fullpath + 'fsps_FILES/' + fname)       
