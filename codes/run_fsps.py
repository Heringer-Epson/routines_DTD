#!/usr/bin/env python

import os
import time   
import fsps
import numpy as np

taus = [1., 1.5, 2., 3., 4., 5., 7., 10.]
out_dir = './../INPUT_FILES/STELLAR_POP/'
header = ('log_age,instantaneous_sfr,integrated_stellar_mass,'\
          'integrated_formed_mass,SDSS_u,SDSS_g,SDSS_r')

class Make_FSPS(object):
    """Imports PythonFSPS (which makes use of FSPS v3.0) to compute a series
    synthetic stellar population and creates output files containing the
    relevant magnitudes, ages and masses.
    """

    def __init__(self, tau_list, write_output=True):
        self.tau_list = tau_list
        self.write_output = write_output
        self.run_fsps()      

    def make_output(self, sp, fname):
        """Given a FSPS object (sp), write an output file."""
        #Data needs to be transposed otehrwise arrays are written as
        #lines rather tahn columns.
        u_mag, g_mag, r_mag = zip(*sp.get_mags(
          bands=['sdss_u', 'sdss_g', 'sdss_r']))
        out_data = (sp.log_age, sp.sfr, sp.stellar_mass, sp.formed_mass,
          u_mag, g_mag, r_mag)
        np.savetxt(fname, np.transpose(out_data), header=header,
          delimiter=',')          

    def run_fsps(self):

        os.system('clear')
        print '****************************************************'
        print '******************* RUNNING FSPS *******************'         
        print '****************************************************'
        print '\n'        

        #Make Simple Stellar Population (SSP).
        print 'Calculating a SSP.'
        sp = fsps.StellarPopulation(
          compute_vega_mags=False, zcontinuous=0, zmet=20,
          add_agb_dust_model=False, add_dust_emission=False,
          add_stellar_remnants=True, fbhb=0., pagb=0., zred=0., imf_type=1,
          sfh=0, dust_type=0.)        

        if self.write_output:
            fname = out_dir + 'SSP.dat'
            self.make_output(sp, fname)

        for tau in self.tau_list:

            print 'Calculating exponential SFH models. Tau=' + str(tau) + ' Gyr.'
            #Call FSPS to compute synthetic stellar populations 
            sp = fsps.StellarPopulation(
              compute_vega_mags=False, zcontinuous=0, zmet=20,
              add_agb_dust_model=False, add_dust_emission=False,
              add_stellar_remnants=True, fbhb=0., pagb=0., zred=0., imf_type=1,
              sfh=1, tau=tau, const=0., fburst=0., dust_type=0.)

            #Write output files.
            if self.write_output:
                fname = out_dir + 'exponential_tau-' + str(tau) + '.dat'
                self.make_output(sp, fname)       
            
        #Print additional relevant information regarding FSPS setup.
        print '\nSettings used were:'
        print '  -Isochrones: ' + sp.libraries[0]
        print '  -Spectral library: ' + sp.libraries[1] + '\n\n'

if __name__ == '__main__': 
    Make_FSPS(taus, write_output=True)


