#!/usr/bin/env python

import time
import fsps
import numpy as np
from astropy import units as u

directory = './../INPUT_FILES/fsps_FILES/'
sfh_type2sfh_key = {'exponential': 1, 'delayed-exponential': 4}
imf_type2imf_key = {'Salpeter': 0, 'Chabrier': 1, 'Kroupa': 2}
Z2Z_key = {0.0002: 1, 0.0003: 2, 0.0004: 3, 0.0005: 4, 0.0006: 5, 0.0008: 6,
           0.0010: 7, 0.0012: 8, 0.0016: 9, 0.0020: 10, 0.0025: 11, 0.0031: 12,
           0.0039: 13, 0.0049: 14, 0.0061: 15, 0.0077: 16, 0.0096: 17,
           0.0120: 18, 0.0150: 19, 0.0190: 20, 0.0240: 21, 0.0300: 22}

class Make_FSPS(object):
    """
    Description:
    ------------
    This is a stand alone routine to generate default FSPS files which can be
    used in case the user does not have fsps and python-fsps installed.
     
    Outputs:
    --------
    ./../INPUT_FILES/RUNS/fsps_FILES/X_Y.dat
        where X is the chosen SFH ('exponential' or 'delayed-exponential')
        and Y in the timescale.
    """  

    def __init__(self):
        print '\n\n>RUNNING FSPS... (requires fsps and python-fsps installed!)\n'
        self.filter_1 = 'sdss_r'
        self.filter_2 = 'sdss_g'
        self.imf_type = 'Chabrier'
        self.sfh_type = ['exponential', 'delayed-exponential']
        self.Z = 0.0190
        
        _tau_list = [1., 1.5, 2., 3., 4., 5., 7., 10.]
        self.tau_list = [tau * 1.e9 * u.yr for tau in _tau_list]
    
        self.run_fsps() 
        
    def make_output(self, sp, fname):
        """Given a FSPS object (sp), write an output file."""
        header = ('log_age,instantaneous_sfr,integrated_stellar_mass,'\
                  + 'integrated_formed_mass,' + self.filter_2 + ','\
                  + self.filter_1)

        #Data needs to be transposed, otherwise arrays are written as
        #lines rather than columns.
        mag_1, mag_2 = zip(*sp.get_mags(
          bands=[self.filter_1, self.filter_2]))
        out_data = (sp.log_age, sp.sfr, sp.stellar_mass, sp.formed_mass,
          mag_2, mag_1)
        np.savetxt(directory + fname, np.transpose(out_data), header=header,
          delimiter=',')          

    def run_fsps(self):
        imf_key = imf_type2imf_key[self.imf_type]
        Z_key = Z2Z_key[self.Z]

        #Make Simple Stellar Population (SSP).
        print '  *Calculating a SSP.'
        sp = fsps.StellarPopulation(
          compute_vega_mags=False, zcontinuous=0, zmet=Z_key,
          add_agb_dust_model=False, add_dust_emission=False,
          add_stellar_remnants=True, fbhb=0., pagb=0., zred=0.,
          imf_type=imf_key, sfh=0, dust_type=0., tage=14.96)        

        #Write output file.
        fname = 'SSP.dat'
        self.make_output(sp, fname)

        #Make complex SFH models:        
        for _sfh_type in self.sfh_type:
       
            sfh_key = sfh_type2sfh_key[_sfh_type]
            print '  *Calculating ' + _sfh_type + ' models.'
        
            for tau in self.tau_list:
                tau = tau.to(u.yr).value / 1.e9
                print '    Tau=' + str(tau) + ' Gyr.'
                #Call FSPS to compute synthetic stellar populations 
                sp = fsps.StellarPopulation(
                  compute_vega_mags=False, zcontinuous=0, zmet=Z_key,
                  add_agb_dust_model=False, add_dust_emission=False,
                  add_stellar_remnants=True, fbhb=0., pagb=0., zred=0., imf_type=imf_key,
                  sfh=sfh_key, tau=tau, const=0., fburst=0., dust_type=0., tage=14.96)

                #Write output files.
                fname = _sfh_type + '_tau-' + str(tau) + '.dat'
                self.make_output(sp, fname)            

        #Make record file.
        with open(directory + 'fsps_info.txt', 'w') as out:
            out.write('Files created on: ' + time.strftime('%d/%m/%Y') + '\n')
            out.write('FSPS version: ' + str(fsps.__version__) + '\n')
            out.write('Isochrones: ' + sp.libraries[0] + '\n')
            out.write('Spectral library: ' + sp.libraries[1])
        
if __name__ == '__main__':
    Make_FSPS()
