#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import kcorrect
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.32)
Q_factor = 1.6
z_ref = 0.

class Process_Data(object):
    """Compute absolute magnitudes and necessary corrections to the data used
    Maox+2012 (http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M). This data
    was provided in priv. comm.
    
    Dependencies: To compute Kcorrection, the package KCORRECT
    (Blanton & Roweis 2007) and a Python wrap are required.
    """
    def __init__(self):
        self.df = None
        self.perform_run()

    def read_data(self):
        fpath = './../INPUT_FILES/Maoz_file/Maoz_matched.csv'
        self.df = pd.read_csv(fpath, header=0)
        
        #Convert number of SN from boolean to integer (SDSS query messed this).
        self.df['n_SN'] = self.df['n_SN'].apply(lambda x: int(x))

    def perform_redshift_check(self):
        fail_cond = (self.df.redshift - self.df.z > 1.e-3)
        if len(self.df[fail_cond]) > 0:
            error = 'Error: The photometry matched to Maoz data may be wrong,'\
                    ' as the given and retrieved redshifts do not agree.'
            raise ValueError(error)
        
    def trim_data(self):
        """Perform data cuts similar to the original publication:
        selection 1: -57 < ra < 57
        selection 2: 0 < z <= 0.4
        """
        self.df = self.df[(self.df.ra > 360. - 57.) | (self.df.ra < 57.0)]
        self.df = self.df[(self.df.redshift > 0.01) & (self.df.redshift <= 0.4)]

    def extinction_correction(self):
        """Compute extinction corrections. This is simply done by using the
        values retrieved from SDSS - to check if these corrections are valid
        only for 'Modelmags' or also for 'petromags'. This correction needs
        to be done before computing Kcorrections. Error not propragated here.
        Subtraction sign is correct; e.g. see quantity 'dered_u' under DR7,
        Views, SpecPhoto, http://skyserver.sdss.org/CasJobs/SchemaBrowser.aspx
        """
        for fltr in ['u', 'g', 'r', 'i', 'z']:
            self.df['ext_' + fltr] = (
              self.df['petroMag_' + fltr] - self.df['extinction_' + fltr])

    def make_Kcorrections(self):
        """Make the appropriate kcorrections using the package KCORRECT by
        Blanton & Roweis 2007. Code is available at http://kcorrect.org/ (v4_3)
        Note that a Python wrap around is also used; developed by nirinA and
        mantained at https://pypi.org/project/kcorrect_python. (v2017.07.05)         
        """
        kcorrect.load_templates()
        kcorrect.load_filters(f='sdss_filters.dat', band_shift=z_ref)
        
        def compute_kcorrection(redshift, u, g, r, i, z,
                                uErr, gErr, rErr, iErr, zErr):           
            out_coeffs = kcorrect.fit_coeffs(
              [redshift, u, uErr, g, gErr, r, rErr, i, iErr, z, zErr])
            out_kcorrection = kcorrect.reconstruct_maggies(out_coeffs)
            u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr = [
              kcorr for kcorr in out_kcorrection[1::]]
            return u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr
        
        self.df['kcorr_u'], self.df['kcorr_g'], self.df['kcorr_r'],\
        self.df['kcorr_i'], self.df['kcorr_z'] =\
          np.vectorize(compute_kcorrection)(
          self.df['redshift'], self.df['ext_u'], self.df['ext_g'], 
          self.df['ext_r'], self.df['ext_i'], self.df['ext_z'],
          self.df['petroMagErr_u'], self.df['petroMagErr_g'],
          self.df['petroMagErr_r'], self.df['petroMagErr_i'],
          self.df['petroMagErr_z'])    

    def compute_abs_mags(self):
        """Compute absolute magnitudes. Note that the extinction correction
        is done here but shouldn't because it impacts the kcorr calculation.
        Note that the kcorr should be subtracted, as explicit in:
        http://cosmo.nyu.edu/blanton/kcorrect/kcorrect_help.html#SDSS_KCORRECT
        """
        def abs_mag(z, m, kcorr):
            #lum_dist is returned in Mpc, as needed for the calculation below.
            lum_dist = cosmo.luminosity_distance(z).value                        
            M = (m - 5. * np.log10(lum_dist) - 25. - kcorr
                 + (z - z_ref) * Q_factor)
            return M
                        
        for fltr in ['u', 'g', 'r', 'i', 'z']:
            self.df['abs_' + fltr] = np.vectorize(abs_mag)(
              self.df['redshift'], self.df['ext_' + fltr],
              self.df['kcorr_' + fltr])        

    def save_output(self):
        fpath = './../INPUT_FILES/Maoz_file/Maoz_processed.csv'
        self.df.to_csv(fpath)

    def perform_run(self):
        self.read_data()
        self.perform_redshift_check()
        self.trim_data()
        self.extinction_correction()
        self.make_Kcorrections()
        self.compute_abs_mags()
        self.save_output()
        
Process_Data()
