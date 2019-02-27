#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

#cosmo = FlatLambdaCDM(H0=67.04, Om0=0.3183) #H17 - Planck, I think
#cosmo = FlatLambdaCDM(H0=70.5, Om0=1. - 0.726) #VESPA
cosmo = FlatLambdaCDM(H0=70.5, Om0=0.274) #FSPS as installed

#http://www.sdss.org/dr12/algorithms/magnitudes/
b_u, b_g, b_r, b_i, b_z = 1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10
ab_u, ab_g, ab_r, ab_i, ab_z = -0.036, 0.012, 0.010, 0.028, 0.040
quadErr_u, quadErr_g, quadErr_r, quadErr_i, quadErr_z = .05, .02, .02, .02, .03 

def lup2maggy(mag, magErr, b, quadErr, ab_corr):
    """
    This follows the description available in the following codes that are
    part of the KCORRECT package:
    
    sdss_kcorrect.pro
    k_lups2maggies.pro
    k_sdssfix.pro
    k_minerror
    k_abfix
    sdss_to_maggies.pro
    kcorrect.pro
    
    In practice, this ensures three procedures:
    
    1) SDSS photometry is given in luptides (rather than magnitudes) and the
    transformation to maggies (linear scale) is slightly different.
    2) Photometry corrections to the AB system.
    3) A minimum error in each band is added in quadrature.
    
    Note, paper I used (erroneously) a simpler transformation as in the
    function 'mag2maggies' below, which does not perform any of the corrections
    mentioned above. 
    """
    try:
        maggy = 2. * b * np.sinh(-np.log(b) - 0.4 * np.log(10.) * mag)
        maggyErr = (2. * b * np.cosh(-np.log(b) - 0.4 * np.log(10.) * mag)
                    * 0.4 * np.log(10.) * magErr)
        
        invvar = maggyErr**-2.                 

        #Make AB corrections (see k_abfix.pro)
        maggy *= 10.**(-0.4 * ab_corr)
        invvar *= 10.**(-0.8 * ab_corr)

        #Add error in quadratrue.
        factor = 2.5 / np.log(10.)
        err = factor / np.sqrt(invvar) / maggy
        err2 = err**2. + quadErr**2.
        invvar = factor**2. / (maggy**2. * err2)
    
    except:
        maggy, invvar = np.nan, np.nan
    return maggy, invvar

def mag2maggy(mag, magErr):
    """Simple correction from magnitudes to maggies."""
    try:
        maggy = 10.**(-0.4 * mag)
        invvar = 1. / (0.4 * np.log(10.) * maggy * magErr)**2.
    except:
        maggy, invvar = np.nan, np.nan
    return maggy, invvar

class Process_Data(object):
    """
    Description:
    ------------
    This code will 'process' the data in the chosen input file. By process,
    it means to compute extinction corrections, subselect galaxies that
    satisfy given selection criteria, such
    as SDSS g-band error, etc. Then compute K-corrections and compute
    absolute magnitudes. The selection criteria are set in the input_params.py
    file. The input file is also chosen via the input_params.py file, through
    the 'data_dir' and 'matching' attributes. Input files are stored under
    the './../../INPUT_FILES/' directory.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.

    Dependencies:
    -------------
    To compute Kcorrection, the package KCORRECT
    (Blanton & Roweis 2007) and a Python wrapper are required.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/data_absmag.csv
    """
    def __init__(self, _inputs):
        self._inputs = _inputs
        self.df = None
        self.perform_run()

    def read_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_merged.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False, dtype='str')
        keys_str = ['CID', 'IAUName', 'Classification', 'objID', 'specobjID', 'is_host', 'zErr']
        for key in self.df.keys():
            if key not in keys_str:
                self.df[key] = self.df[key].astype(float) 
        self.df['is_host'] = (self.df['is_host'] == 'True')

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
              self.df['petroMag_' + fltr].astype(float)
              - self.df['extinction_' + fltr].astype(float))

    def trim_data(self):
        """Perform data cuts similar to the original publication."""


        if self._inputs.matching is not 'File':
            ra = self.df['ra'].values
            ra_neg, ra_pos = ra[(ra > 200.)] - 360., ra[(ra < 200.)]
            #print 'dec', min(self.df['dec'].values), max(self.df['dec'].values)
            #print 'ra', min(ra_neg), max(ra_pos)
            self.df = self.df[(self.df['ra'] > self._inputs.ra_min)
                              | (self.df['ra'] < self._inputs.ra_max)]
            self.df = self.df[(self.df['dec'] > self._inputs.dec_min)
                              & (self.df['dec'] < self._inputs.dec_max)]
        self.df = self.df[(self.df['z'] >= self._inputs.redshift_min) &
                          (self.df['z'] < self._inputs.redshift_max)]
        self.df = self.df[(self.df['petroMag_u'] >= self._inputs.petroMag_u_min) &
                          (self.df['petroMag_u'] < self._inputs.petroMag_u_max)]
        self.df = self.df[(self.df['petroMag_g'] >= self._inputs.petroMag_g_min) &
                          (self.df['petroMag_g'] < self._inputs.petroMag_g_max)]
        self.df = self.df[(self.df['ext_r'] > self._inputs.ext_r_min) &
                          (self.df['ext_r'] < self._inputs.ext_r_max)]                                  
        self.df = self.df[(self.df['petroMagErr_u'] <= self._inputs.uERR_max)]
        self.df = self.df[(self.df['petroMagErr_g'] <= self._inputs.gERR_max)]
        self.df = self.df[(self.df['petroMagErr_r'] <= self._inputs.rERR_max)]

    def initialize_Kcorrection(self):
        for fltr in ['u', 'g', 'r', 'i', 'z']:
            self.df['kcorr_' + fltr] = np.zeros(len(self.df['z']))  

    def make_Kcorrections(self):
        sys.path.append(os.environ['PY_KCORRECT_DIR'])
        import kcorrect
        """Make the appropriate kcorrections using the package KCORRECT by
        Blanton & Roweis 2007. Code is available at http://kcorrect.org/ (v4_3)
        Note that a Python wrap around is also used; developed by nirinA and
        mantained at https://pypi.org/project/kcorrect_python. (v2017.07.05)         
        """
        kcorrect.load_templates()
        kcorrect.load_filters(f='sdss_filters.dat',
                              band_shift=self._inputs.z_ref)
        
        def compute_kcorrection(redshift, u, g, r, i, z,
                                uErr, gErr, rErr, iErr, zErr):           
            
            if self._inputs.kcorr_type == 'simple':
                _u, _uI = mag2maggy(u, uErr)
                _g, _gI = mag2maggy(g, gErr)
                _r, _rI = mag2maggy(r, rErr)
                _i, _iI = mag2maggy(i, iErr)
                _z, _zI = mag2maggy(z, zErr)
            elif self._inputs.kcorr_type == 'complete':
                _u, _uI = lup2maggy(u, uErr, b_u, quadErr_u, ab_u)
                _g, _gI = lup2maggy(g, gErr, b_g, quadErr_g, ab_g)
                _r, _rI = lup2maggy(r, rErr, b_r, quadErr_r, ab_r)
                _i, _iI = lup2maggy(i, iErr, b_i, quadErr_i, ab_i)
                _z, _zI = lup2maggy(z, zErr, b_z, quadErr_z, ab_z)
                            
            inp_array = np.array(
              [redshift, _u, _g, _r, _i, _z, _uI, _gI, _rI, _iI, _zI])
            out_coeffs = kcorrect.fit_coeffs(inp_array)
            
            rec_maggies = kcorrect.reconstruct_maggies(
              out_coeffs, redshift=self._inputs.z_ref)
            rmaggies = kcorrect.reconstruct_maggies(
              out_coeffs, redshift=redshift)            
            
            u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr = (
              2.5 * np.log10(np.divide(rec_maggies[1:6],rmaggies[1:6])))
            
            return u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr
        
        self.df['kcorr_u'], self.df['kcorr_g'], self.df['kcorr_r'],\
        self.df['kcorr_i'], self.df['kcorr_z'] =\
          np.vectorize(compute_kcorrection)(
          self.df['z'], self.df['ext_u'], self.df['ext_g'], 
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
                 + (z - self._inputs.z_ref) * self._inputs.Q_factor)
            return M
                        
        for fltr in ['u', 'g', 'r', 'i', 'z']:
            self.df['abs_' + fltr] = np.vectorize(abs_mag)(
              self.df['z'], self.df['ext_' + fltr],
              self.df['kcorr_' + fltr])        

    def save_output(self):
        fpath = self._inputs.subdir_fullpath + 'data_absmag.csv'
        self.df.to_csv(fpath)

    def perform_run(self):
        self.read_data()
        self.extinction_correction()
        self.trim_data()
        self.initialize_Kcorrection()
        if self._inputs.kcorr_type is not 'none':
            self.make_Kcorrections()           
        self.compute_abs_mags()
        self.save_output()
