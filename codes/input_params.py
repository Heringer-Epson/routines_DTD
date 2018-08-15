#!/usr/bin/env python

import os
import sys
import time
import warnings
import numpy as np
from astropy import units as u

class Input_Parameters(object):
    """
    Code Description
    ----------    
    The user may use this file to set the input parameters and the datasets
    to be used in running the SN rate model under the 'master.py' routine.

    Parameters
    ----------
    filter_1 : ~str
        filter_1 and filter_2 determine the color to be used as
        (filter_2 - filter_1). A list of available filters can be shown by
        calling fsps.list_filters() under run_fsps.py.
   
    filter_2 : ~str
        Same as above.
        
    imf_type : ~str
        Choice of initial mass function for the fsps simulations.
        Accepts: 'Salpeter', 'Chabrier' or 'Kroupa'.

    sfh_type : ~str
        Choice of star formation history for the fsps simulations.
        Accepts: 'exponential' (sfh propto exp(-t/tau)) or
                 'delayed-exponential' (sfh propto t*exp(-t/tau).
        
    Z : ~float
        Choice of metallicity for the fsps simulations.
        Accepts 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010,
                0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049,
                0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0250 or
                0.0300.
    
    t_onset : ~astropy float (unit of time)
        Sets the time past which SN may occur. Determined by the lifetime of
        stars that may contribute to the SN rate.

    t_cutoff : ~astropy float (unit of time)
        Sets the time at which the slope of the DTD may change. Determined by
        theoretical models.        

    Dcolor_min : ~float
        Sets the lower Dcolor limit, below which galaxies are not taken into
        account. Set by the typical Dcolor at which different SFH may not
        converge in the sSNRL models.

    Dcolor_max : ~float
        Dcolor cut to explode (a few) potential outliers that are much redder
        that the red sequence. Originally determined by the uncertainty in
        fitting the red sequence.

    slopes : ~np.array
        Numpy array containing which DTD slopes to use to compute likelihoods.
        This package adopts the same array for slopes pre and post cutoff time.

    tau_list : ~astropy array (unit of time)
        List containing the tau timescales (in yr) for the selected SFH. e.g.:
        tau_list = np.array([1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr

    crtl_fpath : ~str
        Path to the dataset file which contains the information regarding the
        galaxies in the control sample.

    host_fpath : ~str
        Path to the dataset file which contains the information regarding the
        galaxies in the sample of hosts.

    subdir : ~str
        Name of the sub-directory where the outputs will be stored. For
        organization purposes only. 
    """
    def __init__(self, case):

        self.case= case

        self.filter_1 = None
        self.filter_2 = None
        self.imf_type = None
        self.sfh_type = None
        self.tau_list = None
        self.Z = None
        self.t_onset = None
        self.t_cutoff = None
        self.Dcolor_min = None
        self.Dcolor_max = None
        self.slopes = None
        self.crtl_fpath = None
        self.host_fpath = None
        
        self.subdir = None
        self.subdir_fullpath = None
        self.data_dir = None

        #Fontsize for plotting purposes.
        self.fs = 20.

        self.set_params()
                
    def set_params(self):
                        
        if self.case == 'test-case':   
            self.subdir = 'test/'  
            #Uses the same data set as in Heringer+ 2017 and reproduces Fig. 9.

            self.data_dir = './../INPUT_FILES/paper1/'
            self.matching = 'File'

            #Kcorrection
            self.kcorr_type = 'simple'
            self.z_ref = 0.0
            
            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.01)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr
            
            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0.01, 0.2
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 22.2
            self.ext_r_min, self.ext_r_max = 14., 17.77
            self.uERR_max = 1000.
            self.gERR_max = 0.05
            self.rERR_max = 0.05

            #Take into account visibility time.
            self.visibility_flag = True

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
            
            #Figure management.
            self.show_fig = False
            self.save_fig = True        
        
        elif self.case == 'SDSS_gr_paper1':   
            self.subdir = 'paper1/'  
            #Uses the same data set as in Heringer+ 2017 and reproduces Fig. 9.

            self.data_dir = './../INPUT_FILES/paper1/'
            self.matching = 'File'
            #self.matching = 'FileCAS' #For testing of ra and dec only.

            #Kcorrection
            self.kcorr_type = 'simple'
            self.z_ref = 0.0
            
            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.01)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr
            
            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0.01, 0.2
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 22.2
            self.ext_r_min, self.ext_r_max = 14., 17.77
            self.uERR_max = 1000.
            self.gERR_max = 0.2
            self.rERR_max = 0.2

            #Take into account visibility time.
            self.visibility_flag = False

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
            
            #Figure management.
            self.show_fig = False
            self.save_fig = True

        elif self.case == 'SDSS_gr_paper1_test':   
            self.subdir = 'paper1_test/'  
            #Uses the same data set as in Heringer+ 2017 and reproduces Fig. 9.

            self.data_dir = './../INPUT_FILES/paper1/'
            self.matching = 'File'
            #self.matching = 'FileCAS' #For testing of ra and dec only.

            #Kcorrection
            self.kcorr_type = 'simple'
            self.z_ref = 0.0
            
            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.1)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr
            
            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0.01, 0.2
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 22.2
            self.ext_r_min, self.ext_r_max = 14., 17.77
            self.uERR_max = 1000.
            self.gERR_max = 0.2
            self.rERR_max = 0.2

            #Take into account visibility time.
            self.visibility_flag = False

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
            
            #Figure management.
            self.show_fig = False
            self.save_fig = True

        elif self.case == 'SDSS_gr_paper1_vistime':   
            self.subdir = 'paper1_vistime/'  
            #Uses the same data set as in Heringer+ 2017 and reproduces Fig. 9.

            self.data_dir = './../INPUT_FILES/paper1/'
            self.matching = 'File'
            #self.matching = 'FileCAS' #For testing of ra and dec only.

            #Kcorrection
            self.kcorr_type = 'simple'
            self.z_ref = 0.0
            
            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.01)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr
            
            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0.01, 0.2
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 22.2
            self.ext_r_min, self.ext_r_max = 14., 17.77
            self.uERR_max = 1000.
            self.gERR_max = 0.2
            self.rERR_max = 0.2

            #Take into account visibility time.
            self.visibility_flag = True

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
            
            #Figure management.
            self.show_fig = False
            self.save_fig = True

        elif self.case == 'SDSS_gr_paper1_kcorrect':   
            self.subdir = 'paper1_kcorrect/'  
            #Uses the same data set as in Heringer+ 2017 and reproduces Fig. 9.

            self.data_dir = './../INPUT_FILES/paper1/'
            self.matching = 'File'
            #self.matching = 'FileCAS' #For testing of ra and dec only.

            #Kcorrection
            self.kcorr_type = 'complete'
            self.z_ref = 0.0
            
            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.01)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr
            
            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0.01, 0.2
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 22.2
            self.ext_r_min, self.ext_r_max = 14., 17.77
            self.uERR_max = 1000.
            self.gERR_max = 0.2
            self.rERR_max = 0.2

            #Take into account visibility time.
            self.visibility_flag = False

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
            
            #Figure management.
            self.show_fig = False
            self.save_fig = True

        elif self.case == 'SDSS_gr_improved':   
            self.subdir = 'paper1_improved/'  

            self.data_dir = './../INPUT_FILES/paper1/'
            self.matching = 'View'

            #Kcorrection
            self.kcorr_type = 'complete'
            self.z_ref = 0.0
                        
            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.01)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr

            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0.01, 0.2
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 22.2
            self.ext_r_min, self.ext_r_max = 14., 17.77
            self.uERR_max = 1000.
            self.gERR_max = 0.2
            self.rERR_max = 0.2

            #Take into account visibility time.
            self.visibility_flag = True

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
                        
            #Figure management.
            self.show_fig = False
            self.save_fig = True

        elif self.case == 'SDSS_gr_Maoz':   
            self.subdir = 'Maoz_sample/'  
            #Uses the same data set as in Maoz+ 2102.

            self.matching = 'Table'

            self.data_dir = './../INPUT_FILES/Maoz_file/'

            #Kcorrection
            self.kcorr_type = 'complete'
            self.z_ref = 0.0

            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr 
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.01)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr

            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 51., 57.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0., 0.4
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 1000.
            self.ext_r_min, self.ext_r_max = -1000., 1000.
            self.uERR_max = 1000.
            self.gERR_max = 1000.
            self.rERR_max = 1000.

            #Take into account visibility time.
            self.visibility_flag = True

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
                        
            #Figure management.
            self.show_fig = True
            self.save_fig = True

        elif self.case == 'SDSS_gr_Maoz_aspaper1':   
            self.subdir = 'Maoz_paper1/'  
            #Uses the same data set as in Maoz+ 2102.

            self.matching = 'Table'

            self.data_dir = './../INPUT_FILES/Maoz_file/'

            #Kcorrection
            self.kcorr_type = 'complete'
            self.z_ref = 0.0

            #For building Dcolour-rate models.
            self.filter_1 = 'r'
            self.filter_2 = 'g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr 
            self.slopes = np.arange(-3., 0.01, 0.1)
            self.slopes_fine = np.arange(-3., 0.01, 0.01)
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr

            #For sub-selecting data
            self.ra_min, self.ra_max = 360. - 51., 57.
            self.dec_min, self.dec_max = -1.25, 1.25
            self.redshift_min, self.redshift_max = 0.01, 0.2
            self.petroMag_u_min, self.petroMag_u_max = -1000., 1000.
            self.petroMag_g_min, self.petroMag_g_max = -1000., 22.2
            self.ext_r_min, self.ext_r_max = 14., 17.77
            self.uERR_max = 1000.
            self.gERR_max = 0.2
            self.rERR_max = 0.2

            #Take into account visibility time.
            self.visibility_flag = True

            #For fitting the RS.
            self.x_ref = 0.
            self.tol = 2.
            self.slope_guess = -0.0188
            self.intercept_guess = 0.346
            
            #For fitting a gaussian to the RS.
            #Physical.
            self.Dcolor_range = [-0.08, .1]
            self.bin_size = 0.005
            self.bin_range = [-.8, .4]

            #For Selecting the Dcolors accepted to compute likelihoods.
            self.Dcolor_min = -0.4 #Note that Dcolor max is set by the RS fit.
                        
            #Figure management.
            self.show_fig = False
            self.save_fig = True

        else:
            raise ValueError('Case "%s" is not defined.\n\n' %(self.case))            

        #Set relevant variables.
        self.subdir_fullpath = './../OUTPUT_FILES/RUNS/' + self.subdir
