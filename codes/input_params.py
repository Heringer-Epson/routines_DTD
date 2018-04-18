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

        self.set_params()
                
    def set_params(self):
                        
        if self.case == 'SDSS_gr_example1':   
            self.subdir = 'example1/'  
            #Uses the same data set as in paper I.
            #http://adsabs.harvard.edu/abs/2017ApJ...834...15H

            data_dir = './../INPUT_FILES/sample_paper-I/'
            self.ctrl_fpath = data_dir + 'spec_sample_data.csv'
            self.host_fpath = data_dir + 'hosts_SDSS_spec.csv'

            self.filter_1 = 'sdss_r'
            self.filter_2 = 'sdss_g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.Dcolor_min = -0.4
            self.Dcolor_max = 0.08   
            self.slopes = np.arange(-3., 0.01, 0.1)
            
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr

        elif self.case == 'SDSS_gr_example2':   
            self.subdir = 'example2/'  
            #Uses the same data set as in paper I.
            #http://adsabs.harvard.edu/abs/2017ApJ...834...15H

            data_dir = './../INPUT_FILES/sample_paper-I/'
            self.ctrl_fpath = data_dir + 'spec_sample_data.csv'
            self.host_fpath = data_dir + 'hosts_SDSS_spec.csv'

            self.filter_1 = 'sdss_r'
            self.filter_2 = 'sdss_g'
            self.imf_type = 'Chabrier'
            self.sfh_type = 'delayed-exponential'
            self.Z = 0.0190
            self.t_onset = 1.e8 * u.yr
            self.t_cutoff = 1.e9 * u.yr
            self.Dcolor_min = -0.4
            self.Dcolor_max = 0.08   
            self.slopes = np.arange(-3., 0.01, 0.1)
            
            self.tau_list = np.array(
              [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr

        else:
            raise ValueError('Case "%s" is not defined.\n\n' %(self.case))            

