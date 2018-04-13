#!/usr/bin/env python

import os
import sys
import time
import fsps
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
        (filter_2 - filter_1). A list of filters is available via the
        fsps.list_filters() command.
   
    filter_2 : ~str
        See filter_1.
        
    imf_type : ~str
        Choice of initial mass function for the fsps simulations.
        Accepts: 'Salpeter', 'Chabrier' or 'Kroupa'.

    sfh_type : ~str
        Choice of star formation history for the fsps simulations.
        Accepts: 'exponential' (sfh propto exp(-t/tau)) or
                 'delayed-exponential' (sfh propto t*exp(-t/tau)).
        
    Z : ~float
        Choice of metallicity for the fsps simulations.
        Accepts 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010,
                0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049,
                0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0250 or
                0.0300.
    
    t_onset : ~astropy float (units of yr)
        Sets the time past which SN may occur. Determined by the lifetime of
        stars that may contribute to the SN rate.

    t_cutoff : ~astropy float (units of yr)
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

    crtl_fpath : ~str
        Path to the dataset file which contains the information regarding the
        galaxies in the control sample.

    host_fpath : ~str
        Path to the dataset file which contains the information regarding the
        galaxies in the sample of hosts.

    Output
    -------
    Creates a 'record' file in the top level of the directory where the
    information from the run is stored. This file contains the used inputs.
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
        self.crtl_fpath = None
        self.host_fpath = None
        
        self.subdir = None
        self.subdir_fullpath = None

        self.set_params()
        self.initialize_outfolder()
        self.make_record()
                
    def set_params(self):
                
        if self.case == 'SDSS_gr_default':   
            self.subdir = 'test/'  
            #Uses the same data set as in paper I. ADD ref.

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

            _tau_list = [1., 1.5, 2., 3., 4., 5., 7., 10.]
            self.tau_list = [tau * 1.e9 * u.yr for tau in _tau_list]

        else:
            raise ValueError('Case "%s" is not defined.\n\n' %(self.case))            

    def initialize_outfolder(self):
        """Create the directory if it doesn't already exist and dd a copy of
        this input file in that directory."""
        self.subdir_fullpath = './../OUTPUT_FILES/RUNS/' + self.subdir
        if not os.path.exists(self.subdir_fullpath):
            os.makedirs(self.subdir_fullpath)
        if not os.path.exists(self.subdir_fullpath + 'fsps_FILES/'):
            os.makedirs(self.subdir_fullpath + 'fsps_FILES/')        
        if not os.path.exists(self.subdir_fullpath + 'FIGURES/'):
            os.makedirs(self.subdir_fullpath + 'FIGURES/')

    def make_record(self):
        fpath = self.subdir_fullpath + 'record.dat'
        with open(fpath, 'w') as rec:
            rec.write('run date: ' + time.strftime('%d/%m/%Y') + '\n')
            rec.write('FSPS version: ' + str(fsps.__version__) + '\n\n')
            rec.write('---Model params---\n')
            rec.write('Colour: ' + self.filter_2 + '-' + self.filter_1 + '\n')
            rec.write('IMF: ' + self.imf_type + '\n')
            rec.write('SFH: ' + self.sfh_type + '\n')
            rec.write('Metallicity: ' + str(self.Z) + '\n')
            rec.write('t_onset: ' + str(format(self.t_onset.to(u.yr).value
                       / 1.e9, '.2f')) + ' Gyr \n')
            rec.write('t_cutoff: ' + str(format(self.t_cutoff.to(u.yr).value
                       / 1.e9, '.2f')) + ' Gyr \n')
            rec.write('Dcolor_min: ' + str(self.Dcolor_min) + ' \n')
            rec.write('Dcolor_max: ' + str(self.Dcolor_max) + ' \n\n')
            rec.write('---Dataset---\n')
            rec.write('Control galaxies: ' + self.ctrl_fpath + ' \n')
            rec.write('Host galaxies: ' + self.host_fpath)
