import os, shutil
import numpy as np
from astropy import units as u

class Generic_Pars(object):
    """
    Description:
    ------------
    Define a set of input parameters to use to make the Dcolor vs SN rate plot
    in the class below. This is intended to replicate the ./../src/ code
    input_params.py, but only containing the relevant quantities for this plot.

    Parameters:
    -----------
    As described in ./../src/input_params.py
    """  
    def __init__(self, sfh_type):

        self.sfh_type = sfh_type

        self.filter_1 = 'r'
        self.filter_2 = 'g'
        self.spec_lib = 'BASEL'
        self.isoc_lib = 'PADOVA'
        self.imf_type = 'Kroupa'
        self.Z = '0.0190'
        self.fhbh = 0.0
        self.t_cutoff = 1.e9 * u.yr
        self.t_onset = 1.e8 * u.yr     
        self.tau_list = np.array(
          [1., 1.5, 2., 3., 4., 5., 7., 10.]) * 1.e9 * u.yr
        self.subdir_fullpath = './'
        
        self.copy_fsps()

    def copy_fsps(self):
        inppath = ('./../INPUT_FILES/fsps_FILES/Kroupa_' + self.sfh_type
                   + '_0.0190_0.0_BASEL_PADOVA/')
        tgtpath = './fsps_FILES/'
        if os.path.isdir(tgtpath):
            shutil.rmtree(tgtpath)
        shutil.copytree(inppath, tgtpath)

    def clean_fsps_files(self):
        shutil.rmtree('./fsps_FILES/')
        
