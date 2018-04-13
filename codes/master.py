#!/usr/bin/env python

import os
from run_fsps import Make_FSPS
from compute_likelihood import Get_Likelihood

from input_params import Input_Parameters as class_input
        
class Master(object):
    """This code performs three tasks:
    1) If run_fsps_flag is True:
         make synthetic stellar population files by calling fsps.
         --This option requires fsps and python-fsps to be installed.
       Else:
         Use pre-made stellar population files for the particular case where
         filter_1='sdss_r', filter_2='sdss_g', imf_type='Chabrier' and Z=0.0190.
         sfh_type may still be passed as 'exponential' or 'delayed-exponential'.
    
    
    2) Computes the likelihood of parametrized DTD for given dataset. This is
       done through the compute_likelihood routine, which in turn calls the
       following routines: Dcolor2sSNRL |-> SN_rate_gen |-> DTD_gen.
    
    
    3) Generates Figures, which includes the likelihood probability space and
       a useful collection of panels.
    """
    
    def __init__(self, case, run_fsps_flag, likelihood_flag):

        self.inputs = class_input(case=case)
        self.run_fsps_flag = run_fsps_flag
        self.likelihood_flag = likelihood_flag

    def verbose(self):
        os.system('clear')
        print '\n\n****************** SN RATE ANALYSIS *****************\n'                 
        print 'MAKE FSPS FILES------->', self.run_fsps_flag
        print 'COMPUTE LIKELIHOODS------->', self.likelihood_flag
        print '\n\n'

    def run_master(self):
        self.verbose()
        
        fsps_maker  = Make_FSPS(self.inputs)
        if self.run_fsps_flag:
            fsps_maker.run_fsps()
        else:
            fsps_maker.copy_premade_files()

        if self.likelihood_flag:
            Get_Likelihood(self.inputs)
                    
if __name__ == '__main__':
    Master(case='SDSS_gr_default', run_fsps_flag=False,
           likelihood_flag=False).run_master()

