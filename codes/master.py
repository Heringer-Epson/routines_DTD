#!/usr/bin/env python

import os
import fsps

from input_params import Input_Parameters as class_input
from util_tasks import Utility_Routines
from run_fsps import Make_FSPS
from compute_likelihood import Get_Likelihood
from plot_likelihood import Plot_Likelihood
from plot_several import Make_Panels
        
class Master(object):
    """
    Code Description
    ----------    
    This is the main script to be used to make SN rate models and use them
    to analyze SDSS data. The input parameters are set under input_params.py.

    Parameters
    ----------
    run_fsps_flag : ~boolean
        Flag to determine whether or not to compute new FSPS synthetic stellar
        populations. REQUIRES FSPS AND PYTHON-FSPS TO BE INSTALLED.

    likelihood_flag : ~boolean
        Flag to determine whether or not to compute likelihoods for each of
        the parametrized DTDs. If True, an intensity map of the likelihood is
        also produced.
    
    panels_flag : ~boolean
        Flag to determine whether or not to produce several figures, each
        containing panels showing the relationship between relevant quantities,
        such as age, color, mass and SN rate. Each figure produced here adopts
        a unique parametrization (combination of slopes) of the DTD.    
    """
    
    def __init__(self, case=None, run_fsps_flag=False, likelihood_flag=False,
                 panels_flag=False):
        self.case = case
        self.run_fsps_flag = run_fsps_flag
        self.likelihood_flag = likelihood_flag
        self.panels_flag = panels_flag
        self.inputs = None

    def verbose(self):
        os.system('clear')
        print '\n\n****************** SN RATE ANALYSIS *****************\n'                 
        print 'MAKE FSPS FILES------->', self.run_fsps_flag
        print 'COMPUTE LIKELIHOODS------->', self.likelihood_flag
        print 'GENERATE MODEL FIGURES------->', self.panels_flag
        print '\n\n'

    def list_filters(self):
        os.system('clear')
        print '\n\n****************** FSPS FILTERS *****************\n'  
        print fsps.list_filters(), '\n\n'

    def run_master(self):
        self.verbose()
        self.inputs = class_input(case=self.case)
        Utility_Routines(self.inputs)
        
        fsps_maker = Make_FSPS(self.inputs)
        if self.run_fsps_flag:
            fsps_maker.run_fsps()
        else:
            fsps_maker.copy_premade_files()

        if (self.likelihood_flag and self.inputs.filter_1 == 'sdss_r' and
            self.inputs.filter_2 == 'sdss_g'):
                Get_Likelihood(self.inputs)
                Plot_Likelihood(self.inputs, show_fig=False, save_fig=True)
        
        if self.panels_flag:
            print '\n\n>GENERATING MODEL FIGURES...\n'
            for s2 in self.inputs.slopes[::5]:
                s2_str = str(format(s2, '.1f'))
                for s1 in self.inputs.slopes[::5]:
                    s1_str = str(format(s1, '.1f'))
                    print '  *s1/s2=' + s1_str + '/' + s2_str 
                    Make_Panels(self.inputs, s1, s2, show_fig=False,
                                save_fig=True)
                    
if __name__ == '__main__':
    #Master().list_filters()
    Master(case='SDSS_gr_example1', run_fsps_flag=False,
           likelihood_flag=True, panels_flag=True).run_master()

