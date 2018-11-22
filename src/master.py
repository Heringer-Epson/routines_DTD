#!/usr/bin/env python
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import itertools
from input_params import Input_Parameters as class_input
from run_fsps import Make_FSPS
from acquire_hosts import Acquire_Hosts
from process_data import Process_Data
from fit_RS import Fit_RS
from compute_likelihood import Get_Likelihood
from main_plotter import Main_Plotter
from write_record import Write_Record
from util_tasks import Utility_Routines

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
    """
    
    def __init__(self, case, run_fsps_flag, process_data, likelihood_flag,
                 plots_flag, custom_pars=None):
        self.case = case
        self.run_fsps_flag = run_fsps_flag
        self.process_data = process_data
        self.likelihood_flag = likelihood_flag
        self.plots_flag = plots_flag
        self.custom_pars = custom_pars
        self.inputs = None

    def run_master(self):
        self.inputs = class_input(case=self.case, custom_pars=self.custom_pars)
        Utility_Routines(self.inputs)
        
        #Get FSPS files to build Dcolour-rate models.
        fsps_maker = Make_FSPS(self.inputs)
        if self.run_fsps_flag:
            fsps_maker.run_fsps()
        else:
            fsps_maker.copy_premade_files()
            
        if self.process_data:
            Acquire_Hosts(self.inputs)
            Process_Data(self.inputs)
            Fit_RS(self.inputs)
        if self.likelihood_flag:
            Get_Likelihood(self.inputs)            
        if self.plots_flag:
            Main_Plotter(self.inputs)
        Write_Record(self.inputs)
            
if __name__ == '__main__':
    rf = False
    pd = False
    lf = True
    pf = True  

    Master(case='H17', run_fsps_flag=rf, process_data=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()  

    #Individual tests.
    #Master(case='H17', run_fsps_flag=rf, process_data=pd,
    #       likelihood_flag=lf, plots_flag=pf).run_master()
    #Master(case='M12', run_fsps_flag=rf, process_data=pd,
    #       likelihood_flag=lf, plots_flag=pf).run_master()    
    #Master(case='H17_updated_model', run_fsps_flag=rf, process_data=pd,
    #       likelihood_flag=lf, plots_flag=pf).run_master()    
    #Master(case='H17_interpolation', run_fsps_flag=rf, process_data=pd,
    #       likelihood_flag=lf, plots_flag=pf).run_master()           
    '''
    Master(case='H17_Table', run_fsps_flag=rf, process_data=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()    

    #RUN several simulations for a suite of relevant parameters.
    ctrl = ['H17', 'M12']
    SN = ['native', 'S18']
    SN_type = [['SNIa'], ['SNIa', 'zSNIa']]
    z = ['0.2', '0.4']
    
    all_cases = list(itertools.product(ctrl, SN, SN_type, z))
    
    for i, _pars in enumerate(all_cases):
        print 'Running simulation ' + str(i + 1) + '/' + str(len(all_cases))
        Master(case='custom', run_fsps_flag=rf, process_data=pd,
               likelihood_flag=lf, plots_flag=pf, custom_pars=_pars
               ).run_master()    

    #Series of runs to analyse systematic uncertainties. Those use
    #fiducial parameters of 'H17', 'S18' (z)SN Ia and redshit_max=0.2.
    pd = True
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('40','1','exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('70','1','exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','delayed-exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()                                                           
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Chabrier','0.0190',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Salpeter','0.0190',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Kroupa','0.0150',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Kroupa','0.0300',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Kroupa','0.0190',0.1,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Kroupa','0.0190',0.2,'BASEL','PADOVA')).run_master() 
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Kroupa','0.0190',0.0,'BASEL','MIST')).run_master() 
    Master(
      case='sys', run_fsps_flag=rf, process_data=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('100','1','exponential','Kroupa','0.0190',0.0,'MILES','PADOVA')).run_master()     
    '''
