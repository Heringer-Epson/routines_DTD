#!/usr/bin/env python

import itertools
from input_params import Input_Parameters as class_input
from acquire_hosts import Acquire_Hosts
from process_data import Process_Data
from fit_RS import Fit_RS
from compute_likelihood import Get_Likelihood
from compute_prod_eff import Get_Prodeff
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
    data_flag : ~boolean
        Flag to determine whether or not to process observational data (e.g.
        compute K-corrections and absolute magnitudes).
    likelihood_flag : ~boolean
        Flag to determine whether or not to compute likelihoods for each of
        the parametrized DTDs.  
    plots_flag : ~boolean
        Flag to determine whether or not to make standard plots.
    custom_pars : ~tuple
        Tuple containing a set of variables to be set in a given case. The
        sequence of variables needs to be coded in each 'case'. For instance,
        under the case 'custom', this tuple contains
        (ctrl_samp, hosts_samp, host_class, redshift_max). This allows for a
        serie of simulations without needing to defined a series of cases.
        Does not need to be initialized.
    """
    
    def __init__(self, case, data_flag, likelihood_flag,
                 plots_flag, custom_pars=None):
        self.case = case
        self.data_flag = data_flag
        self.likelihood_flag = likelihood_flag
        self.plots_flag = plots_flag
        self.custom_pars = custom_pars
        self.inputs = None

    def run_master(self):
        self.inputs = class_input(case=self.case, custom_pars=self.custom_pars)
        Utility_Routines(self.inputs)
            
        if self.data_flag:
            Acquire_Hosts(self.inputs)
            Process_Data(self.inputs)
            Fit_RS(self.inputs)
        if self.likelihood_flag:
            Get_Likelihood(self.inputs)
            Get_Prodeff(self.inputs)            
        if self.plots_flag:
            Main_Plotter(self.inputs)
        Write_Record(self.inputs)
            
if __name__ == '__main__':
    pd = True
    lf = True
    pf = True

    Master(case='default_test', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()  

    '''
    Master(case='default', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()  
    Master(case='default_40', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master() 
    Master(case='M12_comp', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master() 
    Master(case='H17_updated_model', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master() 



    #Individual tests.
    Master(case='H17', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()
    Master(case='M12', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()    
    Master(case='H17_updated_model', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()    
    Master(case='H17_interpolation', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()           
    Master(case='H17_Table', data_flag=pd,
           likelihood_flag=lf, plots_flag=pf).run_master()    

    #Series of runs to analyse systematic uncertainties. Those use
    #fiducial parameters of 'H17', 'S18' (z)SN Ia and redshit_max=0.2.
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('0.0', '100','1','exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '40','1','exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '70','1','exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','delayed-exponential','Kroupa','0.0190',0.0,'BASEL','PADOVA')).run_master()                                                           
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Chabrier','0.0190',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Salpeter','0.0190',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Kroupa','0.0150',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Kroupa','0.0300',0.0,'BASEL','PADOVA')).run_master()  
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Kroupa','0.0190',0.1,'BASEL','PADOVA')).run_master()     
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Kroupa','0.0190',0.2,'BASEL','PADOVA')).run_master() 
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Kroupa','0.0190',0.0,'BASEL','MIST')).run_master() 
    Master(
      case='sys', data_flag=pd, likelihood_flag=lf, plots_flag=pf,
      custom_pars=('1.6', '100','1','exponential','Kroupa','0.0190',0.0,'MILES','PADOVA')).run_master()     

    #RUN several simulations for a suite of relevant parameters.
    ctrl = ['H17', 'M12']
    SN = ['native', 'S18']
    SN_type = [['SNIa'], ['SNIa', 'zSNIa']]
    z = ['0.2', '0.4']
    
    all_cases = list(itertools.product(ctrl, SN, SN_type, z))
    
    for i, _pars in enumerate([all_cases[0]]):
        print 'Running simulation ' + str(i + 1) + '/' + str(len(all_cases))
        print _pars
        Master(case='custom', data_flag=pd,
               likelihood_flag=lf, plots_flag=pf, custom_pars=_pars
               ).run_master()    
    '''
