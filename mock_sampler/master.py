#!/usr/bin/env python

import sys, os
from astropy import units as u

from generate_galaxies import Galaxy_Sampler
from compute_mock_rates import Mock_Quantities
from get_likelihoods import Get_Likelihoods
from plot_likelihoods import Plot_Likelihoods
from plot_rate_comparison import Plot_Rates
from analytical_check import Analytical_Check

sys.path.append(os.path.join(os.environ['PATH_ssnarl'], 'src'))
from generic_input_pars import Generic_Pars

class Main_Mock(object):
    """
    Code Description
    ----------    
    Create a mock sample of galaxies that is a proxy of the actual SDSS data.
    Use this sample and its known properties to investigate whether the CL and
    SFHR methods are able to recover the input parameters.

    Parameters:
    -----------
    N : ~int
        Number of galaxies to be create.
    A : ~float
        Scale factor (normalization) of the DTD. Used to compute true SN rates.  
    s : ~float
        Slope of the DTD. Used to compute true SN rates.  

    Notes:
    ------
    The mock sample is not meant to reproduce the actual distribution of colors
    and magnitudes from SDSS. This means that the RS color will be artificial
    (from a FSPS SPS) and thus this exercise cannot test the reliability of
    computing galaxies colors with respect to the red-sequence.
    """
    
    def __init__(self, N, A, s1, s2, survey_t, show_fig, save_fig):
        inputs = Generic_Pars(
          sfh_type='exponential',imf_type='Kroupa',Z='0.0190',t_onset=1.e8*u.yr,
          t_cutoff=1.e9*u.yr, fbhb=0.0, spec_lib='BASEL', isoc_lib='PADOVA')
        
        inputs.show_fig = show_fig
        inputs.save_fig = save_fig
                
        #Galaxy_Sampler(inputs,N)
        #Mock_Quantities(inputs,A,s1,s2,survey_t)
        
        Get_Likelihoods(inputs,A,s1,s2,survey_t)
        Plot_Likelihoods(inputs,A,s1,s2,survey_t)
        Plot_Rates(inputs,A,s1,s2,survey_t)
        #Analytical_Check(inputs,A,s1,s2,survey_t)
            
if __name__ == '__main__':
    #Main_Mock(10000, 1.e-12, -1., -1., 1. * u.yr, False, True)
    Main_Mock(10000, 1.e-12, -1.2, -1.2, 1. * u.yr, False, True)
