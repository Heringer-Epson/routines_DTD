#!/usr/bin/env python

import os
from input_params import Input_Parameters as class_input
from plot_fsps_quantities import FSPS_Plotter
from plot_ext_CMD import Ext_CMD
from plot_abs_CMD import Abs_CMD
from plot_Dcolor_hist import Plot_Dcolor
from plot_likelihood import Plot_Likelihood
from analyse_vespa import Vespa_Rates
from analyse_vespa_newmethod import Plot_Vespa_2

class Main_Plotter(object):
    """
    Code Description
    ----------    
    TBW
    """
    
    def __init__(self, _inputs):

        #FSPS_Plotter(_inputs)
        #Ext_CMD(_inputs)        
        #Abs_CMD(_inputs)
        #Plot_Dcolor(_inputs)
        #Plot_Likelihood(_inputs)   
        Vespa_Rates(_inputs)     
        #Plot_Vespa_2(_inputs)     

