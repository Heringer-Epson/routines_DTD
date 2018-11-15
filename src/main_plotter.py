#!/usr/bin/env python

from util_plots.plot_fsps_quantities import FSPS_Plotter
from util_plots.plot_ext_CMD import Ext_CMD
from util_plots.plot_abs_CMD import Abs_CMD
from util_plots.plot_Dcolor_hist import Plot_Dcolor
from util_plots.plot_likelihood_A_s import Plot_As
from util_plots.plot_likelihood_s1_s2 import Plot_s1s2
from util_plots.plot_likelihood_s1_s2_II import Plot_s1s2_II
from util_plots.plot_Vespa_fit import Plot_Vespa

class Main_Plotter(object):
    """
    Description:
    ------------
    This piece of code works as an interface between master.py and modules
    that produce relevant plots for each each individual run.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
    """    
    def __init__(self, _inputs):
        #FSPS_Plotter(_inputs)
        #Ext_CMD(_inputs)        
        #Abs_CMD(_inputs)
        #Plot_Dcolor(_inputs)
        Plot_As(_inputs)
        try:
            #Plot_As(_inputs)
            print 'asdasd'
            #Plot_s1s2(_inputs)
            #Plot_s1s2_II(_inputs)
        except:
            pass

