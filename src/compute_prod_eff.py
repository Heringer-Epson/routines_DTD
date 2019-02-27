#!/usr/bin/env python

import sys, os
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import stats

def v2str(v):
    return str(format(v, '.4f'))

def calculate_PE(_A,_s,_to):
    return _A / (_s + 1.) * (13.7**(_s + 1.) - _to**(_s + 1.)) * 1.e9

def assess_PE(fpath,to):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    N_obs, s1, s2, A, ln_L = stats.read_lnL(fpath)    
    _A, _s, _ln_L = stats.make_A_s_space(N_obs, s1, s2, A, ln_L)
    _s[abs(_s + 1.) < 1.e-6] = -0.9999
    L = stats.clean_array(ln_L)
    PE = calculate_PE(_A,_s,to)
    ns = len(np.unique(_s))
    X, Y, XErr, YErr = stats.plot_contour(ax, PE, PE, _ln_L, 'b', ns, ns, 'test')
    plt.close()
    return X, XErr

class Get_Prodeff(object):
    """
    Description:
    ------------
    This code computes the production efficiency and the corresponding
    uncertainty.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihoods/sSNRL_prod_eff.csv'
    """       
    def __init__(self, _inputs):
        to = _inputs.t_onset.to(u.Gyr).value
        add_vespa = 'M12' in _inputs.case.split('_')
        out = open(
          _inputs.subdir_fullpath + 'likelihoods/production_eff.csv', 'w')
        out.write('method,PE,PE_unc_high,PE_unc_low')
        fpath = _inputs.subdir_fullpath + 'likelihoods/sSNRL_s1_s2.csv'
        PE, PE_unc = assess_PE(fpath,to)
        out.write('\nsSNRL,' + v2str(PE) + ',' + v2str(PE_unc[0])
                  + ',' + v2str(PE_unc[1]))
        if add_vespa:
            fpath = _inputs.subdir_fullpath + 'likelihoods/vespa_s1_s2.csv'
            PE, PE_unc = assess_PE(fpath,to)
            out.write('\nvespa,' + v2str(PE) + ',' + v2str(PE_unc[0])
                      + ',' + v2str(PE_unc[1]))
        out.close()
        
