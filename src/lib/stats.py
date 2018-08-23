#!/usr/bin/env python
import numpy as np
import pandas as pd
import survey_efficiency

"""
Set of functions which can be used to calculate likelihoods.

References:
-----------
Maoz+ 2012 (M12): http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
Gao & Pritchet 2013 (G13): http://adsabs.harvard.edu/abs/2013AJ....145...83G
Heringer+ 2017 (H17): http://adsabs.harvard.edu/abs/2017ApJ...834...15H
"""

def clean_array(inp_array):
    """Limit the number of magnitudes (to 20 orders below maximum) in the input
    array and then normalize it. This is helpful for dealing with likelihoods
    whose log spans ~1000s of mags. Input should be an array of the natural
    log of the values in interest. output array is in linear scale.
    """
    inp_array[inp_array < max(inp_array) -20.] = max(inp_array) - 20.     
    inp_array = inp_array - min(inp_array) 
    inp_array = np.exp(inp_array)
    inp_array = inp_array / sum(inp_array)
    return inp_array 

def get_contour_levels(inp_array, contour):
    """Given an input array that is normalized (i.e. cumulative histogram adds
    to one), return the values in that array that correspond to the confidence
    limits passed in 'contour'.
    """
    _L_hist = sorted(inp_array, reverse=True)
    _L_hist_cum = np.cumsum(_L_hist)

    _L_hist_diff = [abs(value - contour) for value in _L_hist_cum]
    diff_at_contour, idx_at_contour = min((val,idx) for (idx,val)
                                          in enumerate(_L_hist_diff))
    #Check if contour placement is too coarse (>10% level).
    if diff_at_contour > 0.1:
        UserWarning(str(contour * 100.) + '% contour not constrained.')	
    return _L_hist[idx_at_contour]

def binned_DTD_rate(A, s, t0):
    """Computes the average SN rate in each time bin."""
    psi1 = A / (s + 1.) * (0.42**(s + 1.) - t0**(s + 1.)) / (0.42 - t0)
    psi2 = A / (s + 1.) * (2.4**(s + 1.) - 0.42**(s + 1.)) / (2.4 - 0.42)
    psi3 = A / (s + 1.) * (14.**(s + 1.) - 2.4**(s + 1.)) / (14. - 2.4)    
    return psi1, psi2, psi3

def compute_L_from_DTDs(_A, _s, t0, mass1, mass2, mass3, redshift, host_cond,
                        visibility_flag):
    """Compute likelihoods given a DTD parametrized by a multiplicative
    constant and a continuous slope. Time bins are as in M12, with the
    exception that 't0' (in Gyr) sets the arbirtrary starting age of the
    first bin."""
    psi1, psi2, psi3 = binned_DTD_rate(_A,_s, t0)
    rate = mass1 * psi1 + mass2 * psi2 + mass3 * psi3
    if visibility_flag:
        vistime = survey_efficiency.visibility_time(redshift) 
        detect_eff = survey_efficiency.detection_eff(redshift)
        correction_factor = np.multiply(vistime,detect_eff)
        rate = np.multiply(rate,correction_factor)
    ln_L = -np.sum(rate) + np.sum(np.log(rate[host_cond]))
    return ln_L

def compute_L_using_sSNRL(Dcolor2sSNRL_func, Dcolor, absmag, redshift,
                          host_cond, N_obs, visibility_flag, normalize_flag):

    def get_SN_rate(_Dcolor, _absmag, _redshift):
        sSNRL = Dcolor2sSNRL_func(_Dcolor)
        L = 10.**(-0.4 * (_absmag - 5.))
        SNR = np.multiply(sSNRL,L)
        if visibility_flag:
            vistime = survey_efficiency.visibility_time(_redshift) 
            detect_eff = survey_efficiency.detection_eff(_redshift)
            correction_factor = np.multiply(vistime,detect_eff)
            SNR = np.multiply(SNR,correction_factor)
        return SNR
    
    SNR_all = get_SN_rate(Dcolor, absmag, redshift) #Whole sample.
    SNR_host = get_SN_rate(
      Dcolor[host_cond], absmag[host_cond], redshift[host_cond]) #hosts.

    if normalize_flag:
        _N_expected = np.sum(SNR_all)
        A = N_obs / _N_expected
        _lambda = np.log(A * SNR_host)
        _ln_L = - N_obs + np.sum(_lambda)
    else:
        _N_expected = np.sum(SNR_all)
        _ln_L = (-_N_expected - np.sum(SNR_host)
                 + np.sum(np.log(SNR_host)))

    return _N_expected, _ln_L

#Tools for plotting likelihood contours.
def read_lnL(_fpath, colx, coly, colz):
    df = pd.read_csv(_fpath, header=0, dtype=float)
    return df[colx].values, df[coly].values, df[colz].values

def plot_contour(ax, x, y, z, c, label='', ao=0., ls=None, add_max=True):
    contour_list = [0.95, 0.68, 0.] 
    z = clean_array(z)
    _x, _y = np.unique(x), np.unique(y)       
    X = x.reshape(len(_x),len(_y))
    Y = y.reshape(len(_x),len(_y))
    qtty = z.reshape((len(_x), len(_y)))
    levels = [get_contour_levels(z, contour) for contour in contour_list]
    ax.contourf(X, Y, qtty, levels[0:2], colors=c, alpha=0.4 + ao)	 
    ax.contourf(X, Y, qtty, levels[1:3], colors=c, alpha=0.6 + ao)	 

    if add_max:
        ax.plot(
          x[np.argmax(z)], y[np.argmax(z)], ls='None', marker='+',
          color=c, markersize=30.)
    ax.plot([np.nan], [np.nan], color=c, ls='-', lw=15., marker='None',
      label=label)
