#!/usr/bin/env python
import numpy as np
import pandas as pd
import survey_efficiency
from scipy.integrate import simps
from scipy.interpolate import interp1d

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

def integ_DTD(s, to, ti, tf):
    return (tf**(s + 1.) - ti**(s + 1.)) / ((s + 1.) * (tf - ti)) 

def binned_DTD_rate(s1, s2, t_ons, tc):
    """Computes the average SN rate in each time bin."""
    #Assumes that t_cutoff happens in this bin. True for most sensible cases.
    #t_cutoff usually tested in the range 0.5--2 Gyr.
    B = tc**(s1 - s2)
    psi1 = integ_DTD(s1, t_ons, t_ons, 0.42)
    psi2 = integ_DTD(s1, t_ons, 0.42, tc) + B * integ_DTD(s2, t_ons, tc, 2.4)
    psi3 = B * integ_DTD(s2, t_ons, 2.4, 14.)   
    return psi1, psi2, psi3

def compute_L_from_DTDs(_s1, _s2, _t0, _tc, mass1, mass2, mass3, redshift,
                        host_cond, N_obs, visibility_flag):
    """Compute likelihoods given a DTD parametrized by a multiplicative
    constant and a continuous slope. Time bins are as in M12, with the
    exception that 't0' (in Gyr) sets the arbirtrary starting age of the
    first bin."""
    psi1, psi2, psi3 = binned_DTD_rate(_s1, _s2, _t0, _tc)
    rate = mass1 * psi1 + mass2 * psi2 + mass3 * psi3
    if visibility_flag:
        vistime = survey_efficiency.visibility_time(redshift) 
        detect_eff = survey_efficiency.detection_eff(redshift)
        correction_factor = np.multiply(vistime,detect_eff)
        rate = np.multiply(rate,correction_factor)
        
    _N_expected = np.sum(rate)
    _A = N_obs / _N_expected
    _lambda = np.log(_A * rate[host_cond])
    _ln_L = - N_obs + np.sum(_lambda)
    return _A, _ln_L

def compute_L_using_sSNRL(sSNRL, Dcolor, absmag, redshift,
                          host_cond, N_obs, visibility_flag):

    def get_SN_rate(sSNRL, _Dcolor, _absmag, _redshift):
        L = 10.**(-0.4 * (_absmag - 5.))
        SNR = np.multiply(sSNRL,L)
        if visibility_flag:
            vistime = survey_efficiency.visibility_time(_redshift) 
            detect_eff = survey_efficiency.detection_eff(_redshift)
            correction_factor = np.multiply(vistime,detect_eff)
            SNR = np.multiply(SNR,correction_factor)
        return SNR
    
    SNR_all = get_SN_rate(sSNRL, Dcolor, absmag, redshift) #Whole sample.
    SNR_host = get_SN_rate(
      sSNRL[host_cond], Dcolor[host_cond], absmag[host_cond], redshift[host_cond]) #hosts.

    #Normalized rates. Note that un-normalized likelihoods can be analytically
    #derived from these.
    _N_expected = np.sum(SNR_all)
    _A = N_obs / _N_expected
    _lambda = np.log(_A * SNR_host)
    _ln_L = - N_obs + np.sum(_lambda)

    return _A, _ln_L

#Tools for plotting likelihood contours.
def read_lnL(_fpath):
    with open(_fpath, 'r') as inp:
        N_obs = float(inp.readline().split('=')[-1])    
    df = pd.read_csv(_fpath, header=1, dtype=float)
    return (N_obs, df['s1'].values, df['s2'].values, df['norm_A'].values,
            df['ln_L'].values)

def make_A_s_space(N_obs, s1, s2, A, ln_L):
    A_2D, s_2D, ln_L_2D = [], [], []
    cond = (abs(s1 - s2) < 1.e-6)
    #A_target = np.logspace(-12.5, -10., len(s1[cond]))
    A_target = np.logspace(-13.5, -11.5, len(s1[cond]))
        
    for (_s,_A,_ln_L) in zip(s1[cond],A[cond],ln_L[cond]):
        for _At in A_target:
            _f = _At / _A
            s_2D.append(_s)
            A_2D.append(_At)
            ln_L_2D.append(_ln_L + N_obs * (1. - _f + np.log(_f)))
    return np.array(A_2D), np.array(s_2D), np.array(ln_L_2D) 

def plot_contour(ax, x, y, z, c, label='', ao=0., ls=None, add_max=True):
    contour_list = [0.95, 0.68, 0.] 
    z = clean_array(z)
    _x, _y = np.unique(x), np.unique(y)       
    X = x.reshape(len(_x),len(_y))
    Y = y.reshape(len(_x),len(_y))
    qtty = z.reshape((len(_x), len(_y)))
    levels = [get_contour_levels(z, contour) for contour in contour_list]
    ax.contourf(X, Y, qtty, levels[0:2], colors=c, alpha=0.4 + ao)	 
    cs = ax.contourf(X, Y, qtty, levels[1:3], colors=c, alpha=0.6 + ao)	 

    x_best = x[np.argmax(z)]
    y_best = y[np.argmax(z)]

    if add_max:
        ax.plot(x_best, y_best, ls='None', marker='o',color=c, markersize=6.,
        alpha=0.6 + ao, fillstyle='none')
    ax.plot([np.nan], [np.nan], color=c, ls='-', lw=15., marker='None',
            label=label)

    #Estimate parameter uncertainty from ellipsis path.
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    x_cont, y_cont = v[:,0], v[:,1]    
    x_unc = (max(x_cont) - x_best, x_best - min(x_cont))
    y_unc = (max(y_cont) - y_best, y_best - min(y_cont))
    return x_best, y_best, x_unc, y_unc

def compute_rates_using_L(psi, masses, redshift, host_cond, visibility_flag):
    """Attempt to reproduce the method in M12. i.e. given the binned masses,
    compute the most likely rates that would explain the data. Then fit a 
    power-law-like DTD through the most likely rates. 
    """
    rate = masses[0] * psi[0] + masses[1] * psi[1] + masses[2] * psi[2]
    if visibility_flag:
        vistime = survey_efficiency.visibility_time(redshift) 
        detect_eff = survey_efficiency.detection_eff(redshift)
        correction_factor = np.multiply(vistime,detect_eff)
        rate = np.multiply(rate,correction_factor)
    _ln_L = - np.sum(rate) + np.sum(np.log(rate[host_cond]))
    return _ln_L

def compute_curvature(j, k, psi, masses, redshift, host_cond, visibility_flag):

    rate = masses[0] * psi[0] + masses[1] * psi[1] + masses[2] * psi[2]
    n = host_cond.astype(float)
    if visibility_flag:
        vistime = survey_efficiency.visibility_time(redshift) 
        detect_eff = survey_efficiency.detection_eff(redshift)
        correction_factor = np.multiply(vistime,detect_eff)
        rate = np.multiply(rate,correction_factor)
    
    curv = 0.
    for i in range(len(redshift)):
        #Note that M12 has a typoin Eq. (7): not t**2, but (e*t)**2
        #curv += vistime[i]**2. * (n[i] / rate[i] - 1.)**2. * masses[j][i] * masses[k][i]
        curv += correction_factor[i]**2. * (n[i] / rate[i] - 1.)**2. * masses[j][i] * masses[k][i]
    return curv
