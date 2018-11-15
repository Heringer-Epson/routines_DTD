#cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.math cimport exp, log, log10, pow
import numpy as np

cpdef double[:] interpolator(double[:] x, double[:] y, double[:] x_target):
    
    cdef int n_target = x_target.shape[0]
    cdef int N = x.shape[0]
    cdef double xl = 0. 
    cdef double xu = 0. 
    cdef double yl = 0. 
    cdef double yu = 0. 
    cdef double slope = 0. 
    cdef double intercept = 0.
    cdef double[:] sSNRL = np.zeros(n_target)
    
    for m in range(n_target):
        if x_target[m] >= x[N - 1]:
            sSNRL[m] = y[N - 1]        
        elif x_target[m] < x[0]:
            sSNRL[m] = 1.E-40     
        else:
            for i in range(N - 1):
               if ((x_target[m] > x[i]) & (x_target[m] <= x[i+1])):
                    xl = x[i]
                    xu = x[i + 1]
                    yl = log10(y[i]) #interpolate in log space.
                    yu = log10(y[i + 1])
                    break
            slope = (yu - yl) / (xu - xl)
            intercept = yu - slope * xu
            sSNRL[m] = pow(10., slope * x_target[m] + intercept)
    return sSNRL


'''
cdef extern from "sSNR.h":
    double conv_exp "conv_exponential_sSNR" (
      double, double, double, double, double, double, double, double, double)

cdef extern from "sSNR.h":
    double conv_delexp "conv_delayed_exponential_sSNR" (
      double, double, double, double, double, double, double, double, double)

cpdef double[:] get_sSNR(
  double[:] tp, double[:] t_target, double tau, double s1,
  double s2, double t_ons, double t_bre, double sfr_norm, double B, int sfr):
    
    cdef int M = t_target.shape[0]
    cdef int N = tp.shape[0]
    cdef double[:] sSNR_out = np.zeros(M)

    #if sfr == 0:
    #    conv_f = conv_exp
    #elif sfr == 1:
    #    conv_f = conv_delexp
        
    for j in range(M):
        if t_target[j] >= t_ons:
            for i in range(N - 1): #Minus one because we need differences.
                if tp[i] < t_target[j]:          
                    yi = conv_exp(tp[i], t_target[j], tau, s1, s2, t_ons, t_bre, sfr_norm, B)
                    yf = conv_exp(tp[i + 1], t_target[j], tau, s1, s2, t_ons, t_bre, sfr_norm, B)
                    sSNR_out[j] += (yf + yi) / 2. * (tp[i + 1] - tp[i]) 
    return sSNR_out
'''

'''
cpdef double compute_L_sSNRL(
  double[:] sSNRL, double[:] Dcolor, double[:] absmag, double[:] redshift,
  long[:] host_cond, int n_ctrl, int n_host):

    cdef int i = 0
    cdef int j = 0
    cdef double[:] SNR_H = np.zeros(n_host)
    cdef double SNR = 0.
    cdef double _lambda = 0.

    cdef double _ln_L
    cdef double _A
    
    cdef double _N_expected =  0.
    
    for i in range(n_ctrl):
        L = pow(10, -0.4 * (absmag[i] - 5.))
        SNR = sSNRL[i] * L
        if host_cond[i]:
            SNR_H[j] = SNR
            j += 1       
        _N_expected += SNR

    _A = n_host / _N_expected
    for j in range(n_host):
        _lambda += log(_A * SNR_H[j])
    _ln_L = - n_host + _lambda
    
    return _ln_L
'''

        
