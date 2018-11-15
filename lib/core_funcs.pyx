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
        
