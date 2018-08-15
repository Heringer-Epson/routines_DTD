import ctypes
from scipy.integrate import quad
import numpy as np

lib = ctypes.CDLL('./DTD.so')


'''
DTD_f = lib.DTD_func
DTD_f.restype = ctypes.c_double
DTD_f.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double,
  ctypes.c_double, ctypes.c_double, ctypes.c_double)
out = DTD_f(2, 1, 1, 1, 0.1, 10)
print out
'''

'''
int_f = lib.inttest
int_f.restype = ctypes.c_double
int_f.argtypes = (ctypes.c_int, ctypes.c_double)
#I = quad(int_f, 0., 1., (5.))
#print I
'''

'''
conv_f = lib.conv_sSNR
int_f.restype = ctypes.c_float
int_f.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                  ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
conv_f(12, 0, 0, 1, 1, 1, 1, 20)
'''

'''
sfr_f = lib.SFR_func
sfr_f.restype = ctypes.c_double
sfr_f.argtypes = (ctypes.c_double, ctypes.c_double)
print sfr_f(1, 2)
'''

'''
lib = ctypes.CDLL('./DTD.so')
int_f = lib.conv_sSNR
int_f.restype = ctypes.c_double
int_f.argtypes = (ctypes.c_int, ctypes.c_double)
int_f(7, args=([10, 1, 1, 1, 1, 0.1, 1]))
'''

