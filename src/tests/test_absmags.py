#!/usr/bin/env python

import sys
import kcorrect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

#http://www.sdss.org/dr12/algorithms/magnitudes/
b_u, b_g, b_r, b_i, b_z = 1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10
ab_u, ab_g, ab_r, ab_i, ab_z = -0.036, 0.012, 0.010, 0.028, 0.040
quadErr_u, quadErr_g, quadErr_r, quadErr_i, quadErr_z = .05, .02, .02, .02, .03 

def lup2maggy(mag, magErr, b, quadErr):
    """
    TBW.
    """
    try:
        maggy = 2. * b * np.sinh(-np.log(b) - 0.4 * np.log(10.) * mag)
        maggyErr = (2. * b * np.cosh(-np.log(b) - 0.4 * np.log(10.) * mag)
                    * 0.4 * np.log(10.) * magErr)
        
        invvar = maggyErr**-2.                 

        #Make AB corrections (see k_abfix.pro)
        maggy *= 10.**(-0.4 * ab_corr)
        invvar *= 10.**(-0.8 * ab_corr)

        #Add error in quadratrue.
        factor = 2.5 / np.log(10.)
        err = factor / np.sqrt(invvar) / maggy
        err2 = err**2. + quadErr**2.
        invvar = factor**2. / (maggy**2. * err2)
    
    except:
        maggy, invvar = np.nan, np.nan
    return maggy, invvar

def mag2maggy(mag, magErr):
    try:
        maggy = 10.**(-0.4 * mag)
        invvar = 1. / (0.4 * np.log(10.) * maggy * magErr)**2.
    except:
        maggy, invvar = np.nan, np.nan
    return maggy, invvar

class Compare_Dataset(object):
    
    def __init__(self):
        
        self.df_orig = None
        self.df_new = None
        self.idx = None

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.execute()
        
    def set_fig_frame(self):

        x_label = r'$M_r$'
        #y_label = r'$M_g-M_r$'
        y_label = r'$M_{r,old}-M_{r,new}$'
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xlim(-24., -16.)
        self.ax.set_ylim(-0.2,0.2)
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        self.ax.xaxis.set_minor_locator(MultipleLocator(1.))
        self.ax.xaxis.set_major_locator(MultipleLocator(2.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
        self.ax.yaxis.set_major_locator(MultipleLocator(.1))  
        
    def get_data(self):
        fpath_orig = '/home/heringer/Research/A_2BPMSNIa/data/catalog/SDSS/'\
                     'spec/method_1/SDSS_gr_DEFAULT/Dcolour_gr.csv'
        fpath_new = '/home/heringer/Research/routines_DTD/OUTPUT_FILES/'\
                    'RUNS/paper1/data_Dcolor.csv'
        
        self.df_orig = pd.read_csv(fpath_orig, header=0)
        self.df_new = pd.read_csv(fpath_new, header=0)

    def test_redshift(self):
        z_diff = self.df_orig['redshift'].values - self.df_new['z'].values
        if z_diff[z_diff > 0.0001].size > 0:
            pass
            print 'Warning, objects are not properly matchdd between datasets.'

    def make_plot(self):
        #Dcolor_diff = self.df_orig['r_abs'].values - self.df_new['abs_r'].values
        Dcolor_diff = self.df_orig['Dcolour_gr'].values - self.df_new['Dcolor_gr'].values
        Mr = self.df_orig['r_abs'].values

        self.ax.plot(Mr, Dcolor_diff, ls='None', marker='o', markersize=5.,
                     color='k')

    def execute(self):
        self.set_fig_frame()
        self.get_data()
        self.test_redshift()
        self.make_plot()
        plt.show()

class Compare_Testcases(object):
    
    def __init__(self):
        self.compare_individual()
        
    def compare_individual(self):

        kcorrect.load_templates()
        kcorrect.load_filters(f='sdss_filters.dat', band_shift=0.)

        def compute_kcorrection(redshift, u, g, r, i, z,
                                uErr, gErr, rErr, iErr, zErr):           
            
            _u, _uI = mag2maggy(u, uErr)
            _g, _gI = mag2maggy(g, gErr)
            _r, _rI = mag2maggy(r, rErr)
            _i, _iI = mag2maggy(i, iErr)
            _z, _zI = mag2maggy(z, zErr)
            
            inp_array = np.array(
              [redshift, _u, _g, _r, _i, _z, _uI, _gI, _rI, _iI, _zI])
            out_coeffs = kcorrect.fit_coeffs(inp_array)
            out_kcorrection = kcorrect.reconstruct_maggies(
              out_coeffs, redshift=0.0)

            rec_maggies = kcorrect.reconstruct_maggies(
              out_coeffs, redshift=0.0)
            rmaggies = kcorrect.reconstruct_maggies(
              out_coeffs, redshift=redshift)
            
            u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr = (
              2.5 * np.log10(np.divide(rec_maggies[1:6],rmaggies[1:6])))
            
            return u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr
        
        kcorr_u, kcorr_g, kcorr_r, kcorr_i, kcorr_z =\
          np.vectorize(compute_kcorrection)(
            0.165085, 20.478412, 18.407257, 17.224196, 16.763526, 16.502568,
            0.545784, 0.035066, 0.015088, 0.014665, 0.037173)  
        print '\nTest case 1: (redshift=0.165085)'
        print '  Got: ', kcorr_u, kcorr_g, kcorr_r, kcorr_i, kcorr_z
        print '  Expected:  : 0.710970 0.599360 0.208734 0.114247 0.0747090'

        #Original extinction corrected data for the most discrepant case.
        kcorr_u, kcorr_g, kcorr_r, kcorr_i, kcorr_z =\
          np.vectorize(compute_kcorrection)(
            0.043196, 18.524137, 16.654535, 16.067731, 15.082049, 15.135383,
            0.050584, 0.037669, 0.041673, 0.014656, 0.036576)  
        print '\nTest case 1: (redshift=0.165085)'
        print '  Got: ', kcorr_u, kcorr_g, kcorr_r, kcorr_i, kcorr_z
        print '  Expected: 0.235204 0.105466 0.113530 0.0467658 0.0604789'

        kcorr_u, kcorr_g, kcorr_r, kcorr_i, kcorr_z =\
          np.vectorize(compute_kcorrection)(
            0.125693, 19.524085, 18.139808, 17.577219, 17.307329, 17.414966,
            0.317116, 0.041545, 0.021797, 0.168508, 0.11463)  
        print '\nTest case 1: (redshift=0.165085)'
        print '  Got: ', kcorr_u, kcorr_g, kcorr_r, kcorr_i, kcorr_z
        print '  Expected: 0.394772 0.187751 0.121808 -0.164960 -0.0251995'

Compare_Dataset()
#Compare_Testcases()
