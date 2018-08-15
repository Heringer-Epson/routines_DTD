#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u

mag_range = 20
contour_list = [0.95, 0.68, 0.] 
t0 = 0.1 #in Gyr.
fs = 24.
c = ['slateblue', 'orangered', 'limegreen']

def visibility_time(redshift): 
    survey_duration = 269. * u.day
    survey_duration = survey_duration.to(u.year).value
    vistime = np.ones(len(redshift)) * survey_duration        
    vistime = np.divide(vistime,(1. + redshift)) #In the galaxy rest frame.
    return vistime

def detection_eff(redshift):
    if redshift < 0.175:
        detection_eff = 0.72
    else:
        detection_eff = -3.2 * redshift + 1.28
    return detection_eff 

def binned_DTD_rate(A, s):
    """Computes the average SN rate in each time bin."""
    psi1 = A / (s + 1.) * (0.42**(s + 1.) - t0**(s + 1.)) / (0.42 - t0)
    psi2 = A / (s + 1.) * (2.4**(s + 1.) - 0.42**(s + 1.)) / (2.4 - 0.42)
    psi3 = A / (s + 1.) * (14.**(s + 1.) - 2.4**(s + 1.)) / (14. - 2.4)    
    return psi1, psi2, psi3

def clean_normalize_array(inp_array, orders):
    """Limit the number of magnitudes in an array and then normalize it.
    This is helpful for dealing with likelihoods whose log spans ~1000s of mags.
    """
    inp_array[inp_array < max(inp_array) - mag_range] = max(inp_array) - mag_range      
    inp_array = inp_array - min(inp_array) 
    inp_array = np.exp(inp_array)
    inp_array = inp_array / sum(inp_array)
    return inp_array 

def get_contour_levels(inp_array, contour):
    """Note that inp_array needs to be normalized so that cum histogram is norm.
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

def plot_contour(ax, x, y, z, color, label):
    x_min, x_max = np.log10(min(x)), np.log10(max(x))
    y_min, y_max = min(y), max(y)

    X, Y = np.meshgrid(x, y)		
    qtty = np.reshape(z, (len(x), len(y)))
    levels = [get_contour_levels(z, contour) for contour in contour_list]

    ax.contourf(
      qtty, levels[0:2], origin='lower', colors=color, alpha=0.4, 
      extent=[x_min, x_max, y_min, y_max], zorder=5)	
    ax.contourf(
      qtty, levels[1:3], origin='lower', colors=color, alpha=0.75, 
      extent=[x_min, x_max, y_min, y_max], zorder=5)	     
    ax.plot([np.nan], [np.nan], color=color, ls='-', lw=15., marker='None',
      label=label)

class Make_Contours(object):
    
    def __init__(self, _inputs):

        self._inputs = _inputs
        self.vistime = None
        self.eff_corr = None
        self.eff_vistime = None
        self.opt_DTD = None
        self.DTDErr = None
        self.A = None
        self.s = None
        self.s_unc = None
        self.L = None
        self.sSNRL_A = None
        self.sSNRL_s = None

        self.n_bins = 50
        self.A_smp = np.logspace(-13., -12.4, self.n_bins)
        self.s_smp = np.linspace(-1.6, -.8, self.n_bins)

        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.add_vespa = 'Maoz' in self._inputs.case.split('_')       
        #self.add_vespa = False
   
        self.run_plot()

    def set_fig_frame(self):
        x_label = r'$\mathrm{log}\, A$'
        y_label = r'$s$'
        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xlim(-13., -12.2)
        self.ax.set_ylim(-2.5, 0.)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params(
          'both', length=8, width=1., which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1., which='minor', direction='in')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')

    def read_sSNRL_data(self):
        fpath = self._inputs.subdir_fullpath + 'likelihood_A_s.csv'
        #fpath = './../OUTPUT_FILES/RUNS/paper1_test/likelihood_A_s.csv'
        self.sSNRL_A, self.sSNRL_s, self.sSNRL_ln_L = np.loadtxt(
          fpath, delimiter=',', skiprows=7, usecols=(0,1,3), unpack=True)          

        self.sSNRL_A = np.unique(self.sSNRL_A)
        self.sSNRL_s = np.unique(self.sSNRL_s)
        self.sSNRL_ln_L = clean_normalize_array(self.sSNRL_ln_L, mag_range)        

    def retrieve_vespa_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0)
        self.f1, self.f2 = self._inputs.filter_1, self._inputs.filter_2
        photo1 = self.df['abs_' + self.f1]
        photo2 = self.df['abs_' + self.f2]
        Dcolor = self.df['Dcolor_' + self.f2 + self.f1]

        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
        #self.df = self.df[(Dcolor  >= -10. * RS_std)
        #                  & (Dcolor <= 2. * RS_std)]

        self.host_cond = self.df['n_SN'].values
        self.vistime = visibility_time(self.df['z'].values)
        self.eff_corr = np.vectorize(detection_eff)(self.df['z'].values)
        self.eff_vistime = np.multiply(self.vistime,self.eff_corr)
        
        self.mass1 = (self.df['vespa1'].values + self.df['vespa2'].values) * .55
        self.mass2 = self.df['vespa3'].values * .55
        self.mass3 = self.df['vespa4'].values * .55

    def compute_L_from_DTDs(self, _A, _s):
        psi1, psi2, psi3 = binned_DTD_rate(_A,_s)
        rate = self.mass1 * psi1 + self.mass2 * psi2 + self.mass3 * psi3
        lmbd = np.multiply(rate,self.eff_vistime)
        ln_L = -np.sum(lmbd) + np.sum(np.log(lmbd[self.host_cond]))
        return ln_L
    
    def determine_most_likely_DTD(self):
        stacked_grid = np.vstack(np.meshgrid(self.A_smp,self.s_smp)).reshape(2,-1).T
        _L = []
        for psis in stacked_grid:
            _L.append(np.vectorize(self.compute_L_from_DTDs)(psis[0], psis[1]))
        self.L = np.asarray(_L)        
        self.opt_DTD = stacked_grid[self.L.argmax()]        
        self.L = clean_normalize_array(self.L, mag_range)                

    def plot_Maoz_result(self):
        self.ax.axhspan(-1.07 + 0.07, -1.07 - 0.07, alpha=0.5, color='gray')
        self.ax.plot([np.nan], [np.nan], color='gray', ls='-', lw=15., alpha=0.5,
                     marker='None', label=r'Maoz${\it \, et\, al}$ (2012)')
                
    def plot_contours(self):
        plot_contour(self.ax, self.sSNRL_A, self.sSNRL_s, self.sSNRL_ln_L,
                     c[0], r'$sSNR_L$ method')
        if self.add_vespa:
            plot_contour(self.ax, self.A_smp, self.s_smp, self.L,
                         c[1], r'Vespa method')
        self.ax.legend(
          frameon=False, fontsize=fs, numpoints=1, ncol=1,
          loc=2, labelspacing=-0.1, handlelength=1.5, handletextpad=.5)            

    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/A-s_grid.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show() 
                
    def run_plot(self):
        self.set_fig_frame()
        self.read_sSNRL_data()
        if self.add_vespa:
            self.retrieve_vespa_data()
            self.determine_most_likely_DTD()
        self.plot_Maoz_result()
        self.plot_contours()
        self.manage_output()             

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Make_Contours(class_input(case='SDSS_gr_Maoz'))
