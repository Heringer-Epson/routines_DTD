#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib import colors
from astropy import units as u
from SN_rate_gen import Model_Rates

class Plot_Likelihood(object):
    """make a plot of the likelihood of a given DTD in the space of slope pre-
    t_break and slope post t_break.
    """
    
    def __init__(self, sfh_type, t_onset, t_break, show_fig=True, save_fig=False):
        """Makes a figure where a set of DTDs is plotted as a function of age.
        """

        self.sfh_type = sfh_type
        self.t_onset = t_onset
        self.t_break = t_break
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)
        self.contour_list = [0.95, 0.68]  
        self.fs = 20.
        
        self.s1 = None
        self.s2 = None
        self.ln_L = None   
        self.L = None
        self.slopes = None
        self.N_s = None
        self.D = {} #Dictionary for contour variables.   
        
        self.make_plot()
        
    def set_fig_frame(self):
        
        x_label = r'$s_1$'
        y_label = r'$s_2$'
        
        self.ax.set_xlabel(x_label, fontsize=self.fs)
        self.ax.set_ylabel(y_label, fontsize=self.fs)
        #self.ax.set_xlim(-3.,0.)      
        #self.ax.set_ylim(-3.,0.)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_off()
        #self.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        #self.ax.xaxis.set_major_locator(MultipleLocator(0.5))
        #self.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        #self.ax.yaxis.set_major_locator(MultipleLocator(0.5))  
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')

    def get_data(self):
        
        directory = './../OUTPUT_FILES/FILES/'
        fname = ('likelihood_' + str(self.t_onset.to(u.yr).value / 1.e9) + '_'\
                 + str(self.t_break.to(u.yr).value / 1.e9) + '_'\
                 + self.sfh_type + '.csv')
        fpath = directory + fname
        
        self.s1, self.s2, self.ln_L = np.loadtxt(
          fpath, delimiter=',', skiprows=9, usecols=(0,1,3), unpack=True)           

        #Get how many slopes there is self.s1 and self.s2 
        self.slopes = np.unique(self.s2)
        self.N_s = len(self.slopes)
        
    def normalize_likelihood(self):
        _L = np.exp(self.ln_L)
        self.L = _L / sum(_L)
    
    def plot_data(self):

        cmap = plt.get_cmap('Greys')
        
        qtty_max = max(self.L)
        qtty_min = min(self.L)
        s_min = min(self.slopes)
        s_max = max(self.slopes)
        s_step = abs(self.slopes[1] - self.slopes[0])
        s_hs = s_step / 2. #half step
        
        qtty = np.reshape(self.L, (self.N_s, self.N_s))
        
        _im = self.ax.imshow(
          qtty, interpolation='nearest', aspect='auto',
          extent=[s_min - s_hs, s_max + s_hs, s_min - s_hs, s_max + s_hs],
          origin='lower', cmap=cmap,
          norm=colors.Normalize(vmin=qtty_min,  vmax=qtty_max))        

        #Format ticks. Note that the range of values plotted in x (and y) is
        #defined when calling imshow, under the 'extent' argument.
        ticks_pos = np.arange(s_min, self.N_s, s_step)
        ticks_label = [''] * self.N_s
        for i in xrange(0, self.N_s, 5):
            ticks_label[i] = self.slopes[i]

        self.ax.set_xticks(ticks_pos)
        self.ax.set_xticklabels(ticks_label, rotation='vertical')        
        self.ax.set_yticks(ticks_pos)
        self.ax.set_yticklabels(ticks_label, rotation='vertical')

    def find_sigma_fractions(self):
        
        #Note that the cumulative histogram is normalized be self.L is norm.
        _L_hist = sorted(self.L, reverse=True)
        _L_hist_cum = np.cumsum(_L_hist)

        for contour in self.contour_list:
            _L_hist_diff = [abs(value - contour) for value in _L_hist_cum]
            diff_at_contour, idx_at_contour = min((val,idx) for (idx,val)
                                                  in enumerate(_L_hist_diff))
            self.D['prob_' + str(contour)] = _L_hist[idx_at_contour]
            
            if diff_at_contour > 0.1:
                UserWarning(str(contour * 100.) + '% contour not constrained.')	

    def plot_contour(self):
        X, Y = np.meshgrid(self.slopes, self.slopes)		
        levels = [self.D['prob_' + str(contour)]
                  for contour in self.contour_list]	
        
        qtty = np.reshape(self.L, (self.N_s, self.N_s))
        self.ax.contour(
          X, Y, qtty, levels, colors=['r', 'r'],
          linestyles=['--', '-'], linewidths=(2., 3.), zorder=5)	

    def add_legend(self):
        self.ax.plot(np.nan, np.nan, ls='-', color='r',
                     linewidth=3., label=r'$68\%$')
        self.ax.plot(np.nan, np.nan, ls='--', color='r',
                     linewidth=2., label=r'$95\%$')        

        self.ax.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1)
                      
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        fname = (
          'Fig_likelihood_' + str(self.t_onset.to(u.yr).value / 1.e9) + '_'\
          + str(self.t_break.to(u.yr).value / 1.e9) + '_' + self.sfh_type)

        if self.save_fig:
            plt.savefig(directory + fname + '.' + extension, format=extension,
                        dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def make_plot(self):
        self.set_fig_frame()
        self.get_data()
        self.normalize_likelihood()
        self.plot_data()
        self.find_sigma_fractions()
        self.plot_contour()
        self.add_legend()
        plt.tight_layout()
        self.save_figure()
        self.show_figure()  

if __name__ == '__main__':
    Plot_Likelihood('delayed-exponential', 1.e8 * u.yr, 1.e9 * u.yr,
                    show_fig=True, save_fig=True)
 
