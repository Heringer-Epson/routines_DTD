#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib import colors

class Plot_Likelihood(object):
    """make an intensity plot of the likelihood of several parametrized DTDs."""
    
    def __init__(self, _inputs, show_fig=True, save_fig=False):

        self._inputs = _inputs
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)
        self.contour_list = [0.95, 0.68]  
        self.fs = 20.
        
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
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')

    def get_data(self):
        
        fpath = self._inputs.subdir_fullpath + 'likelihood.csv'
        slopes_1, slopes_2, self.ln_L = np.loadtxt(
          fpath, delimiter=',', skiprows=9, usecols=(0,1,3), unpack=True)           

        #Get how many slopes there is self.s1 and self.s2 
        self.slopes = np.unique(slopes_1)
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
        ticks_pos = np.arange(s_min, s_max + 1.e-6, s_step) #include s_max
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
        fpath = self._inputs.subdir_fullpath + 'FIGURES/likelihood'
        if self.save_fig:
            plt.savefig(fpath + '.' + extension, format=extension, dpi=dpi)
        
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
