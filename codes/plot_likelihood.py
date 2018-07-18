#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib import colors

class Plot_Likelihood(object):
    """make an intensity plot of the likelihood of several parametrized DTDs."""
    
    def __init__(self, _inputs):

        self._inputs = _inputs
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)
        self.contour_list = [0.95, 0.68]  
        
        self.ln_L = None   
        self.L = None
        self.slopes = None
        self.slopes_1 = None
        self.slopes_2 = None
        self.N_s = None
        self.D = {} #Dictionary for contour variables.   
        
        self.make_plot()
        
    def set_fig_frame(self):
        x_label = r'$s_1$'
        y_label = r'$s_2$'
        self.ax.set_xlabel(x_label, fontsize=24.)
        self.ax.set_ylabel(y_label, fontsize=24.)
        self.ax.tick_params(axis='y', which='major', labelsize=24., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=24., pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor') 

    def get_data(self):
        
        fpath = self._inputs.subdir_fullpath + 'likelihood_grid.csv'
        self.slopes_1, self.slopes_2, self.ln_L = np.loadtxt(
          fpath, delimiter=',', skiprows=7, usecols=(0,1,3), unpack=True)           

        #max_L_cond = (self.ln_L == max(self.ln_L))
        #print slopes_1[max_L_cond], slopes_2[max_L_cond], self.ln_L[max_L_cond]

        #Get how many slopes there is self.s1 and self.s2 
        self.slopes = np.unique(self.slopes_1)
        self.N_s = len(self.slopes)
        
    def normalize_likelihood(self):
        #multiplicative factor to make exponentials small. Otherwise difficult
        #to handle e^-1000.
        _L = self.ln_L - min(self.ln_L) 
        #Normalize in linear scale.
        _L = np.exp(_L)
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
        ticks_pos_minor = np.arange(s_min, s_max + 1.e-6, 2.5 * s_step)
        ticks_pos_major = np.arange(s_min, s_max + 1.e-6, 5. * s_step) 
        ticks_label = []
        for i in xrange(0, self.N_s, 5):
            ticks_label.append(self.slopes[i])

        #Seems that labels do not follow the imshow plot, so has to be inverted.
        ticks_label = ticks_label[::-1] 

        self.ax.set_xticks(ticks_pos_major, minor=False)
        self.ax.set_xticklabels(ticks_label)        
        self.ax.set_xticks(ticks_pos_minor, minor=True)
        
        self.ax.set_yticks(ticks_pos_major)
        self.ax.set_yticklabels(ticks_label)
        self.ax.set_yticks(ticks_pos_minor, minor=True)
  
    def find_sigma_fractions(self):
        
        #Note that the cumulative histogram is normalized since self.L is norm.
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
        self.ax.legend(frameon=False, fontsize=24., numpoints=1, ncol=1)
                      
    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/likelihood.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
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
        self.manage_output() 
