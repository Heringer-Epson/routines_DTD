#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm

class Fit_RS(object):
    
    def __init__(self, _inputs):
        self._inputs = _inputs
                
        self.x_data, self.y_data, self.hosts = None, None, None
        self.x_rej, self.y_rej, self.hosts_rej = [], [], []
        self.fit_func = None
        
        self.run_fit()
        
    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_processed.csv'
        self.df = pd.read_csv(fpath, header=0)
        self.f1, self.f2 = self._inputs.filter_1, self._inputs.filter_2
        photo1, photo2 = self.df['abs_' + self.f1], self.df['abs_' + self.f2]
        self.x_data, self.y_data = photo1, photo2 - photo1  
        self.hosts = self.df['n_SN']       

    def remove_nonRS(self):
        """Rough color cut to remove objects that clearly do not belong to
        the RS. This prevents the fit being skewed towards these galaxies."""
        trim_cond = ((self.y_data > 0.6) & (self.y_data < 1.1))
        rej_cond = np.logical_not(trim_cond)
        
        #For plotting purposes only.
        self.x_rej += list(self.x_data[rej_cond])
        self.y_rej += list(self.y_data[rej_cond])
        self.hosts_rej += list(self.hosts[rej_cond]) 
        
        #Used for fitting the RS.
        self.x_data = self.x_data[trim_cond]
        self.y_data = self.y_data[trim_cond]
        self.hosts = self.hosts[trim_cond]

    def make_fit(self, _x, _y):
        fit_coeffs, cov_matrix = np.polyfit(_x, _y, 1, cov=True)
        slope_unc = np.sqrt(cov_matrix[0][0])
        intercept_unc = np.sqrt(cov_matrix[1][1])
        _fit_func = np.poly1d(fit_coeffs)
        return _fit_func

    def compute_std(self, _x, _y, _fit_func):
        dist = _y - _fit_func(_x)
        _std = (np.sum(np.power(dist, 2)) / (len(_y) - 1.))**0.5
        _acc_cond = (np.absolute(dist) <= _std * self._inputs.tol)
        return _acc_cond.values
 
    def iterate_fit(self):
        
        #Use guess--if the first fit were poor, RS galaxies could be rejected.
        counter = 1
        guess_func = np.poly1d([self._inputs.slope_guess,
                                self._inputs.intercept_guess])
        
        acc_cond = self.compute_std(self.x_data, self.y_data, guess_func)
        new_x, new_y = self.x_data[acc_cond], self.y_data[acc_cond]
        new_hosts = self.hosts[acc_cond]
        rel_rej = (abs(float(len(new_x)) - float(len(self.x_data)))
                   / float(len(self.x_data)))
        inp_x, inp_y = new_x, new_y
        inp_hosts = new_hosts
        
        #Plot first fit (guessed).
        rej_cond = np.logical_not(acc_cond)
        self.x_rej += list(self.x_data[rej_cond])
        self.y_rej += list(self.y_data[rej_cond])        
        self.hosts_rej += list(self.hosts[rej_cond])
        Plot_Fit(self._inputs, new_x, new_y,  new_hosts, self.x_rej, self.y_rej,
                 self.hosts_rej, guess_func, guess_func, counter)

        #Iterate until only a small fraction is rejected or til the iter limit.
        while ((rel_rej > 1.e-3) & (counter < 100)):
            counter += 1
            self.fit_func = self.make_fit(inp_x, inp_y)
            acc_cond = self.compute_std(inp_x, inp_y, self.fit_func)
            new_x, new_y = inp_x[acc_cond], inp_y[acc_cond]
            new_hosts = inp_hosts[acc_cond]
            rel_rej = abs(float(len(new_x)) - float(len(inp_x))) / float(len(inp_x))
   
            #Make plots.
            rej_cond = np.logical_not(acc_cond)
            self.x_rej += list(inp_x[rej_cond])
            self.y_rej += list(inp_y[rej_cond])
            self.hosts_rej += list(inp_hosts[rej_cond])   
            if counter % 2 == 0:
                Plot_Fit(self._inputs, new_x, new_y, new_hosts, self.x_rej,
                         self.y_rej, self.hosts_rej, self.fit_func, guess_func,
                         counter)

            inp_x, inp_y, inp_hosts = new_x, new_y, new_hosts

        if counter == 100:
            sys.exit('No convergence reached trying to fit the red sequence.')
    
        self.x_out, self.y_out = new_x, new_y

    def compute_dcolor(self):
        mag = self.df['abs_' + self.f1]
        color = self.df['abs_' + self.f2] - self.df['abs_' + self.f1]
        #self.fit_func(mag) = np.poly1d([-0.0188, 0.346]) #test only.
        self.df['Dcolor_' + self.f2 + self.f1]  = color - self.fit_func(mag)

    def fit_RS_hist(self):

        Dcolor = self.df['Dcolor_' + self.f2 + self.f1]
        acc_cond = ((Dcolor >= self._inputs.Dcolor_range[0]) &
                    (Dcolor <= self._inputs.Dcolor_range[1]))
        Dcolor_acc = Dcolor[acc_cond]

        bins = np.arange(self._inputs.bin_range[0], self._inputs.bin_range[1]
                         + 1.e-5, self._inputs.bin_size)
        
        mu, std = norm.fit(Dcolor_acc)
        
        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        with open(fpath, 'w') as out:
            out.write('RS_mu,RS_std\n')
            out.write(str(format(mu, '.3f')) + ',' +  str(format(std, '.2f')))
        
    def save_output(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df.to_csv(fpath)

    def run_fit(self):
        self.retrieve_data()
        self.remove_nonRS()
        self.iterate_fit()
        self.compute_dcolor()
        self.fit_RS_hist()
        self.save_output()

class Plot_Fit(object):
    
    def __init__(self, _inputs, x_acc, y_acc, hosts_acc, x_rej, y_rej,
                 hosts_rej, fit_func, fit_orig, count):
        
        self._inputs = _inputs
        
        self.x_acc = x_acc
        self.y_acc = y_acc
        self.hosts_acc = hosts_acc
        self.x_rej = np.array(x_rej)
        self.y_rej = np.array(y_rej)
        self.hosts_rej = np.array(hosts_rej)
        self.fit_func = fit_func
        self.fit_orig = fit_orig
        self.count = count

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fs = self._inputs.fs
        
        if ((self._inputs.show_fig) or (self._inputs.save_fig)):
            self.make_plot()
        
    def set_fig_frame(self):
        
        f1 = self._inputs.filter_1
        f2 = self._inputs.filter_2

        x_label = r'$M_{' + f1 + '}$'
        y_label = r'$M_{' + f2 + '} - M_{' + f1 + '}$'
        self.ax.set_xlabel(x_label, fontsize=self.fs)
        self.ax.set_ylabel(y_label, fontsize=self.fs)
        self.ax.set_xlim(-24., -16.)
        self.ax.set_ylim(-.2, 1.2)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        self.ax.xaxis.set_minor_locator(MultipleLocator(1.))
        self.ax.xaxis.set_major_locator(MultipleLocator(2.))  
        self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax.yaxis.set_major_locator(MultipleLocator(.2))  
    
    def plot_state(self):

        self.ax.plot(self.x_acc, self.y_acc, ls='None', marker='.', color='k',
                     label='Accepted')
        self.ax.plot(self.x_acc[self.hosts_acc], self.y_acc[self.hosts_acc],
                     markersize=7., ls='None', marker='^', color='b')        
        
        self.ax.plot(self.x_rej, self.y_rej, ls='None', marker='.', 
                     color='gray', alpha=0.5, label='Rejected')
        self.ax.plot(self.x_rej[self.hosts_rej], self.y_rej[self.hosts_rej],
                     markersize=7., ls='None', marker='^', color='skyblue',
                     alpha=0.5)
        
        x_range = [-24., -16.]
        self.ax.plot(x_range, self.fit_func(x_range), color='#e41a1c', ls='-',
                     lw=2., label='Best fit')
        self.ax.plot(x_range, self.fit_orig(x_range), color='#4daf4a', ls='--',
                     lw=2., label='Guess fit')
        self.ax.plot(x_range, [0.6, 0.6], color='#377eb8', ls='--', lw=4.,
                     label='Pre cut')                

        self.ax.plot([np.nan], [np.nan], markersize=7., ls='None', marker='^',
                  color='b', label='Hosts')
        plt.title('Iteration = ' + str(self.count), fontsize=self.fs - 4.)
        self.ax.legend(frameon=True, fontsize=self.fs - 4., numpoints=1, loc=3)

    def manage_output(self):
        
        if self._inputs.save_fig:
            fig_dir = self._inputs.subdir_fullpath + 'FIGURES/RS_fit/'
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            plt.savefig(fig_dir + 'iter_' + str(self.count) + '.pdf',
                        format='pdf')
        
        if self._inputs.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.plot_state()
        self.manage_output()             
        plt.close(self.fig)    


