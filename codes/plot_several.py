#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from SN_rate_gen import Model_Rates

tau_list = [1., 2., 5., 7., 10.]
marker_ages = [1.e8, 1.e9, 1.e10] #in units of yr.
markers = ['^', 'o', 's']

dashes = [(None,None), (4,4), (1,5), (4,2,1,2), (5,2,20,2)]
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

class Make_Several(object):
    
    def __init__(self, s1, s2, t_onset, t_break, sfh_type, show_fig=True,
                 save_fig=False):
        """Makes a figure where a set of DTDs is plotted as a function of age.
        """

        self.s1 = s1
        self.s2 = s2
        self.t_onset = t_onset
        self.t_break = t_break
        self.sfh_type = sfh_type
        
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(14,22))        
        self.ax_a = plt.subplot(621)  
        self.ax_b = plt.subplot(622)  
        self.ax_c = plt.subplot(623, sharex=self.ax_a)  
        self.ax_d = plt.subplot(624, sharex=self.ax_b, sharey=self.ax_c)  
        self.ax_e = plt.subplot(625, sharex=self.ax_a)  
        self.ax_f = plt.subplot(626, sharex=self.ax_b, sharey=self.ax_e)  
        self.ax_g = plt.subplot(627, sharex=self.ax_a)  
        self.ax_h = plt.subplot(628, sharex=self.ax_b, sharey=self.ax_g)  
        self.ax_i = plt.subplot(629, sharex=self.ax_a)  
        self.ax_k = plt.subplot(6,2,10, sharex=self.ax_b, sharey=self.ax_i) 
        self.ax_l = plt.subplot(6,2,11, sharex=self.ax_a)
        self.ax_m = plt.subplot(6,2,12, sharex=self.ax_b, sharey=self.ax_l)
        
        self.M = {}  
        self.fs = 20.
        self.marker_cond = None
        
        self.make_plot()

    def set_fig_frame(self):

        if self.sfh_type == 'exponential':
            sfh_str = 'e^{-t/ \\tau}'
        elif self.sfh_type == 'delayed-exponential':
            sfh_str = 't\\times e^{-t/ \\tau}'
        
        title = (
          r'$(s_1,s_2,t_{\rm{WD}},t_{\rm{c}},\rm{sfh_{type}})=(' + str(self.s1)\
          + ',' + str(self.s2) + ',' + str(self.t_onset.to(u.yr).value / 1.e9)\
          + ',' + str(self.t_break.to(u.yr).value / 1.e9) + ',' + sfh_str + ')$')   

         
        self.FIG.suptitle(title, fontsize=self.fs, fontweight='bold')
        
        age_label = r'$\rm{log}\ t\ \rm{[yr]}$'
        Dcolor_label = r'$\Delta (g-r)$'

        self.ax_l.set_xlabel(age_label, fontsize=self.fs)
        self.ax_l.set_xlim(7., 10.5)      
        self.ax_l.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax_l.minorticks_on()
        self.ax_l.xaxis.set_minor_locator(MultipleLocator(.5))
        self.ax_l.xaxis.set_major_locator(MultipleLocator(1.))

        self.ax_m.set_xlabel(Dcolor_label, fontsize=self.fs)
        self.ax_m.set_xlim(-1.2, 0.2)      
        self.ax_m.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax_m.minorticks_on()
        self.ax_m.xaxis.set_minor_locator(MultipleLocator(.2))
        self.ax_m.xaxis.set_major_locator(MultipleLocator(.4))

        #Make legend for age markers.
        for _t, marker in zip(marker_ages, markers):
            label = r'$t=' + str(_t / 1.e9) + '\ \mathrm{Gyr}$'
            
            self.ax_b.plot(np.nan, np.nan, ls='None', marker=marker,
            fillstyle='none', markersize=10., color='k', label=label)
        
        self.ax_b.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1,
                         loc=4)  

            
    def collect_data(self):
        for tau in tau_list:
            fname = self.sfh_type + '_tau-' + str(tau) + '.dat'
            self.M['model' + str(tau)] = Model_Rates(
              self.s1, self.s2, self.t_onset, self.t_break, self.sfh_type,
              tau * 1.e9 * u.yr)
        
        #Collect the boolean array of where the array ages from fsps matches
        #the require ages where markers should be placed. Since all the fsps
        #are consistent for different simulations, any simulation will suffice.
        _age = self.M['model' + str(tau)].age.to(u.yr).value
        self.marker_cond = np.in1d(_age,np.array(marker_ages))
            
    def plot_age_Dcolor(self):

        self.ax_a.set_ylabel(r'$\Delta (g-r)$', fontsize=self.fs)
        self.ax_a.set_ylim(-1.2, 0.2)
        self.ax_a.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_a.yaxis.set_minor_locator(MultipleLocator(.2))
        self.ax_a.yaxis.set_major_locator(MultipleLocator(.4))  
        self.ax_a.tick_params('both', length=8, width=1., which='major')
        self.ax_a.tick_params('both', length=4, width=1., which='minor')
        self.ax_a.tick_params(labelbottom='off') 

        self.ax_b.set_ylabel(r'$\rm{log}\ t\ \rm{[yr]}$', fontsize=self.fs)
        self.ax_b.yaxis.set_label_position('right')
        self.ax_b.yaxis.tick_right()
        self.ax_b.set_ylim(7., 10.5)
        self.ax_b.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_b.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax_b.yaxis.set_major_locator(MultipleLocator(1.))  
        self.ax_b.tick_params('both', length=8, width=1., which='major')
        self.ax_b.tick_params('both', length=4, width=1., which='minor')
        self.ax_b.tick_params(labelbottom='off') 
        
        for i, tau in enumerate(tau_list):
            label = r'$\tau =' + str(tau) + '\ \mathrm{Gyr}$'
            age = np.log10(self.M['model' + str(tau)].age.to(u.yr).value)

            Dcolor = self.M['model' + str(tau)].Dcolor

            self.ax_a.plot(age, Dcolor, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i], label=label)
            self.ax_b.plot(Dcolor, age, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], age[self.marker_cond])):
                self.ax_b.plot(x, y, ls='None', marker=markers[j], fillstyle='none',
                markersize=10., color=colors[i])        
        
        self.ax_a.legend(frameon=False, fontsize=self.fs, numpoints=1, ncol=1,
                         loc=2)        

    def plot_mass(self):

        mass_label = r'$m\ \rm{[M_\odot]}$'
       
        self.ax_c.set_ylabel(mass_label, fontsize=self.fs)
        self.ax_c.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_c.tick_params('both', length=8, width=1., which='major')
        self.ax_c.tick_params('both', length=4, width=1., which='minor')        
        self.ax_c.tick_params(labelbottom='off') 
      
        self.ax_d.set_ylabel(mass_label, fontsize=self.fs)
        self.ax_d.yaxis.set_label_position('right')
        self.ax_d.yaxis.tick_right()
        self.ax_d.set_ylim(0., 1.)
        self.ax_d.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_d.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax_d.yaxis.set_major_locator(MultipleLocator(.2))  
        self.ax_d.tick_params('both', length=8, width=1., which='major')
        self.ax_d.tick_params('both', length=4, width=1., which='minor')
        self.ax_d.tick_params(labelbottom='off') 
        
        for i, tau in enumerate(tau_list):
            age = np.log10(self.M['model' + str(tau)].age.to(u.yr).value)
            Dcolor = self.M['model' + str(tau)].Dcolor
            mass = self.M['model' + str(tau)].int_formed_mass
            
            self.ax_c.plot(age, mass, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])
            self.ax_d.plot(Dcolor, mass, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], mass[self.marker_cond])):
                self.ax_d.plot(x, y, ls='None', marker=markers[j], fillstyle='none',
                markersize=10., color=colors[i])

    def plot_lum(self):

        lum_label = r'$L_{r}\ \rm{[L_\odot]}$'
        
        self.ax_e.set_ylabel(lum_label, fontsize=self.fs)
        self.ax_e.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_e.tick_params('both', length=8, width=1., which='major')
        self.ax_e.tick_params('both', length=4, width=1., which='minor')        
        self.ax_e.tick_params(labelbottom='off') 
      
        self.ax_f.set_ylabel(lum_label, fontsize=self.fs)
        self.ax_f.yaxis.set_label_position('right')
        self.ax_f.yaxis.tick_right()
        self.ax_f.set_ylim(0., 3.)
        self.ax_f.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_f.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax_f.yaxis.set_major_locator(MultipleLocator(1.))  
        self.ax_f.tick_params('both', length=8, width=1., which='major')
        self.ax_f.tick_params('both', length=4, width=1., which='minor')
        self.ax_f.tick_params(labelbottom='off') 
        
        for i, tau in enumerate(tau_list):
            age = np.log10(self.M['model' + str(tau)].age.to(u.yr).value)
            Dcolor = self.M['model' + str(tau)].Dcolor
            lum = self.M['model' + str(tau)].L
                        
            self.ax_e.plot(age, lum, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])
            self.ax_f.plot(Dcolor, lum, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], lum[self.marker_cond])):
                self.ax_f.plot(x, y, ls='None', marker=markers[j], fillstyle='none',
                markersize=10., color=colors[i]) 

    def plot_sSNR(self):

        sSNR_label = r'$sSNR\ \rm{[yr^{-1}]}$'
        
        self.ax_g.set_ylabel(sSNR_label, fontsize=self.fs)
        self.ax_g.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_g.tick_params('both', length=8, width=1., which='major')
        self.ax_g.tick_params('both', length=4, width=1., which='minor')        
        self.ax_g.tick_params(labelbottom='off') 
      
        self.ax_h.set_ylabel(sSNR_label, fontsize=self.fs)
        self.ax_h.yaxis.set_label_position('right')
        self.ax_h.yaxis.tick_right()
        self.ax_h.set_ylim(-14., -11.5)
        self.ax_h.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_h.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax_h.yaxis.set_major_locator(MultipleLocator(1.))  
        self.ax_h.tick_params('both', length=8, width=1., which='major')
        self.ax_h.tick_params('both', length=4, width=1., which='minor')
        self.ax_h.tick_params(labelbottom='off') 
        
        for i, tau in enumerate(tau_list):
            age = np.log10(self.M['model' + str(tau)].age.to(u.yr).value)
            Dcolor = self.M['model' + str(tau)].Dcolor
            sSNR = self.M['model' + str(tau)].sSNR
            sSNR[sSNR <= 0.] = 1.e-40
            sSNR = np.log10(sSNR)
                                    
            self.ax_g.plot(age, sSNR, lw=3., marker='None',color=colors[i],
                           dashes=dashes[i])
            self.ax_h.plot(Dcolor, sSNR, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], sSNR[self.marker_cond])):
                self.ax_h.plot(x, y, ls='None', marker=markers[j], fillstyle='none',
                markersize=10., color=colors[i]) 
    
    def plot_sSNRm(self):

        sSNRm_label = r'$sSNR_{m}\ \rm{[yr^{-1}\ M_\odot ^{-1}]}$'
        
        self.ax_i.set_ylabel(sSNRm_label, fontsize=self.fs)
        self.ax_i.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_i.tick_params('both', length=8, width=1., which='major')
        self.ax_i.tick_params('both', length=4, width=1., which='minor')        
        self.ax_i.tick_params(labelbottom='off') 
      
        self.ax_k.set_ylabel(sSNRm_label, fontsize=self.fs)
        self.ax_k.yaxis.set_label_position('right')
        self.ax_k.yaxis.tick_right()
        self.ax_k.set_ylim(-14., -10.5)
        self.ax_k.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_k.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax_k.yaxis.set_major_locator(MultipleLocator(1.))  
        self.ax_k.tick_params('both', length=8, width=1., which='major')
        self.ax_k.tick_params('both', length=4, width=1., which='minor')
        self.ax_k.tick_params(labelbottom='off') 
        
        for i, tau in enumerate(tau_list):
            age = np.log10(self.M['model' + str(tau)].age.to(u.yr).value)
            Dcolor = self.M['model' + str(tau)].Dcolor
            sSNRm = self.M['model' + str(tau)].sSNRm
            sSNRm[sSNRm <= 0.] = 1.e-40
            sSNRm = np.log10(sSNRm)
                                    
            self.ax_i.plot(age, sSNRm, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])
            self.ax_k.plot(Dcolor, sSNRm, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], sSNRm[self.marker_cond])):
                self.ax_k.plot(x, y, ls='None', marker=markers[j], fillstyle='none',
                markersize=10., color=colors[i]) 
                
    def plot_sSNRL(self):

        sSNRm_label = r'$sSNR_{L}\ \rm{[yr^{-1}\ L_\odot ^{-1}]}$'
        
        self.ax_l.set_ylabel(sSNRm_label, fontsize=self.fs)
        self.ax_l.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_l.tick_params('both', length=8, width=1., which='major')
        self.ax_l.tick_params('both', length=4, width=1., which='minor')        
      
        self.ax_m.set_ylabel(sSNRm_label, fontsize=self.fs)
        self.ax_m.yaxis.set_label_position('right')
        self.ax_m.yaxis.tick_right()
        self.ax_m.set_ylim(-14., -11.)
        self.ax_m.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax_m.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax_m.yaxis.set_major_locator(MultipleLocator(1.))  
        self.ax_m.tick_params('both', length=8, width=1., which='major')
        self.ax_m.tick_params('both', length=4, width=1., which='minor')
        
        for i, tau in enumerate(tau_list):
            age = np.log10(self.M['model' + str(tau)].age.to(u.yr).value)
            Dcolor = self.M['model' + str(tau)].Dcolor
            sSNRL = self.M['model' + str(tau)].sSNRL
            sSNRL[sSNRL <= 0.] = 1.e-40
            sSNRL = np.log10(sSNRL)
                                    
            self.ax_l.plot(age, sSNRL, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])
            self.ax_m.plot(Dcolor, sSNRL, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], sSNRL[self.marker_cond])):
                self.ax_m.plot(x, y, ls='None', marker=markers[j], fillstyle='none',
                markersize=10., color=colors[i])     
    
    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/' + self.sfh_type + '/'
        fname = (
          'Fig_' + self.sfh_type + '_' + str(self.s1) + '_' + str(self.s2) + '_'\
          + str(self.t_onset.to(u.yr).value / 1.e9) + '_'\
          + str(self.t_break.to(u.yr).value / 1.e9) + '.')
        
        if self.save_fig:
            plt.savefig(directory + fname + extension, format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def make_plot(self):
        self.set_fig_frame()
        self.collect_data()
        self.plot_age_Dcolor()
        self.plot_mass()
        self.plot_lum()
        self.plot_sSNR()
        self.plot_sSNRm()
        self.plot_sSNRL()
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        self.FIG.subplots_adjust(top=0.95)
        self.save_figure()
        self.show_figure()
        plt.close(self.FIG) 

if __name__ == '__main__':

    slope_list = np.arange(-3., 0.01, 0.2)
    for s2 in slope_list:
        for s1 in slope_list:
            s1, s2 = float(format(s1, '.1f')), float(format(s2, '.1f'))
            print str(s1) + '/' + str(s2)
            Make_Several(s1, s2, 1.e8 * u.yr, 1.e9 * u.yr, 'exponential',
                         show_fig=False, save_fig=True)     
            Make_Several(s1, s2, 1.e8 * u.yr, 1.e9 * u.yr, 'delayed-exponential',
                         show_fig=False, save_fig=True) 

    #Make_Several(-1., -1., 1.e8 * u.yr, 1.e9 * u.yr, 'exponential',
    #             show_fig=True, save_fig=False) 
