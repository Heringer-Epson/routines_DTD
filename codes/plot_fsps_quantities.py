#!/usr/bin/env python

import os
import warnings
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

class Make_Panels(object):
    
    def __init__(self, _inputs, _s1, _s2):
        """Makes a figure where a set of DTDs is plotted as a function of age.
        """
        self._inputs = _inputs
        self._s1 = _s1
        self._s2 = _s2
       
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
        self._taus = None
        self.outdir = None
        
        self.make_plot()
        
    def make_output_folder(self):
        self.outdir = self._inputs.subdir_fullpath + 'FIGURES/PANELS/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def select_taus_to_plot(self):
        """Select a few of the SFH timescales to be plotted."""        
        
        try:
            self._taus = np.array([self._inputs.tau_list[k].to(u.yr).value for k in
                                  [0,2,5,6,7]]) * u.yr 
        except:
            warning_msg = (
              'The SFH timescales that will be plotted under "plot_several"'\
              + 'have been redefined to [1, 2, 5, 7, 10] Gyr.')
            warnings.warn(warning_msg)
            self._taus = np.array([1., 2., 5., 7., 10.]) * 1.e9 * u.yr

    def set_fig_frame(self):

        if self._inputs.sfh_type == 'exponential':
            sfh_str = 'e^{-t/ \\tau}'
        elif self._inputs.sfh_type == 'delayed-exponential':
            sfh_str = 't\\times e^{-t/ \\tau}'
        
        title = (
          r'$(s_1,s_2,t_{\rm{WD}},t_{\rm{c}},\rm{sfh_{type}})=(' + str(self._s1)\
          + ',' + str(self._s2) + ',' + str(self._inputs.t_onset.to(u.yr).value\
          / 1.e9) + ',' + str(self._inputs.t_cutoff.to(u.yr).value / 1.e9)
          + ',' + sfh_str + ')$')   

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
        for tau in self._taus:

            tau_suffix = str(tau.to(u.yr).value / 1.e9)
            synpop_dir = self._inputs.subdir_fullpath + 'fsps_FILES/'
            synpop_fname = self._inputs.sfh_type + '_tau-' + tau_suffix + '.dat'

            self.M['model' + tau_suffix] = Model_Rates(
              self._s1, self._s2, self._inputs.t_onset,
              self._inputs.t_cutoff, self._inputs.filter_1,
              self._inputs.filter_2, self._inputs.imf_type,
              self._inputs.sfh_type, self._inputs.Z,
              synpop_dir, synpop_fname)
        
        #Collect the boolean array of where the array ages from fsps matches
        #the require ages where markers should be placed. Since all the fsps
        #are consistent for different simulations, any simulation will suffice.
        _age = self.M['model' + tau_suffix].age.to(u.yr).value
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
        
        for i, tau in enumerate(self._taus):
            tau_suffix = str(tau.to(u.yr).value / 1.e9)            
            label = r'$\tau =' + tau_suffix + '\ \mathrm{Gyr}$'
            age = np.log10(self.M['model' + tau_suffix].age.to(u.yr).value)
            Dcolor = self.M['model' + tau_suffix].Dcolor

            self.ax_a.plot(age, Dcolor, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i], label=label)
            self.ax_b.plot(Dcolor, age, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], age[self.marker_cond])):
                self.ax_b.plot(x, y, ls='None', marker=markers[j],
                fillstyle='none', markersize=10., color=colors[i])        
        
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
        
        for i, tau in enumerate(self._taus):
            tau_suffix = str(tau.to(u.yr).value / 1.e9)            
            age = np.log10(self.M['model' + tau_suffix].age.to(u.yr).value)
            Dcolor = self.M['model' + tau_suffix].Dcolor
            mass = self.M['model' + tau_suffix].int_formed_mass
            
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
        
        for i, tau in enumerate(self._taus):
            tau_suffix = str(tau.to(u.yr).value / 1.e9) 
            age = np.log10(self.M['model' + tau_suffix].age.to(u.yr).value)
            Dcolor = self.M['model' + tau_suffix].Dcolor
            lum = self.M['model' + tau_suffix].L
                        
            self.ax_e.plot(age, lum, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])
            self.ax_f.plot(Dcolor, lum, lw=3., marker='None', color=colors[i],
                           dashes=dashes[i])

            for j, (x, y) in enumerate(zip(
              Dcolor[self.marker_cond], lum[self.marker_cond])):
                self.ax_f.plot(x, y, ls='None', marker=markers[j],
                fillstyle='none', markersize=10., color=colors[i]) 

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
        
        for i, tau in enumerate(self._taus):
            tau_suffix = str(tau.to(u.yr).value / 1.e9)           
            age = np.log10(self.M['model' + tau_suffix].age.to(u.yr).value)
            Dcolor = self.M['model' + tau_suffix].Dcolor
            sSNR = self.M['model' + tau_suffix].sSNR
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
        
        for i, tau in enumerate(self._taus):
            tau_suffix = str(tau.to(u.yr).value / 1.e9)      
            age = np.log10(self.M['model' + tau_suffix].age.to(u.yr).value)
            Dcolor = self.M['model' + tau_suffix].Dcolor
            sSNRm = self.M['model' + tau_suffix].sSNRm
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
        
        for i, tau in enumerate(self._taus):
            tau_suffix = str(tau.to(u.yr).value / 1.e9)          
            age = np.log10(self.M['model' + tau_suffix].age.to(u.yr).value)
            Dcolor = self.M['model' + tau_suffix].Dcolor
            sSNRL = self.M['model' + tau_suffix].sSNRL
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
        fname = ('Fig_' + str(format(self._s1, '.1f'))
                 + '_' + str(format(self._s2, '.1f')) + '.')
        if self._inputs.save_fig:
            plt.savefig(self.outdir + fname + extension,
                        format=extension, dpi=dpi)
        
    def show_figure(self):
        if self._inputs.show_fig:
            plt.show()
                
    def make_plot(self):
        self.make_output_folder()
        self.select_taus_to_plot()
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

class Run_Plotter(object):
    
    def __init__(self, _inputs):

        print '\n\n>GENERATING MODEL FIGURES...\n'
        for s2 in _inputs.slopes[::5]:
            s2_str = str(format(s2, '.1f'))
            for s1 in _inputs.slopes[::5]:
                s1_str = str(format(s1, '.1f'))
                print '  *s1/s2=' + s1_str + '/' + s2_str 
                Make_Panels(_inputs, s1, s2)        

