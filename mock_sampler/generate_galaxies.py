#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd
from astropy import units as u

def norm(tau, logage):
    age = 10.**logage / 1.e9
    return -1. / (tau * (np.exp(-age[-1] / tau) - np.exp(-age[0] / tau)))   

def sfh(t, tau):
    return np.exp(-t / tau)
    
def get_binned_mass(T, tau, to, norm):    
    #By construction, T is always older than 0.5 Gyr.
    m1 = norm * tau * (sfh(T - to,tau) - sfh(T - 0.,tau))
    m2 = norm * tau * (sfh(T - 0.42,tau) - sfh(T - to,tau))
    if T < 2.4: #Then t_final == T.
        m3 = norm * tau * (sfh(T - T,tau) - sfh(T - 0.42,tau))
        m4 = 0.
    elif T >= 2.4:
        m3 = norm * tau * (sfh(T - 2.4,tau) - sfh(T - 0.42,tau))
        m4 = norm * tau * (sfh(T - T,tau) - sfh(T - 2.4,tau))
    return (m1,m2,m3,m4)

class Galaxy_Sampler(object):
    """
    Code Description
    ----------    
    Create a sample of galaxies containing colors and magnitudes. The SFH,
    age and mass are randomly sampled, with uninformative priors. 

    Notes:
    ------
    The FSPS parameters are as those used for the 'standard' sample:
    solar metallicity, Kroupa IMF, exponential SFH, etc...

    Parameters:
    -----------
    N : ~int
        Number of galaxies to be create.
        
    Outputs:
    --------
    ./../OUTPUT_FILES/MOCK_SAMPLE/data_mock.csv
    """
    
    def __init__(self, inputs, N):
        self.inputs, self.N = inputs, N
        
        self.D, self.O = {}, {}
        self.rdm_pars = []
        self.logage, self.RS_color = None, None
        
        self.run_sampler()
        
    def read_fsps_files(self):      
        subdir_path = os.path.join(
          os.environ['PATH_ssnarl'], 'fsps_files', self.inputs.fsps_path)

        #Get SSP.
        self.D['SSP'] = pd.read_csv(
              subdir_path + 'SSP.dat', header=0, escapechar='#')
        
        #Get exponential tracks.        
        for tau in self.inputs.tau_list:
            TS = str(tau.to(u.yr).value / 1.e9)
            fname = 'tau-' + TS + '.dat'
            self.D[TS] = pd.read_csv(
              subdir_path + '/' + fname, header=0, escapechar='#')
        #Read array of FSPS ages.
        self.logage = self.D['SSP'][' log_age']

    def get_RS_color(self):
        RS_cond = (self.logage == 10.0)
        self.RS_color = float(
          self.D['SSP']['g'][RS_cond] - self.D['SSP']['r'][RS_cond])
    
    def create_random_inputs(self):

        #Only use galaxy ages >= 0.5 Gyr
        #replace=True means that elements may be repeated.       
        self.O['logmass'] = np.random.uniform(low=8., high=10.5, size=self.N)
        self.O['logage'] = np.random.choice(
          self.logage[10.**self.logage / 1.e9 >= 0.5], replace=True, size=self.N) 
        self.O['tau'] = np.random.choice(
          self.inputs.tau_list.to(u.Gyr).value, replace=True, size=self.N)
        self.rdm_pars = np.array([self.O['tau'], self.O['logage'],
                                  self.O['logmass']]).transpose().astype(float)        
        
    def create_survey(self):
        self.O['petroMag_g'], self.O['petroMag_r'] = [], []
        for pars in self.rdm_pars:
            tau, logage, logmass = pars
            cond = (self.D[str(tau)][' log_age'] == logage)
            self.O['petroMag_g'].append(float(self.D[str(tau)]['g'][cond]))
            self.O['petroMag_r'].append(float(self.D[str(tau)]['r'][cond]))
        self.O['petroMag_g'] = np.array(self.O['petroMag_g'])
        self.O['petroMag_r'] = np.array(self.O['petroMag_r'])
        self.O['Dcolor_gr'] = (
          self.O['petroMag_g'] - self.O['petroMag_r'] - self.RS_color)
        
        #Redshift is zero to avoid K-corrections. One has to be careful with
        #the visibility window, which is not 1 for z=0.
        self.O['z'] = np.zeros(self.N)
        self.O['extinction_g'] = np.zeros(self.N)
        self.O['extinction_r'] = np.zeros(self.N)

    def compute_binned_mass(self):
        #Formed mass (rather than present day mass).
        for tau in self.inputs.tau_list:
            TS = str(tau.to(u.yr).value / 1.e9)
            self.D[TS] = norm(tau.to(u.Gyr).value, self.logage.values) #FSPS logage.
                        
        #logage below refers to the mock galaxy.
        masses = np.array([get_binned_mass(
          10.**logage / 1.e9, tau, self.inputs.t_onset.to(u.Gyr).value,
          self.D[str(tau)]) for (tau,logage) in zip(self.O['tau'],self.O['logage'])])
          
        self.O['vespa1'], self.O['vespa2'], self.O['vespa3'],\
          self.O['vespa4'] = masses.transpose()
        
    def save_output(self):
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/data_mock.csv'
        df = pd.DataFrame(self.O)
        df.to_csv(fpath)
            
    def run_sampler(self):
        self.read_fsps_files()
        self.get_RS_color()
        self.create_random_inputs()
        self.create_survey()
        self.compute_binned_mass()
        self.save_output()
