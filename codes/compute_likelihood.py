#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from Dcolor2sSNRL_gen import Generate_Curve

t_onset = 1.e8 * u.yr
t_break = 1.e9 * u.yr
tau_list = [1., 1.5, 2., 3., 4., 5., 7., 10.]
tau_list = [tau * 1.e9 * u.yr for tau in tau_list]
Dcolor_min = -0.4
Dcolor_max = 0.08

class Compute_Rates(object):
    
    def __init__(self, sfh_type='exponential'):
        """TBW.
        """
        self.sfh_type = sfh_type
        self.M = {}        
        self.N_obs = None
        self.run_analysis()
  
    #@profile
    def get_data(self):
        
        #Control sample.
        directory = './../INPUT_FILES/sample_paper-I/'
        fpath = directory + 'spec_sample_data.csv'
        
        self.M['M_r_ctrl'], self.M['Dcolor_ctrl'], self.M['Dcolor_err_ctrl'] =\
          np.loadtxt(fpath, delimiter=',', skiprows=1, usecols=(21,26,27),
          unpack=True)   

        #Hosts
        directory = './../INPUT_FILES/sample_paper-I/'
        fpath = directory + 'hosts_SDSS_spec.csv'
        
        self.M['M_r_host'], self.M['Dcolor_host'], self.M['Dcolor_err_host'] =\
          np.loadtxt(fpath, delimiter=',', skiprows=1, usecols=(21,26,27),
          unpack=True)   
        self.M['stretch_host'] = np.genfromtxt(np.loadtxt(
          fpath, delimiter=',', skiprows=1, usecols=(31), unpack=True, dtype=str)) 

    #@profile
    def trim_samples_by_Dcolor(self):
        
        #Control sample.
        Dcolor_cond = ((self.M['Dcolor_ctrl'] >= Dcolor_min) &
                       (self.M['Dcolor_ctrl'] <= Dcolor_max))
        self.M['M_r_ctrl'] = self.M['M_r_ctrl'][Dcolor_cond]
        self.M['Dcolor_err_ctrl'] = self.M['Dcolor_err_ctrl'][Dcolor_cond]
        self.M['Dcolor_ctrl'] = self.M['Dcolor_ctrl'][Dcolor_cond]
        
        #Hosts.
        Dcolor_cond = ((self.M['Dcolor_host'] >= Dcolor_min) &
                       (self.M['Dcolor_host'] <= Dcolor_max))     
        self.M['M_r_host'] = self.M['M_r_host'][Dcolor_cond]
        self.M['Dcolor_err_host'] = self.M['Dcolor_err_host'][Dcolor_cond]
        self.M['stretch_host'] = self.M['stretch_host'][Dcolor_cond]
        self.M['Dcolor_host'] = self.M['Dcolor_host'][Dcolor_cond]        

        self.N_obs = len(self.M['Dcolor_host'])

    #@profile
    def calculate_likelihood(self, _s1, _s2):

        generator = Generate_Curve(
          _s1, _s2, t_onset, t_break, self.sfh_type, tau_list)

        #Control sample.
        sSNRL_ctrl = generator.Dcolor2sSNRL(self.M['Dcolor_ctrl'])
        L_ctrl = 10.**(-0.4 * (self.M['M_r_ctrl'] - 5.))
        SNR_ctrl = np.multiply(sSNRL_ctrl,L_ctrl)
        
        #Hosts.
        sSNRL_host = generator.Dcolor2sSNRL(self.M['Dcolor_host'])
        L_host = 10.**(-0.4 * (self.M['M_r_host'] - 5.))
        SNR_host = np.multiply(sSNRL_host,L_host)
        
        #ln L.
        _N_expected = np.sum(SNR_ctrl)
        A = self.N_obs / _N_expected
        _lambda = np.log(A * SNR_host)
        _ln_L = - self.N_obs + np.sum(_lambda)
        
        return _N_expected, _ln_L       
        
    #@profile
    def write_output(self):
        """Copmute SN rates for the control sample (required for normalization)
        and the hosts. Then compute the likelihood for each model. The
        equations and notations follow paper I. Write output within the loop.
        """
        
        #Dictironary for header information.
        W = {}
        
        directory = './../OUTPUT_FILES/FILES/'
        fname = ('likelihood_' + str(t_onset.to(u.yr).value / 1.e9) + '_'\
                 + str(t_break.to(u.yr).value / 1.e9) + '_' + self.sfh_type + '.csv')
        
        #Set header information.
        W['0'] = '-------- General info --------'
        W['1'] = 'sfh_type: ' + self.sfh_type
        W['2'] = 't_onset [yr]: ' + str(int(t_onset.to(u.yr).value))
        W['3'] = 't_break [yr]: ' + str(int(t_break.to(u.yr).value))
        W['4'] = 'N_obs: ' + str(self.N_obs)
        W['5'] = 'min(D(g-r)): ' + str(Dcolor_min)
        W['6'] = 'max(D(g-r)): ' + str(Dcolor_max)
        W['7'] = '-------- Columns --------'
        W['8'] = 'slope1,slope2,N_expected,ln_L' 
        
        with open(directory + fname, 'w') as out:
            
            #Write header.
            for i in range(len(W.keys())):
                out.write(W[str(i)] + '\n')

            #Compute rates and write outputs.
            for s2 in np.arange(-3., 0.01, 0.1)[::-1]:
                for s1 in np.arange(-3., 0.01, 0.1)[::-1]: 
                    print str(format(s1, '.1f')), str(format(s2, '.1f'))
                    N_expected, ln_L = self.calculate_likelihood(s1, s2)
                    line = (str(format(s1, '.1f')) + ',' + str(format(s2, '.1f'))\
                            + ',' + str(N_expected) + ',' + str(ln_L) + '\n')
                    out.write(line)

    def run_analysis(self):
        self.get_data()
        self.trim_samples_by_Dcolor()
        self.write_output()

if __name__ == '__main__':
    Compute_Rates(sfh_type='delayed-exponential')
 
