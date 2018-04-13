#!/usr/bin/env python

import numpy as np
from astropy import units as u
from Dcolor2sSNRL_gen import Generate_Curve

class Get_Likelihood(object):
    
    def __init__(self, _inputs):
        """TBW.
        """
        self._inputs = _inputs
                
        self.M = {}        
        self.N_obs = None

        print '\n\n>COMPUTING LIKELIHOOD OF MODELS...\n'
        self.run_analysis()

  
    #@profile
    def get_data(self):
        
        #!~!~!~Here, need to fix which columns are read.
        self.M['M_r_ctrl'], self.M['Dcolor_ctrl'], self.M['Dcolor_err_ctrl'] =\
          np.loadtxt(self._inputs.ctrl_fpath, delimiter=',', skiprows=1,
          usecols=(21,26,27), unpack=True)   
        
        self.M['M_r_host'], self.M['Dcolor_host'], self.M['Dcolor_err_host'] =\
          np.loadtxt(self._inputs.host_fpath, delimiter=',', skiprows=1,
          usecols=(21,26,27), unpack=True)   
        self.M['stretch_host'] = np.genfromtxt(np.loadtxt(
          self._inputs.host_fpath, delimiter=',', skiprows=1, usecols=(31),
          unpack=True, dtype=str)) 

    #@profile
    def trim_samples_by_Dcolor(self):
        
        #Control sample.
        Dcolor_cond = ((self.M['Dcolor_ctrl'] >= self._inputs.Dcolor_min) &
                       (self.M['Dcolor_ctrl'] <= self._inputs.Dcolor_max))
        self.M['M_r_ctrl'] = self.M['M_r_ctrl'][Dcolor_cond]
        self.M['Dcolor_err_ctrl'] = self.M['Dcolor_err_ctrl'][Dcolor_cond]
        self.M['Dcolor_ctrl'] = self.M['Dcolor_ctrl'][Dcolor_cond]
        
        #Hosts.
        Dcolor_cond = ((self.M['Dcolor_host'] >= self._inputs.Dcolor_min) &
                       (self.M['Dcolor_host'] <= self._inputs.Dcolor_max))     
        self.M['M_r_host'] = self.M['M_r_host'][Dcolor_cond]
        self.M['Dcolor_err_host'] = self.M['Dcolor_err_host'][Dcolor_cond]
        self.M['stretch_host'] = self.M['stretch_host'][Dcolor_cond]
        self.M['Dcolor_host'] = self.M['Dcolor_host'][Dcolor_cond]        

        self.N_obs = len(self.M['Dcolor_host'])

    #@profile
    def calculate_likelihood(self, _s1, _s2):

        generator = Generate_Curve(
          _s1, _s2, self._inputs.t_onset, self._inputs.t_cutoff,
          self._inputs.filter_1, self._inputs.filter_2,
          self._inputs.imf_type, self._inputs.sfh_type, self._inputs.Z,
          self._inputs.tau_list)
          
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
        
        fpath = self._inputs.subdir_fullpath + 'likelihood.csv'

        #Dictironary for header information.
        W = {}
    
        #Set header information.
        W['0'] = '-------- General info --------'
        W['1'] = 'sfh_type: ' + self._inputs.sfh_type
        W['2'] = 't_onset [yr]: ' + str(int(self._inputs.t_onset.to(u.yr).value))
        W['3'] = 't_cutoff [yr]: ' + str(int(self._inputs.t_cutoff.to(u.yr).value))
        W['4'] = 'N_obs: ' + str(self.N_obs)
        W['5'] = 'min(D(g-r)): ' + str(self._inputs.Dcolor_min)
        W['6'] = 'max(D(g-r)): ' + str(self._inputs.Dcolor_max)
        W['7'] = '-------- Columns --------'
        W['8'] = 'slope1,slope2,N_expected,ln_L' 
        
        with open(fpath, 'w') as out:
            
            #Write header.
            for i in range(len(W.keys())):
                out.write(W[str(i)] + '\n')

            #Compute rates and write outputs.
            for s2 in np.arange(-3., 0.01, 0.1)[::-1]:
                s2_str = str(format(s2, '.1f'))
                for s1 in np.arange(-3., 0.01, 0.1)[::-1]: 
                    s1_str = str(format(s1, '.1f'))
                    print '  *s1/s2=' + s1_str + '/' + s2_str
                    N_expected, ln_L = self.calculate_likelihood(s1, s2)
                    line = (s1_str + ',' + s2_str + ',' + str(N_expected)
                            + ',' + str(ln_L) + '\n')
                    out.write(line)

    def run_analysis(self):
        self.get_data()
        self.trim_samples_by_Dcolor()
        self.write_output()

