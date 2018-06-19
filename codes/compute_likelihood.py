#!/usr/bin/env python

import numpy as np
import pandas as pd
from astropy import units as u
from Dcolor2sSNRL_gen import Generate_Curve

class Get_Likelihood(object):
    
    def __init__(self, _inputs):
        """TBW.
        """
        self._inputs = _inputs
                
        self.df = None        
        self.N_obs = None

        print '\n\n>COMPUTING LIKELIHOOD OF MODELS...\n'
        self.run_analysis()

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0)
        f1, f2 = self._inputs.filter_1, self._inputs.filter_2
        photo1 = self.df['abs_' + f1]
        photo2 = self.df['abs_' + f2]
        self.Dcolor = self.df['Dcolor_' + f2 + f1]
        self.hosts = self.df['n_SN']
        self.abs_mag = photo1

    #@profile
    def subselect_data(self):
        
        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
                               
        Dcolor_cond = ((self.Dcolor >= -10. * RS_std) &
                       (self.Dcolor <= 2. * RS_std))

        self.hosts = self.hosts[Dcolor_cond].values
        self.abs_mag = self.abs_mag[Dcolor_cond].values
        self.Dcolor = self.Dcolor[Dcolor_cond].values

        host_cond = self.hosts
        ctrl_cond = np.logical_not(self.hosts)

        self.host_abs_mag = self.abs_mag[host_cond]
        self.host_Dcolor = self.Dcolor[host_cond]
        self.ctrl_abs_mag = self.abs_mag[ctrl_cond]
        self.ctrl_Dcolor = self.Dcolor[ctrl_cond]
        
        self.N_obs = len(self.host_Dcolor)
        
        #np.logical_not(_acc_cond)

        #print len(self.Dcolor[self.hosts].values)
        #print len(self.Dcolor.values)

    #@profile
    def calculate_likelihood(self, _s1, _s2):

        generator = Generate_Curve(self._inputs, _s1, _s2)

        #Control sample.
        sSNRL_ctrl = generator.Dcolor2sSNRL(self.ctrl_Dcolor)
        L_ctrl = 10.**(-0.4 * (self.ctrl_abs_mag - 5.))
        SNR_ctrl = np.multiply(sSNRL_ctrl,L_ctrl)
        
        #Hosts.
        sSNRL_host = generator.Dcolor2sSNRL(self.host_Dcolor)
        L_host = 10.**(-0.4 * (self.host_abs_mag - 5.))
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

        W = {} #Dictironary for header information.
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
            for i in range(len(W.keys())): #Write header.
                out.write(W[str(i)] + '\n')

            #Compute rates and write outputs. Note that the order here matters
            #to later read the data and produce the likelihood figure.
            for s2 in self._inputs.slopes[::-1]:
                s2_str = str(format(s2, '.1f'))
                for s1 in self._inputs.slopes[::-1]: 
                    s1_str = str(format(s1, '.1f'))
                    print '  *s1/s2=' + s1_str + '/' + s2_str
                    N_expected, ln_L = self.calculate_likelihood(s1, s2)
                    line = (s1_str + ',' + s2_str + ',' + str(N_expected)
                            + ',' + str(ln_L) + '\n')
                    out.write(line)

    def run_analysis(self):
        self.retrieve_data()
        self.subselect_data()
        self.write_output()
