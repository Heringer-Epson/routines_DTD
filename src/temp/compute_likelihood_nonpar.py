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
        self.redshift = self.df['z']
        self.abs_mag = photo1

    #@profile
    def subselect_data(self):
        
        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
                               
        Dcolor_cond = ((self.Dcolor >= self._inputs.Dcolor_min) &
                       (self.Dcolor <= 2. * RS_std))

        self.hosts = self.hosts[Dcolor_cond].values
        self.abs_mag = self.abs_mag[Dcolor_cond].values
        self.redshift = self.redshift[Dcolor_cond].values
        self.Dcolor = self.Dcolor[Dcolor_cond].values

        host_cond = self.hosts
        ctrl_cond = np.logical_not(self.hosts)

        self.host_abs_mag = self.abs_mag[host_cond]
        self.host_Dcolor = self.Dcolor[host_cond]
        self.host_redshift = self.redshift[host_cond]
        self.ctrl_abs_mag = self.abs_mag[ctrl_cond]
        self.ctrl_redshift = self.redshift[ctrl_cond]
        self.ctrl_Dcolor = self.Dcolor[ctrl_cond]
                
        self.N_obs = len(self.host_Dcolor)

    def visibility_time(self, redshift):

        def detection_eff_func(_z):
            if _z < 0.175:
                detection_eff = 0.72
            else:
                detection_eff = -3.2 * _z + 1.28
            return detection_eff
        
        survey_duration = 269. * u.day
        survey_duration = survey_duration.to(u.year).value
        _time = np.ones(len(redshift)) * survey_duration        
        _time = np.divide(_time,(1. + redshift)) #In the galaxy rest frame.
        
        vec_func = np.vectorize(detection_eff_func)
        eff_correction = vec_func(redshift)
        _time = np.multiply(_time,eff_correction)
        return _time

    #@profile
    def calculate_likelihood(self, _A, _s1, _s2, norm=True):

        generator = Generate_Curve(self._inputs, _A, _s1, _s2)

        #Control sample.
        sSNRL_ctrl = generator.Dcolor2sSNRL(self.ctrl_Dcolor)
        L_ctrl = 10.**(-0.4 * (self.ctrl_abs_mag - 5.))
        SNR_ctrl = np.multiply(sSNRL_ctrl,L_ctrl)
        if self._inputs.visibility_flag:
            correction_factor = self.visibility_time(self.ctrl_redshift)
            SNR_ctrl = np.multiply(SNR_ctrl,correction_factor)
        
        #Hosts.
        sSNRL_host = generator.Dcolor2sSNRL(self.host_Dcolor)
        L_host = 10.**(-0.4 * (self.host_abs_mag - 5.))
        SNR_host = np.multiply(sSNRL_host,L_host)
        if self._inputs.visibility_flag:
            correction_factor = self.visibility_time(self.host_redshift)
            SNR_host = np.multiply(SNR_host,correction_factor)
        
        #ln L.
        if norm:
            _N_expected = np.sum(SNR_ctrl)
            A = self.N_obs / _N_expected
            _lambda = np.log(A * SNR_host)
            _ln_L = - self.N_obs + np.sum(_lambda)
        else:
            _N_expected = np.sum(SNR_ctrl)
            _ln_L = (-np.sum(SNR_ctrl) - np.sum(SNR_host)
                     + np.sum(np.log(SNR_host)))

        return _N_expected, _ln_L       
        
    #@profile
    def write_output_for_grid(self):
        """Copmute SN rates for the control sample (required for normalization)
        and the hosts. Then compute the likelihood for each model. The
        equations and notations follow paper I. Write output within the loop.
        """
        fpath = self._inputs.subdir_fullpath + 'likelihood_grid.csv'

        W = {} #Dictironary for header information.
        W['0'] = '-------- General info --------'
        W['1'] = 'sfh_type: ' + self._inputs.sfh_type
        W['2'] = 't_onset [yr]: ' + str(int(self._inputs.t_onset.to(u.yr).value))
        W['3'] = 't_cutoff [yr]: ' + str(int(self._inputs.t_cutoff.to(u.yr).value))
        W['4'] = 'N_obs: ' + str(self.N_obs)
        W['5'] = '-------- Columns --------'
        W['6'] = 'slope1,slope2,N_expected,ln_L' 
        
        with open(fpath, 'w') as out:           
            for i in range(len(W.keys())): #Write header.
                out.write(W[str(i)] + '\n')

            #Compute rates and write outputs. Note that the order here matters
            #to later read the data and produce the likelihood figure.
            for s2 in self._inputs.slopes[::-1]:
                s2_str = str(format(s2, '.5f'))
                for s1 in self._inputs.slopes[::-1]: 
                    s1_str = str(format(s1, '.5f'))
                    print '  *s1/s2=' + s1_str + '/' + s2_str
                    N_expected, ln_L = self.calculate_likelihood(
                      1.e-3, s1, s2, norm=True)
                    line = (s1_str + ',' + s2_str + ',' + str(N_expected)
                            + ',' + str(ln_L) + '\n')
                    out.write(line)

    def write_output_for_stats_def(self):
        """Copmute SN rates for the control sample (required for normalization)
        and the hosts. Then compute the likelihood for each model. The
        equations and notations follow paper I. Write output within the loop.
        """
        fpath = self._inputs.subdir_fullpath + 'likelihood_def.csv'

        W = {} #Dictironary for header information.
        W['0'] = '-------- General info --------'
        W['1'] = 'sfh_type: ' + self._inputs.sfh_type
        W['2'] = 't_onset [yr]: ' + str(int(self._inputs.t_onset.to(u.yr).value))
        W['3'] = 't_cutoff [yr]: ' + str(int(self._inputs.t_cutoff.to(u.yr).value))
        W['4'] = 'N_obs: ' + str(self.N_obs)
        W['5'] = '-------- Columns --------'
        W['6'] = 'slope1,slope2,N_expected,ln_L' 
        
        with open(fpath, 'w') as out:           
            for i in range(len(W.keys())): #Write header.
                out.write(W[str(i)] + '\n')

            #Compute rates and write outputs. Note that the order here matters
            #to later read the data and produce the likelihood figure.
            for s in self._inputs.slopes_fine[::-1]:
                s_str = str(format(s, '.5f'))
                print '  *s1/s2=-1/' + s_str
                N_expected, ln_L = self.calculate_likelihood(
                  1.e-3, -1., s, norm=True)
                line = ('-1,' + s_str + ',' + str(N_expected)
                        + ',' + str(ln_L) + '\n')
                out.write(line)

    def write_output_for_stats_cont(self):
        """Copmute SN rates for the control sample (required for normalization)
        and the hosts. Then compute the likelihood for each model. The
        equations and notations follow paper I. Write output within the loop.
        """
        fpath = self._inputs.subdir_fullpath + 'likelihood_cont.csv'

        W = {} #Dictironary for header information.
        W['0'] = '-------- General info --------'
        W['1'] = 'sfh_type: ' + self._inputs.sfh_type
        W['2'] = 't_onset [yr]: ' + str(int(self._inputs.t_onset.to(u.yr).value))
        W['3'] = 't_cutoff [yr]: ' + str(int(self._inputs.t_cutoff.to(u.yr).value))
        W['4'] = 'N_obs: ' + str(self.N_obs)
        W['5'] = '-------- Columns --------'
        W['6'] = 'slope1,slope2,N_expected,ln_L' 
        
        with open(fpath, 'w') as out:           
            for i in range(len(W.keys())): #Write header.
                out.write(W[str(i)] + '\n')

            #Compute rates and write outputs. Note that the order here matters
            #to later read the data and produce the likelihood figure.
            for s in self._inputs.slopes_fine[::-1]:
                s_str = str(format(s, '.5f'))
                print '  *s1/s2=' + s_str + '/' + s_str
                N_expected, ln_L = self.calculate_likelihood(
                  1.e-3, s, s, norm=True)
                line = (s_str + ',' + s_str + ',' + str(N_expected)
                        + ',' + str(ln_L) + '\n')
                out.write(line)

    #@profile
    def write_output_for_grid_notnorm(self):
        """This assumes a continuous DTD, but leaves the constant
        (normalization) as a free parameter. Useful for comparing against
        results derived from the VESPA analyses. Write output within the loop.
        """
        fpath = self._inputs.subdir_fullpath + 'likelihood_const-slope_grid.csv'

        W = {} #Dictironary for header information.
        W['0'] = '-------- General info --------'
        W['1'] = 'sfh_type: ' + self._inputs.sfh_type
        W['2'] = 't_onset [yr]: ' + str(int(self._inputs.t_onset.to(u.yr).value))
        W['3'] = 't_cutoff [yr]: None'
        W['4'] = 'N_obs: ' + str(self.N_obs)
        W['5'] = '-------- Columns --------'
        W['6'] = 'slope1,slope2,N_expected,ln_L' 
        
        with open(fpath, 'w') as out:           
            for i in range(len(W.keys())): #Write header.
                out.write(W[str(i)] + '\n')

            #Compute rates and write outputs. Note that the order here matters
            #to later read the data and produce the likelihood figure.
            for A in self._inputs.const_s1[::-1]:
                A_str = str(format(A, '.5f'))
                for s in self._inputs.slopes[::-1]: 
                    s_str = str(format(s, '.5f'))
                    print '  *A/s=' + A_str + '/' + s_str
                    N_expected, ln_L = self.calculate_likelihood(
                      A, s, s, norm=False)
                    line = (A_str + ',' + s_str + ',' + str(N_expected)
                            + ',' + str(ln_L) + '\n')
                    out.write(line)

    #@profile
    def run_analysis(self):
        self.retrieve_data()
        self.subselect_data()
        #self.write_output_for_grid()
        #self.write_output_for_stats_def()
        #self.write_output_for_stats_cont()
        self.write_output_for_grid_notnorm()

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Get_Likelihood(class_input(case='SDSS_gr_Maoz'))
