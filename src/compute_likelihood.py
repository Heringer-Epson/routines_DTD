#!/usr/bin/env python

import numpy as np
import pandas as pd
from astropy import units as u
from Dcolor2sSNRL_gen import Generate_Curve
from multiprocessing import Pool
from functools import partial
from lib import stats

def calculate_likelihood(mode, _inputs, _df, _N_obs, _s1, _s2):
    if mode == 'sSNRL':
        generator = Generate_Curve(_inputs, _s1, _s2)
        A, ln_L = stats.compute_L_using_sSNRL(
          generator.Dcolor2sSNRL, _df['Dcolor'], _df['absmag'], _df['z'],
          _df['is_host'], _N_obs, _inputs.visibility_flag)
    elif mode == 'vespa':
        _t_ons = _inputs.t_onset.to(u.Gyr).value
        _t_cut = _inputs.t_cutoff.to(u.Gyr).value   
        A, ln_L = stats.compute_L_from_DTDs(
          _s1, _s2, _t_ons, _t_cut, _df['mass1'], _df['mass2'],  _df['mass3'],
          _df['z'], _df['is_host'], _N_obs, _inputs.visibility_flag)
    line = '\n' + str(format(_s1, '.5e')) + ',' + str(format(_s2, '.5e'))\
      + ',' + str(A) + ','  + str(ln_L)
    return line 

class Get_Likelihood(object):
    """
    Description:
    ------------
    This code will create files containing the a list of likelihood values
    computed for DTDs is /propto t**s1/s2. The rates are normalized to match
    the observed one and the shape of the DTD is tested (break or not
    around 1Gyr). The likelihood for other DTD constants can be obtained
    analytically.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihoods/sSNRL_s1_s2.csv'
    """       
    def __init__(self, _inputs):
        self._inputs = _inputs
                
        self.df = None        
        self.sSNRL_trim_df = None        
        self.v_trim_df = None        
        self.v_df = None        
        self.N_obs = None

        if self._inputs.subdir[:-1].split('_')[0] == 'M12':
            self.add_vespa = True
        else:
            self.add_vespa = False            

        print '\n\n>COMPUTING LIKELIHOOD OF MODELS...\n'
        self.run_analysis()

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)

    def subselect_data(self):

        f1, f2 = self._inputs.filter_1, self._inputs.filter_2
        Dcolor = self.df['Dcolor_' + f2 + f1]

        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
        
        Dcolor_cond = np.array(((Dcolor >= self._inputs.Dcolor_min) &
                               (Dcolor <= 2. * RS_std)), dtype=bool)
        
        hosts = self.df['is_host'][Dcolor_cond].values
        abs_mag = self.df['abs_' + f1][Dcolor_cond].values
        redshift = self.df['z'][Dcolor_cond].values
        Dcolor = Dcolor[Dcolor_cond].values

        self.N_obs = np.sum(hosts)
        self.reduced_df = {
          'absmag': abs_mag, 'Dcolor': Dcolor, 'z': redshift, 'is_host': hosts}

        if self.add_vespa:
            mass1 = (self.df['vespa1'].values + self.df['vespa2'].values) * .55
            mass2 = self.df['vespa3'].values * .55
            mass3 = self.df['vespa4'].values * .55
            redshift = self.df['z'].values
            hosts = self.df['is_host'].values
            
            self.v_df = {
              'mass1': mass1, 'mass2': mass2, 'mass3': mass3,
              'z': redshift, 'is_host': hosts}            
            self.v_trim_df = {
              'mass1': mass1[Dcolor_cond], 'mass2': mass2[Dcolor_cond],
              'mass3': mass3[Dcolor_cond], 'z': redshift[Dcolor_cond],
              'is_host': hosts[Dcolor_cond]} 

                            
    def write_sSNRL_output(self):
        """This assumes a continuous DTD, but leaves the constant
        (normalization) as a free parameter. Useful for comparing against
        results derived from the VESPA analyses. Write output within the loop.
        """
        slopes = self._inputs.slopes
        #t_ons = self._inputs.t_onset.to(u.Gyr).value
        #t_cut = self._inputs.t_cutoff.to(u.Gyr).value       
                
        fpath = self._inputs.subdir_fullpath + 'likelihoods/sSNRL_s1_s2.csv'
        out = open(fpath, 'w')
        out.write('N_obs=' + str(self.N_obs) + '\n')
        out.write('s1,s2,norm_A,ln_L') 

        output = []
        for i, v1 in enumerate(slopes):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            L_of_v2 = partial(calculate_likelihood, 'sSNRL', self._inputs,
                              self.reduced_df, self.N_obs, v1)
            pool = Pool(5)
            output += pool.map(L_of_v2,slopes)
            pool.close()
            pool.join()

        for line in output:
            out.write(line) 
        out.close()

    def write_vespa_nottrim_outputs(self):
        print 'Calculating likelihoods using VESPA masses...'
        slopes = self._inputs.slopes 
                
        fpath = self._inputs.subdir_fullpath + 'likelihoods/vespa_s1_s2.csv'
        out = open(fpath, 'w')
        out.write('N_obs=' + str(self.N_obs) + '\n')
        out.write('s1,s2,norm_A,ln_L') 

        output = []
        for i, v1 in enumerate(slopes):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            L_of_v2 = partial(calculate_likelihood, 'vespa', self._inputs,
                              self.v_df, self.N_obs, v1)
            pool = Pool(5)
            output += pool.map(L_of_v2,slopes)
            pool.close()
            pool.join()

        for line in output:
            out.write(line) 
        out.close()

    def write_vespa_trimmed_outputs(self):
        print 'Calculating likelihoods using VESPA masses...'
        slopes = self._inputs.slopes 
                
        fpath = self._inputs.subdir_fullpath + 'likelihoods/vespatrim_s1_s2.csv'
        out = open(fpath, 'w')
        out.write('N_obs=' + str(self.N_obs) + '\n')
        out.write('s1,s2,norm_A,ln_L') 

        output = []
        for i, v1 in enumerate(slopes):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            L_of_v2 = partial(calculate_likelihood, 'vespa', self._inputs,
                              self.v_trim_df, self.N_obs, v1)
            pool = Pool(5)
            output += pool.map(L_of_v2,slopes)
            pool.close()
            pool.join()

        for line in output:
            out.write(line) 
        out.close()

    def run_analysis(self):
        self.retrieve_data()
        self.subselect_data()
        self.write_sSNRL_output()
        if self.add_vespa:
            self.write_vespa_nottrim_outputs()
            self.write_vespa_trimmed_outputs()

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Get_Likelihood(class_input(case='test-case'))

