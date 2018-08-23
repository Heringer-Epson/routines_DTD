#!/usr/bin/env python

import numpy as np
import pandas as pd
from astropy import units as u
from Dcolor2sSNRL_gen import Generate_Curve
from multiprocessing import Pool
from functools import partial
from lib import stats

def calculate_likelihood(_inputs, df, mode, N_obs, norm, _v1, _v2):
    if mode == 's1/s2':
        _A, _s1, _s2 = 1.e-12, _v1, _v2
    elif mode == 's1=s2':
        _A, _s1, _s2 = _v1, _v2, _v2
    generator = Generate_Curve(_inputs, _A, _s1, _s2)
    N_expected, ln_L = stats.compute_L_using_sSNRL(
      generator.Dcolor2sSNRL, df['Dcolor'], df['absmag'], df['z'],
      df['is_host'], N_obs, _inputs.visibility_flag, norm)
    line = '\n' + str(_A) + ',' + str(format(_s1, '.5e')) + ','\
      + str(format(_s2, '.5e')) + ',' + str(N_expected) + ',' + str(ln_L)
    return line 
    
def calculate_likelihood_3D(_inputs, df, N_obs, _A, _s1, _s2):
    generator = Generate_Curve(_inputs, _A, _s1, _s2)
    N_expected, ln_L = stats.compute_L_using_sSNRL(
      generator.Dcolor2sSNRL, df['Dcolor'], df['absmag'], df['z'],
      df['is_host'], N_obs, _inputs.visibility_flag, False)
    line = '\n' + str(_A) + ',' + str(format(_s1, '.5e')) + ','\
      + str(format(_s2, '.5e')) + ',' + str(N_expected) + ',' + str(ln_L)
    return line 

class Get_Likelihood(object):
    """
    Description:
    ------------
    This code will create files containing the a list of likelihood values
    computed for DTDs which assume different parameters. Two main cases are
    currently considered: (i) SN rate = A*t**s, (ii) SN rate /propto t**s1/s2.
    In the latter case, the rates are normalized to match the observed one and
    the shape of the DTD is tested (break or not around 1Gyr). In the former
    case, the normalization is also considered, assuming a continuous slope.
    Ideally, one could perform a complete analysis in a 3D space, including
    A, s1 and s2, but this is not yet implemented.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihoods/sSNRL_s1_s2.csv'
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihoods/sSNRL_A_s.csv'
    """       
    def __init__(self, _inputs):
        self._inputs = _inputs
                
        self.df = None        
        self.sSNRL_trim_df = None        
        self.v_trim_df = None        
        self.v_df = None        
        self.N_obs = None

        if self._inputs.subdir.split('_')[0] == 'M12':
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

                            
    def write_sSNRL_outputs(self):
        """This assumes a continuous DTD, but leaves the constant
        (normalization) as a free parameter. Useful for comparing against
        results derived from the VESPA analyses. Write output within the loop.
        """
        slopes = self._inputs.slopes
        As = self._inputs.A
        t_ons = self._inputs.t_onset.to(u.Gyr).value
        t_cut = self._inputs.t_cutoff.to(u.Gyr).value

        fnames = ['s1_s2', 'A_s']
        modes = ['s1/s2', 's1=s2']
        norm_flags = [True, False]
        pars = [(slopes, slopes), (As, slopes)]

        for fname, mode, par, norm_flag in\
            zip(fnames,modes,pars,norm_flags):
            
            par1, par2 = par[0], par[1]
            
            fpath = (self._inputs.subdir_fullpath + 'likelihoods/sSNRL_'
                     + fname + '.csv')
            out = open(fpath, 'w')
            out.write('A,s1,s2,N_expected,ln_L') 

            output = []
            for i, v1 in enumerate(par1):
                print 'Calculating set ' + str(i + 1) + '/' + str(len(par1))
                L_of_v2 = partial(
                  calculate_likelihood, self._inputs, self.reduced_df, mode,
                  self.N_obs, norm_flag, v1)
                pool = Pool(5)
                output += pool.map(L_of_v2,par2)
                pool.close()
                pool.join()

            for line in output:
                out.write(line) 
            out.close()

    def write_sSNRL_outputs_3D(self):
        """Allows the factor A and the slopes pre and post t_cutoff to vary.
        """
        slopes = self._inputs.slopes[::-1]
        As = self._inputs.A[::-1]
        t_ons = self._inputs.t_onset.to(u.Gyr).value
        t_cut = self._inputs.t_cutoff.to(u.Gyr).value

        fpath = (self._inputs.subdir_fullpath + 'likelihoods/sSNRL_3D.csv')
        out = open(fpath, 'w')
        out.write('A,s1,s2,N_expected,ln_L') 

        output = []
        for i, A in enumerate(As):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(As))
            for j, s1 in enumerate(slopes):
                print '  -Calculating subset ' + str(j + 1) + '/' + str(len(slopes))
                L_of_s2 = partial(
                  calculate_likelihood_3D, self._inputs, self.reduced_df,
                  self.N_obs, A, s1)
                pool = Pool(5)
                output += pool.map(L_of_s2,slopes)
                pool.close()
                pool.join()
            
        for line in output:
            out.write(line) 
       
        out.close()

    def write_vespa_outputs(self):
        print 'Calculating likelihoods using VESPA masses...'
        slopes = self._inputs.slopes
        As = self._inputs.A  
        
        t_ons = self._inputs.t_onset.to(u.Gyr).value
        t_cut = self._inputs.t_cutoff.to(u.Gyr).value
        
        fpath_v = self._inputs.subdir_fullpath + 'likelihoods/vespa_A_s.csv'
        fpath_v_trim = (self._inputs.subdir_fullpath + 'likelihoods/'\
                        + 'vespatrim_A_s.csv')
        
        with open(fpath_v, 'w') as out1, open(fpath_v_trim, 'w') as out2:
            out1.write('A,s1,s2,ln_L') 
            out2.write('A,s1,s2,ln_L') 
            for A in As:
                for s in slopes:
                    ln_L = stats.compute_L_from_DTDs(A, s, t_ons,
                      self.v_df['mass1'], self.v_df['mass2'], self.v_df['mass3'],
                      self.v_df['z'], self.v_df['is_host'], self._inputs.visibility_flag)
                    out1.write('\n' + str(A) + ',' + str(s) + ',' + str(s)
                               + ',' + str(ln_L))
                    
                    ln_L = stats.compute_L_from_DTDs(A, s, t_ons,
                      self.v_trim_df['mass1'], self.v_trim_df['mass2'],
                      self.v_trim_df['mass3'], self.v_trim_df['z'],
                      self.v_trim_df['is_host'], self._inputs.visibility_flag)
                    out2.write('\n' + str(A) + ',' + str(s) + ',' + str(s)
                               + ',' + str(ln_L))
                                    
    def run_analysis(self):
        self.retrieve_data()
        self.subselect_data()
        self.write_sSNRL_outputs()
        if self._inputs.likelihood_3D:
            self.write_sSNRL_outputs_3D()
        if self.add_vespa:
            self.write_vespa_outputs()

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Get_Likelihood(class_input(case='SDSS_gr_Maoz'))

