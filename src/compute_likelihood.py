#!/usr/bin/env python
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import numpy as np
import pandas as pd
from astropy import units as u
from multiprocessing import Pool
from functools import partial
import stats
import core_funcs

sys.path.append(os.path.join(os.environ['PATH_ssnarl'], 'src'))
from Dcolor2sSNRL_gen import Generate_Curve
from build_fsps_model import Build_Fsps

filter2sunmag = {'u': 6.39, 'g': 5.12, 'r': 4.65, 'i': 4.53, 'z': 4.51}

#@profile
def calculate_likelihood(mode, _inputs, _df, _N_obs, _D, _s1, _s2):
    if mode == 'sSNRL':
        sunmag = filter2sunmag[_inputs.f0]
        Sgen = Generate_Curve(_inputs, _D, _s1, _s2)
        if _inputs.model_Drange == 'reduced':
            x, y = Sgen.Dcolor_at10Gyr[::-1], Sgen.sSNRL_at10Gyr[::-1]
        elif _inputs.model_Drange == 'extended':
            x, y = Sgen.Dcd_fine, Sgen.sSNRL_fine
        sSNRL = np.asarray(core_funcs.interp_nobound(x, y, _df['Dcolor']))
        A, ln_L = stats.compute_L_using_sSNRL(
          sSNRL, _df['Dcolor'], _df['absmag'], _df['z'],
          _df['is_host'], _N_obs, _inputs.visibility_flag, sunmag)
    elif mode == 'vespa':
        A, ln_L = stats.compute_L_from_DTDs(
          _s1, _s2, _D['t_ons'], _D['t_bre'], _df['mass1'], _df['mass2'],
          _df['mass3'], _df['z'], _df['is_host'], _N_obs, _inputs.visibility_flag)
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
        self.D = Build_Fsps(self._inputs).D

        if self._inputs.data_dir == './../INPUT_FILES/M12/':
            self.add_vespa = True
        else:
            self.add_vespa = False            

        print '\n\n>COMPUTING LIKELIHOOD OF MODELS...\n'
        self.run_analysis()

    def subselect_data(self):

        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)

        f2, f1, f0 = self._inputs.f2, self._inputs.f1, self._inputs.f0
        Dcolor = self.df['Dcolor_' + f2 + f1]

        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
        
        if self._inputs.data_Drange == 'limited':
            Dcolor_cond = np.array(((Dcolor >= self._inputs.Dcolor_min) &
                                   (Dcolor <= 2. * RS_std)), dtype=bool)
        elif self._inputs.data_Drange == 'full':
            Dcolor_cond = np.ones(len(Dcolor), dtype=bool)
        
        hosts = self.df['is_host'][Dcolor_cond].values
        abs_mag = self.df['abs_' + f0][Dcolor_cond].values
        redshift = self.df['z'][Dcolor_cond].values
        Dcolor = Dcolor[Dcolor_cond].values

        self.N_obs = np.sum(hosts)
        self.reduced_df = {
          'absmag': abs_mag, 'Dcolor': Dcolor, 'z': redshift, 'is_host': hosts}

        if self.add_vespa:
            mass_corr = .55
            mass1 = (self.df['vespa1'].values + self.df['vespa2'].values) * mass_corr
            mass2 = self.df['vespa3'].values * mass_corr
            mass3 = self.df['vespa4'].values * mass_corr
            redshift = self.df['z'].values
            hosts = self.df['is_host'].values
            
            self.v_df = {
              'mass1': mass1, 'mass2': mass2, 'mass3': mass3,
              'z': redshift, 'is_host': hosts}            
            self.v_trim_df = {
              'mass1': mass1[Dcolor_cond], 'mass2': mass2[Dcolor_cond],
              'mass3': mass3[Dcolor_cond], 'z': redshift[Dcolor_cond],
              'is_host': hosts[Dcolor_cond]} 

                            
    #@profile
    def write_sSNRL_output(self):
        """This assumes a continuous DTD, but leaves the constant
        (normalization) as a free parameter. Useful for comparing against
        results derived from the VESPA analyses. Write output within the loop.
        """
        slopes = self._inputs.slopes
                        
        fpath = self._inputs.subdir_fullpath + 'likelihoods/sSNRL_s1_s2.csv'
        out = open(fpath, 'w')
        out.write('N_obs=' + str(self.N_obs) + '\n')
        out.write('s1,s2,norm_A,ln_L') 

        output = []

        '''
        #Non-parallel version. Keep for de-debugging.
        for i, v1 in enumerate(slopes):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            for j, v2 in enumerate(slopes):
                output +=  calculate_likelihood(
                  'sSNRL', self._inputs, self.reduced_df, self.N_obs,
                   self.D, v1, v2)
        '''
        
        for i, v1 in enumerate(slopes):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            L_of_v2 = partial(calculate_likelihood, 'sSNRL', self._inputs,
                              self.reduced_df, self.N_obs, self.D, v1)
            pool = Pool(5)
            output += pool.map(L_of_v2,slopes)
            pool.close()
            pool.join()

        for line in output:
            out.write(line) 
        out.close()
    #@profile
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
                              self.v_df, self.N_obs, self.D, v1)
            pool = Pool(2)
            output += pool.map(L_of_v2,slopes)
            pool.close()
            pool.join()

        for line in output:
            out.write(line) 
        out.close()

    #@profile
    def run_analysis(self):
        self.subselect_data()
        self.write_sSNRL_output()
        if self.add_vespa:
            self.write_vespa_nottrim_outputs()

