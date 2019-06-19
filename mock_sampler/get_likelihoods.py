#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd
from astropy import units as u
from multiprocessing import Pool
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import stats
import core_funcs

sys.path.append(os.path.join(os.environ['PATH_ssnarl'], 'src'))
from build_fsps_model import Build_Fsps
from Dcolor2sSNRL_gen import Generate_Curve


#slopes = np.arange(-3., 0.0001, 0.05)
slopes = np.arange(-3., 0.0001, 0.1)
#slopes = [-1.]

def calculate_likelihood(mode, _inputs, _df, _N_obs, _D, _s1, _s2):
    if mode == 'sSNRL':
        Sgen = Generate_Curve(_inputs, _D, _s1, _s2)
        if _inputs.model_Drange == 'reduced':
            x, y = Sgen.Dcolor_at10Gyr[::-1], Sgen.sSNRL_at10Gyr[::-1]
        elif _inputs.model_Drange == 'extended':
            x, y = Sgen.Dcd_fine, Sgen.sSNRL_fine
        sSNRL = np.asarray(core_funcs.interp_nobound(x, y, _df['Dcolor']))
        A, ln_L = stats.compute_L_using_sSNRL(
          sSNRL, _df['Dcolor'], _df['absmag'], _df['z'],
          _df['is_host'], _N_obs, _inputs.visibility_flag)
    elif mode == 'vespa':
        A, ln_L = stats.compute_L_from_DTDs(
          _s1, _s2, _D['t_ons'], _D['t_bre'], _df['mass1'], _df['mass2'],
          _df['mass3'], _df['z'], _df['is_host'], _N_obs, _inputs.visibility_flag)
    line = '\n' + str(format(_s1, '.5e')) + ',' + str(format(_s2, '.5e'))\
      + ',' + str(A) + ','  + str(ln_L)
    return line 

class Get_Likelihoods(object):
    """
    Code Description
    ----------    
    Given the masses, colors and hosts in the mock sample, compute likelihood
    contours for the CL and SFHR methods.

    Parameters:
    -----------
    A : ~float
        DTD normalization.
    s : ~float
        DTD slope.
        
    Outputs:
    --------
    TBW
    """
    
    def __init__(self, inputs, A, s1, s2, survey_t):
        self.inputs, self.A, self.s1, self.s2 = inputs, A, s1, s2
        self.survey_t = survey_t
        
        self.D = Build_Fsps(self.inputs).D
        self.N_obs = None
        
        self.run_calculations()
        
    def get_data(self):      
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/data_mock_final.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
        
        is_host = self.df['is_host'].values
        self.N_obs = float(len(is_host[(is_host == True)]))

        m_fsps = self.df['vespa1'] + self.df['vespa2'] + self.df['vespa3'] + self.df['vespa4'] 
        self.df['m_fsps'] = m_fsps

    def compute_CL_rates(self):

        #This does not require mass corrections.
        #This does not require detection efficiency corrections.
        #This does not require visibility time corrections if survey_t=1yr.
        self.inputs.visibility_flag = False
        self.inputs.model_Drange = 'extended'

        output = []
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/sSNRL_s1_s2.csv'
        out = open(fpath, 'w')
        out.write('N_obs=' + str(self.N_obs) + '\n')
        out.write('s1,s2,norm_A,ln_L') 

        abs_mag = (
          -2.5 * (self.df['logmass'].values - np.log10(self.df['m_fsps'].values))
          + self.df['petroMag_r'].values)

        CL_df = {
          'absmag': abs_mag, 'Dcolor': self.df['Dcolor_gr'].values,
          'z': self.df['z'], 'is_host': self.df['is_host'].values}

        for i, v1 in enumerate(slopes):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            L_of_v2 = partial(calculate_likelihood, 'sSNRL', self.inputs,
                              CL_df, self.N_obs, self.D, v1)
            pool = Pool(5)
            output += pool.map(L_of_v2,slopes)
            pool.close()
            pool.join()

        #Non-parallel version. Keep for de-debugging.
        '''
        for i, v1 in enumerate(slopes):
            print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            for j, v2 in enumerate(slopes):
                output +=  calculate_likelihood(
                  'sSNRL', self.inputs, CL_df, self.N_obs, self.D, v1, v2)
        '''
        for line in output:
            out.write(line) 
        out.close()

    def compute_SFHR_rates(self):

        #This does not require mass corrections.
        #This does not require detection efficiency corrections.
        #This does not require visibility time corrections if survey_t=1yr.
        self.inputs.visibility_flag = False
        
        output = []       
        
        #Below, one needs to divide by the formed mass. In other words, it's
        #mecessary to first obtain the binned rate for a galaxy, then make
        #that rate per unit of mass, then multiply it by the total mass.
        m_factor = np.divide(10.**self.df['logmass'].values,self.df['m_fsps'].values)
        #m_factor = 10.**self.df['logmass'].values

        SFHR_df = {
          'mass1': np.multiply(self.df['vespa2'].values,m_factor),
          'mass2': np.multiply(self.df['vespa3'].values,m_factor),
          'mass3': np.multiply(self.df['vespa4'].values,m_factor),
          'z': self.df['z'].values, 'is_host': self.df['is_host'].values}            
        
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/vespa_s1_s2.csv'
        out = open(fpath, 'w')
        out.write('N_obs=' + str(self.N_obs) + '\n')
        out.write('s1,s2,norm_A,ln_L') 

        #Non-parallel version. Keep for de-debugging.
        for i, v1 in enumerate(slopes):
            #print 'Calculating set ' + str(i + 1) + '/' + str(len(slopes))
            for j, v2 in enumerate(slopes):
                output.append(calculate_likelihood(
                  'vespa', self.inputs, SFHR_df, self.N_obs,
                   self.D, v1, v2))        

        for line in output:
            out.write(line) 
        out.close()
            
    def run_calculations(self):
        self.get_data()
        self.compute_CL_rates()
        self.compute_SFHR_rates()

