#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd
from astropy import units as u

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import stats

sys.path.append(os.path.join(os.environ['PATH_ssnarl'], 'src'))
from SN_rate import Model_Rates
from build_fsps_model import Build_Fsps

class Mock_Quantities(object):
    """
    Code Description
    ----------    
    TBW. 

    Parameters:
    -----------
    A : ~float
        DTD normalization.
    s : ~float
        DTD slope.
        
    Outputs:
    --------
    ./../OUTPUT_FILES/MOCK_SAMPLE/X.csv
    """
    
    def __init__(self, inputs, A, s1, s2, survey_t):
        self.inputs, self.A, self.s1, self.s2 = inputs, A, s1, s2
        self.survey_t = survey_t
        self.df, self.D = None, None
        self.M = {}
        
        self.run_calculations()
        
    def get_mock_data(self):      
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/data_mock.csv'
        self.df = pd.read_csv(fpath, header=0, low_memory=False)
          
    def compute_lum(self):
        L_FSPS = stats.mag2lum(self.df['petroMag_r'])
        M_FSPS = (self.df['vespa1'] + self.df['vespa2'] + self.df['vespa3']
                  + self.df['vespa4'])
        M_target = 10.**self.df['logmass']
        self.df['L_r'] = L_FSPS * M_target / M_FSPS

    def compute_true_rates(self):
        #Get generic rate prediction as a function of age for each tau.
        self.D = Build_Fsps(self.inputs).D #Contains FSPS info for multiple taus.
        for tau in self.inputs.tau_list:
            TS = str(tau.to(u.yr).value / 1.e9)
            self.M[TS] = Model_Rates(self.inputs, self.D, TS, self.s1, self.s2)
            
        #For each mock galaxy, assign a true rate based on its mass. Note
        #that this rate is dimensionless for an assumed survey of visibity
        #window of 1 yr.
        
        sSNR = []
        for (logage,tau,logmass) in zip(
          self.df['logage'],self.df['tau'],self.df['logmass']):
            cond = (abs(self.D['logage'] - logage) < 0.001)
            sSNR.append(self.A * 10.**logmass * self.M[str(tau)].sSNRm[cond][0]
                        * self.survey_t.to(u.yr).value)

        self.df['sSNR'] = np.array(sSNR)

    def sprinkle_SNe(self):
        
        N_SN = []
        for sSNR in self.df['sSNR']:
            N_SN.append(np.random.poisson(sSNR,1)[0])
        is_host = np.array(N_SN).astype(bool)
        self.df['is_host'] = is_host

        N_SN = np.array(N_SN).astype(int)
        print len(N_SN[N_SN == 2]), ' galaxies have hosted more than 1 supernova.'        
        print len(is_host[is_host == True]), 'galaxies have hosted SNe.'
        
    def save_output(self):
        fpath = './../OUTPUT_FILES/MOCK_SAMPLE/data_mock_final.csv'
        df = pd.DataFrame(self.df)
        df.to_csv(fpath)
            
    def run_calculations(self):
        self.get_mock_data()
        self.compute_lum()
        self.compute_true_rates()
        self.sprinkle_SNe()
        self.save_output()
