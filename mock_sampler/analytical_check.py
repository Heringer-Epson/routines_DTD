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

#As used in the code.
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

tau = 10. * u.Gyr
#logage = 8.85
logage = 8.7
mass = 1.e9

class Analytical_Check(object):
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

        self.D = Build_Fsps(self.inputs).D #Contains FSPS info for multiple taus.
        self.M = {}
        
        self.run_calculations()

    def compute_true_rates(self):
        #Get generic rate prediction as a function of age for each tau.
        
        TS = str(tau.to(u.yr).value / 1.e9)
        model_true = Model_Rates(self.inputs, self.D, TS, self.s1, self.s2)
        
        cond = (abs(self.D['logage'] - logage) < 0.001)
        #print 'True rate', self.A * model_true.sSNR[cond]
        print 'True rate', self.A * model_true.sSNRm[cond] * mass
    
    def compute_SFHR_rates(self):
        
        for _tau in self.inputs.tau_list:
            TS = str(_tau.to(u.yr).value / 1.e9)
            self.D[TS] = norm(_tau.to(u.Gyr).value, self.D['logage']) #FSPS logage.
        
        T = 10.**logage / 1.e9
        TS_tau = str(tau.to(u.yr).value / 1.e9)
        m1,m2,m3,m4 = get_binned_mass(
          T, tau.to(u.Gyr).value, self.inputs.t_onset.to(u.Gyr).value, self.D[TS_tau])

        mformed = (m1 + m2 + m3 + m4)
        psi1, psi2, psi3 = stats.binned_DTD_rate(
          self.s1, self.s2, self.D['t_ons'], self.D['t_bre'])
        
        SFHR_rate = self.A * (psi1 * m2 + psi2 * m3 + psi3 * m4) / mformed * mass
        #SFHR_rate = self.A * (psi1 * m2 + psi2 * m3 + psi3 * m4)
        
        #print self.D['logage']
        
        #print 'norm', self.D[TS_tau]
        #print 'T', T
        #print 'tau', TS_tau
        print m1, m2, m3, m4
        print 'SFHR rate', SFHR_rate
        

    def run_calculations(self):
        self.compute_true_rates()
        self.compute_SFHR_rates()
