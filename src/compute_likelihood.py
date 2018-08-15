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
    return N_expected, ln_L 

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
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihood_s1_s2.csv'
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihood_A_s.csv'
    """       
    def __init__(self, _inputs):
        self._inputs = _inputs
                
        self.df = None        
        self.reduced_df = None        
        self.N_obs = None

        print '\n\n>COMPUTING LIKELIHOOD OF MODELS...\n'
        self.run_analysis()

    def retrieve_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        self.df = pd.read_csv(fpath, header=0)

    def subselect_data(self):

        f1, f2 = self._inputs.filter_1, self._inputs.filter_2
        Dcolor = self.df['Dcolor_' + f2 + f1]

        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_std = abs(float(np.loadtxt(fpath, delimiter=',', skiprows=1,
                                      usecols=(1,), unpack=True)))
                               
        Dcolor_cond = ((Dcolor >= self._inputs.Dcolor_min) &
                       (Dcolor <= 2. * RS_std))

        
        hosts = self.df['is_host'][Dcolor_cond].values
        abs_mag = self.df['abs_' + f1][Dcolor_cond].values
        redshift = self.df['z'][Dcolor_cond].values
        Dcolor = Dcolor[Dcolor_cond].values

        #ctrl_cond = np.logical_not(self.hosts)
        self.N_obs = np.sum(hosts)
        self.reduced_df = {
          'absmag': abs_mag, 'Dcolor': Dcolor, 'z': redshift, 'is_host': hosts}
                        
    def write_outputs(self):
        """This assumes a continuous DTD, but leaves the constant
        (normalization) as a free parameter. Useful for comparing against
        results derived from the VESPA analyses. Write output within the loop.
        """
        slopes_A_s = self._inputs.slopes_A_s[::-1]
        slopes_s_s = self._inputs.slopes_s_s[::-1]
        As = self._inputs.A[::-1]
        t_ons = self._inputs.t_onset.to(u.Gyr).value
        t_cut = self._inputs.t_cutoff.to(u.Gyr).value

        fnames = ['s1_s2', 'A_s']#[1:]#[0:1]#
        modes = ['s1/s2', 's1=s2']#[1:]#[0:1]#
        labels = ['s1,s2,', 'A,s,']#[1:]#[0:1]#
        norm_flags = [True, False]#[1:]#[0:1]#
        pars = [(slopes_s_s, slopes_s_s), (As, slopes_A_s)]#[1:]#[0:]#

        for fname, mode, par, label, norm_flag in\
            zip(fnames,modes,pars,labels,norm_flags):
            
            par1, par2 = par[0], par[1]
            
            fpath = self._inputs.subdir_fullpath + 'likelihood_' + fname + '.csv'
            out = open(fpath, 'w')
            
            out.write('-------- General info --------\n')
            out.write('sfh_type: ' + self._inputs.sfh_type + '\n')
            out.write('t_onset [Gyr]: ' + str(int(t_ons)) + '\n')
            out.write('t_cutoff [Gyr]: ' + str(int(t_cut)) + '\n')
            out.write('N_obs: ' + str(self.N_obs) + '\n')
            out.write('-------- Columns --------\n')
            out.write(label + 'N_expected,ln_L\n') 

            N_expected, ln_L = [], [] 
            for i, v1 in enumerate(par1):
                print 'Calculating set ' + str(i + 1) + '/' + str(len(par1))
                L_of_v2 = partial(
                  calculate_likelihood, self._inputs, self.reduced_df, mode,
                  self.N_obs, norm_flag, v1)
                pool = Pool(5)
                out1, out2 = zip(*pool.map(L_of_v2, par2))
                pool.close()
                pool.join()
                N_expected += out1
                ln_L += out2
            
            for k, v2 in enumerate(par2):
                v2_str = str(format(v2, '.5e'))
                for j, v1 in enumerate(par1): 
                    v1_str = str(format(v1, '.5e'))
                    idx = j * len(par2) + k
                    line = (v1_str + ',' + v2_str + ',' + str(N_expected[idx])
                            + ',' + str(ln_L[idx]) + '\n')
                    out.write(line)   
            out.close()
   
    def run_analysis(self):
        self.retrieve_data()
        self.subselect_data()
        self.write_outputs()

if __name__ == '__main__':
    from input_params import Input_Parameters as class_input
    Get_Likelihood(class_input(case='SDSS_gr_Maoz'))

