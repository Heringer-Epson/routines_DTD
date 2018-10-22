#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
from astropy import units as u

def count_objs(df, _fpath, section_name, host_types, out):   
    df = df.replace('null', np.nan)    
    host_cond = df['is_host'].astype(bool).values
    N_hosts = df[host_cond].shape[0]
    N_ctrl = df.shape[0]
    median_redshift = str(format(np.median(df['z'].values), '.3f'))
    median_uErr = str(format(np.median(df['petroMagErr_u'].values.astype(float)), '.3f'))
    median_gErr = str(format(np.median(df['petroMagErr_g'].values), '.3f'))
    median_rErr = str(format(np.median(df['petroMagErr_r'].values), '.3f'))
    
    out.write('\n\n------------------- ' + section_name + ' -------------------')
    out.write('\nFile: ' + _fpath)
    out.write('\nN_ctrl = ' + str(N_ctrl))
    out.write('\nN_hosts ' + str(host_types) + ' = ' + str(N_hosts))
    out.write('\nmedian redshift = ' + median_redshift)
    out.write('\nmedian u_err = ' + median_uErr)
    out.write('\nmedian g_err = ' + median_gErr)
    out.write('\nmedian r_err = ' + median_rErr + '\n')

class Write_Record(object):
    """
    Description:
    ------------
    This code produces a file containing the number of galaxies in the host and
    control samples before and after processing the data.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/RUNS/$RUN_DIR/record.dat
    """    
    def __init__(self, _inputs):
        self._inputs = _inputs
        self.df = None

        self.out = open(self._inputs.subdir_fullpath + 'record.dat', 'w')
        self.data_type = {'z': float, 'petroMagErr_u': float,
                          'petroMagErr_g': float, 'petroMagErr_r': float}
        
        #Check if data_Dcolor.csv exists. If so, make the record output file.
        #This routine relies on the data files have been created beforehand.
        if os.path.isfile(self._inputs.subdir_fullpath + 'data_Dcolor.csv'):
            self.run_record()

    def make_record(self):
        self.out.write('\n\n')
        self.out.write('--------------------')
        self.out.write('\nrun date: ' + time.strftime('%d/%m/%Y') + '\n')
        self.out.write('--------------------')
        self.out.write('\n\n')
        self.out.write('---------------------Model params---------------------\n')
        self.out.write('Colour: ' + self._inputs.filter_2 + '-'\
                  + self._inputs.filter_1 + '\n')
        self.out.write('Spectral_library: ' + self._inputs.spec_lib + '\n')
        self.out.write('Isochrone_library: ' + self._inputs.isoc_lib + '\n')
        self.out.write('IMF: ' + self._inputs.imf_type + '\n')
        self.out.write('SFH: ' + self._inputs.sfh_type + '\n')
        self.out.write('Metallicity: ' + str(self._inputs.Z) + '\n')
        self.out.write('t_onset: ' + str(format(
          self._inputs.t_onset.to(u.yr).value / 1.e9, '.2f')) + ' Gyr \n')
        self.out.write('t_cutoff: ' + str(format(
          self._inputs.t_cutoff.to(u.yr).value / 1.e9, '.2f')) + ' Gyr \n\n\n')
        self.out.write('--------------------Data Selection--------------------\n')
        self.out.write('Affects the objects under the "Processed" section.\n')
        self.out.write(str(self._inputs.ra_min) + ' < ra < '
                  + str(self._inputs.ra_max) + ' \n')            
        self.out.write(str(self._inputs.dec_min) + ' < dec < '
                  + str(self._inputs.dec_max) + ' \n') 
        self.out.write(str(self._inputs.redshift_min) + ' <= redshift <= '
                  + str(self._inputs.redshift_max) + ' \n')
        self.out.write(str(self._inputs.petroMag_u_min) + ' <= u_ext < '
                  + str(self._inputs.petroMag_u_max) + ' \n')
        self.out.write(str(self._inputs.petroMag_g_min) + ' <= g_ext < '
                  + str(self._inputs.petroMag_g_max) + ' \n')
        self.out.write(str(self._inputs.ext_r_min) + ' <= r_ext < '
                  + str(self._inputs.ext_r_max) + ' \n')
        self.out.write('u_err <= ' + str(self._inputs.uERR_max) + '\n')
        self.out.write('g_err <= ' + str(self._inputs.gERR_max) + '\n')
        self.out.write('r_err <= ' + str(self._inputs.rERR_max) + '\n')

    def original_data(self):
        if 'M12' in self._inputs.data_dir.split('/'):
            fpath_all = self._inputs.data_dir + 'original_files/radec5.dat'
        elif 'H17' in self._inputs.data_dir.split('/'):
            fpath_all = self._inputs.data_dir + 'formatted_IDs.csv'
        with open(fpath_all, 'r') as inp:
            N_all = sum(1 for line in inp) - 1 #Does not count the header.

        fpath_hosts = './../INPUT_FILES/' + self._inputs.hosts_from\
                      + '/formatted_hosts.csv'       
        with open(fpath_hosts, 'r') as inp:
            N_hosts = sum(1 for line in inp) - 1 #Does not count the header.
            
        self.out.write('\n\n-------------------Original Data-------------------')
        self.out.write('\nFile sample: ' + fpath_all)
        self.out.write('\nFile hosts: ' + fpath_hosts)
        self.out.write('\nN_sample = ' + str(N_all))
        self.out.write('\nN_hosts (all) = ' + str(N_hosts) + '\n')    

    def subselected_data(self):
        fpath = self._inputs.subdir_fullpath + 'data_merged.csv'
        df = pd.read_csv(fpath, header=0, dtype=self.data_type, low_memory=False)
        count_objs(df, fpath, 'Hosts Merged', self._inputs.host_class, self.out)
        
        fpath = self._inputs.subdir_fullpath + 'data_absmag.csv'
        df = pd.read_csv(fpath, header=0, dtype=self.data_type, low_memory=False)
        count_objs(df, fpath, 'Processed', self._inputs.host_class, self.out)

    def Dcolor_data(self):

        #Get RS info for trimming the data.
        fpath = self._inputs.subdir_fullpath + 'RS_fit.csv'
        RS_mu, RS_std = np.loadtxt(fpath, delimiter=',', skiprows=1,
                                   usecols=(0,1), unpack=True)                
        RS_mu, RS_std = float(RS_mu), float(abs(RS_std))

        fpath = self._inputs.subdir_fullpath + 'data_Dcolor.csv'
        df = pd.read_csv(fpath, header=0, dtype=self.data_type, low_memory=False)
        f1, f2 = self._inputs.filter_1, self._inputs.filter_2
        Dcolor = df['Dcolor_' + f2 + f1].values
        Dcolor_cond = ((Dcolor >= self._inputs.Dcolor_min)
                       & (Dcolor <= 2. * RS_std))        

        df = df[Dcolor_cond]
        count_objs(df, fpath, 'Dcolor trimmed', self._inputs.host_class, self.out)
        self.out.write('Criterion: ' + str(self._inputs.Dcolor_min) + ' <= '\
                       + 'Dcolor(' + f2 + '-' + f1 + ') <= ' + str(2. * RS_std))
        self.out.write('\n\n\n')
        
    def run_record(self):
        self.make_record()
        self.original_data()
        self.subselected_data()
        self.Dcolor_data()
        self.out.close()
