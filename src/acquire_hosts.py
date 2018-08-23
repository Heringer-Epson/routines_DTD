#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd

class Acquire_Hosts(object):
    """
    Description:
    ------------
    This code reads the input data file containing the whole sample of galaxies
    and a data file containing the DR7 objIDs of the hosts. It then mergers
    (joins) the host data into the whole sample. A new column 'is_host' is also
    created, specifying whether each galaxy is a host or not.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/data_merged.csv
    """
    def __init__(self, _inputs):
        self._inputs = _inputs
        self.df = None
        self.perform_run()

    def join_data(self):
        #Read the relevant data.
        fpath_all = (
          self._inputs.data_dir + 'data_' + self._inputs.matching + '.csv')
        fpath_hosts = ('./../INPUT_FILES/' + self._inputs.hosts_from
                       + '/formatted_hosts.csv')
        self.df = pd.read_csv(fpath_all, header=0, dtype={'objID': str})
        df_hosts = pd.read_csv(
          fpath_hosts, header=0, dtype={'objID': str, 'CID': str})

        #Remove or keep hosts of SNe observed during engineering time (2004).
        if not self._inputs.hosts_from_2004:
            year = np.array([x[0:4] for x in df_hosts['IAUName'].values])
            cond = (year != '2004')
            df_hosts = df_hosts[cond]

        #Some values retrieved by SDSS might be 'null' strings, which are then
        #read as strings. Replace those with np.nan and convert columns to floats.
        self.df = self.df.replace('null', np.nan)

        #Make the 'objID' column the index in the host's dataframe. This is
        #required for using pandas 'join'.
        df_hosts.set_index('objID', inplace=True)
        
        #Select only the hosts whose classification (SNIa, zSNIA, pSNIa) is in
        #self._inputs.host_class
        df_hosts = df_hosts[df_hosts['Classification'].isin(self._inputs.host_class)]
        
        #Merge the hosts with the control (whole) sample. 
        self.df = self.df.join(
          df_hosts[['CID', 'IAUName', 'Classification']], on='objID')
        
        #Create new column which will then specify if the galaxy is a host
        #(True) or not (False).
        self.df['is_host'] = self.df.apply(
          lambda row: isinstance(row['CID'], str), axis=1)        
                
    def save_output(self):
        fpath = self._inputs.subdir_fullpath + 'data_merged.csv'
        self.df.to_csv(fpath)

    def perform_run(self):
        self.join_data()
        self.save_output()
