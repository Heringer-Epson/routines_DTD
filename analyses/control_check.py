#!/usr/bin/env python

import numpy as np
import pandas as pd

def get_full(fpath, redshift_entry):
    df = pd.read_csv(fpath, header=0, low_memory=False)
    ID = df['objID'].values
    z = df[redshift_entry].values.astype(float)
    cond = ((z >= .01) & (z <= .2)) 
    #cond = ((z >= .2) & (z <= .4)) 
    ID_z = ID[cond]
    return ID, ID_z

def get_lists(matching, photo_cut, ctrl, host, redshift_entry):
    #Sample indicates both the source for the control and host datasets.
    run_dir = (
      'hosts/' + ctrl + '_' + host + '_' + matching + '_' + photo_cut + '/')
    fpath = ('./../OUTPUT_FILES/RUNS/' + run_dir + '/data_Dcolor.csv')
    df = pd.read_csv(fpath, header=0, low_memory=False)
    return df['objID'].values



class Check_Control(object):
    """
    Description:
    ------------
    This code reads the data from the standard input file files used in
    M12 and H17 and prints out a comparison between control and hosts..
    
    References:
    -----------
    Dilday+ 2010: http://adsabs.harvard.edu/abs/2010ApJ...713.1026D
    Maoz+ 2012: http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    Heringer+ 2017: http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    """         
    def __init__(self):

        #=-=-=-=-=-=-=-=-=Get full lists of spec. confirmed SNe=-=-=-=-=-=-=-=-
        #M12.
        fpath = './../INPUT_FILES/M12/data_Table.csv'
        M12_T, M12_Tz = get_full(fpath, 'z')

        fpath = './../INPUT_FILES/M12/data_View.csv'
        M12_V, M12_Vz = get_full(fpath, 'z')        
        
        #H17.
        fpath = './../INPUT_FILES/H17/data_Table.csv'
        H17_T, H17_Tz = get_full(fpath, 'z')

        fpath = './../INPUT_FILES/H17/data_View.csv'
        H17_V, H17_Vz = get_full(fpath, 'z')

        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-Get trimmed lists=-=-=-=-=-=-=-=-=-=-=-=-=

        M12_Ttz = get_lists('Table', 'PC', 'M12', 'M12', 'z')
        M12_Vtz = get_lists('View', 'PC', 'M12', 'M12', 'z')

        H17_Ttz = get_lists('Table', 'PC', 'H17', 'H17', 'z')
        H17_Vtz = get_lists('View', 'PC', 'H17', 'H17', 'z')
        

        print 'M12 [NT,NV]: ', len(M12_T), len(M12_V)
        print 'M12 at (0.01 <= z <= 0.2) [NT,NV]: ', len(M12_Tz), len(M12_Vz)
        
        print 'H17 [NT,NV]: ', len(H17_T), len(H17_V)
        print 'H17 at (0.01 <= z <= 0.2) [NTz,NVz]: ', len(H17_Tz), len(H17_Vz)

        print 'M12 trimmed at (0.01 <= z <= 0.2) [NTtz]: ', len(M12_Ttz)
        print 'H17 trimmed at (0.01 <= z <= 0.2) [NVtz]: ', len(H17_Vtz)
        print 'Intersection between above: ', len(set(M12_Ttz).intersection(set(H17_Vtz)))
        print 'Intersection between above: ', len(set(H17_Vtz).intersection(set(M12_Ttz)))

        print 'M12 View trimmed at (0.01 <= z <= 0.2) [NVtz]: ', len(M12_Vtz)
        print 'H17 Table trimmed at (0.01 <= z <= 0.2) [NTtz]: ', len(H17_Ttz)


        
if __name__ == '__main__':
    Check_Control()
 
