#!/usr/bin/env python

import numpy as np
import pandas as pd

def get_trimmed(matching, photo_cut, ctrl, host, peculiar=None):
    #Sample indicates both the source for the control and host datasets.
    run_dir = (
      'hosts/' + ctrl + '_' + host + '_' + matching + '_' + photo_cut + '/')
    fpath = ('./../OUTPUT_FILES/RUNS/' + run_dir + '/data_Dcolor.csv')
    #fpath = ('./../OUTPUT_FILES/RUNS/' + run_dir + '/data_merged.csv')
    #fpath = ('./../OUTPUT_FILES/RUNS/' + run_dir + '/data_absmag.csv')
    df = pd.read_csv(fpath, header=0, low_memory=False)
    #Condition for hosts of spectroscopic events (SNIa).
    if peculiar:
        st = df['subtype'].values
        cond = ((df['is_host'].values == True)
                & (df['Classification'].values == 'SNIa')        
                & ((st == '3') | (st == '4') | (st == '5')))
    else:
        cond = ((df['is_host'].values == True)
                & (df['Classification'].values == 'SNIa'))

    ID = df[cond]['IAUName'].values
    return ID

def get_lists(fpath, classif, redshift_entry):
    df = pd.read_csv(fpath, header=0, low_memory=False)
    if classif is not None:
        cond = (df['Classification'].values == 'SNIa')
        df = df[cond]
    ID = df['IAUName'].values
    z = df[redshift_entry].values.astype(float)
    cond = ((z >= .01) & (z <= .2)) 
    ID_z = ID[cond]
    return ID, ID_z

class Check_Hosts(object):
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
        #D10.
        fpath = './../INPUT_FILES/D10/D10_SNeIa.csv'
        D10, D10_z = get_lists(fpath, None, 'SN_z')
        #M12.
        fpath = './../INPUT_FILES/M12/formatted_hosts.csv'
        M12, M12_z = get_lists(fpath, 'SNIa', 'SN_z')
        #H17.
        fpath = './../INPUT_FILES/H17/formatted_hosts.csv'
        H17, H17_z = get_lists(fpath, 'SNIa', 'SN_z')
        #S18.
        fpath = './../INPUT_FILES/S18/formatted_hosts.csv'
        S18, S18_z = get_lists(fpath, 'SNIa', 'zspecHelio')

        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-Get trimmed lists=-=-=-=-=-=-=-=-=-=-=-=-=

        M12_t = get_trimmed('Table', 'noPC', 'M12', 'M12')
        H17_t = get_trimmed('View', 'noPC', 'H17', 'H17')

        print 'D10 [N,Nz]: ', len(D10), len(D10_z)
        print 'M12 [N,Nz]: ', len(M12), len(M12_z)
        print 'H17 [N,Nz]: ', len(H17), len(H17_z)
        print 'S18 [N,Nz]: ', len(S18), len(S18_z)

        print '\m'


        print 'M12 trimmed [Nz]: ', len(M12_t)
        print 'H17 trimmed [Nz]: ', len(H17_t)

        print '\n'

        aux = set(H17_t) - set(M12_t)
        print 'Nz in H17 trimmed but not in M12s trimmed list: ', len(aux)
        aux = set(M12_t) - set(H17_t)
        print 'Nz in M12 trimmed but not in H17s trimmed list: ', len(aux)
        
        print '\n'
        
        aux = set(H17_t) - set(M12_z)
        print 'Nz in H17 trimmed but not in M12s full list: ', len(aux)
        aux = set(H17_t) - set(D10_z)
        print 'Nz in H17 trimmed but not in D10s full list: ', len(aux)
        aux = set(M12_t) - set(H17_z)
        print 'Nz in M12 trimmed but not in H17s full list: ', len(aux)

        print '\n'

        aux = set(M12_z) - set(S18_z)
        print 'Nz in M12 but not in S18s list: ', len(aux)
        aux = set(H17_t) - set(S18_z)
        print 'Nz in H17 trimmed but not in S18s list: ', len(aux)

        H17_S18_t = get_trimmed('View', 'PC', 'H17', 'S18')
        print 'Nz of S18 hosts in H17 trimmed: ', len(H17_S18_t)
        H17_S18_t_pec = get_trimmed('View', 'PC', 'H17', 'S18', peculiar=True)
        print 'Nz of peculiar S18 hosts in H17 trimmed: ', len(H17_S18_t_pec)
        

        

if __name__ == '__main__':
    Check_Hosts()
 
