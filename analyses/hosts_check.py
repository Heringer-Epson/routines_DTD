#!/usr/bin/env python

import numpy as np
import pandas as pd

class Check_Hosts(object):
    """
    Description:
    ------------
    This code reads the data from the standard input file files used in
    M12 and H17 and prints out a comparison between control and hosts. In
    particular, it shows that the identification of hosts is in disagreement
    between the two papers.
    
    References:
    -----------
    Maoz+ 2012: http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    Gao & Pritchet 2013: http://adsabs.harvard.edu/abs/2013AJ....145...83G
    Heringer+ 2017: http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    """         
    def __init__(self):

        self.host_ID_M12 = None
        self.host_ID_G13 = None
        self.host_ID_H17 = None

        self.run_check()
        
    def read_data(self):

        #Read data used in Maoz+ 2012.
        fpath = ('./../../INPUT_FILES/Maoz_file/data_Tablematched.csv')
        self.df_M12 = pd.read_csv(fpath, header=0)
        self.df_M12 = self.df_M12.replace('null', np.nan)
        for key in self.df_M12.keys():
            if key != 'objID' and key != 'specobjID':
                self.df_M12[key] = self.df_M12[key].astype(float) 
        self.df_M12['n_SN'] = self.df_M12['n_SN'].astype(bool)
        self.df_M12['objID'] = self.df_M12['objID'].astype(str)

        #Read data used in Gao+ 2013.
        fpath = ('./../../INPUT_FILES/Gao_hosts/Table_1_Gao2013.csv')
        self.df_G13 = pd.read_csv(fpath, header=0)
        self.df_G13 = self.df_G13.replace('null', np.nan)
        self.df_G13['objID'] = self.df_G13['objID'].astype(str)
               
        #Read data used in Heringer+ 2017.
        #fpath = ('./../../INPUT_FILES/paper1/data_Filematched.csv')
        fpath = ('./../../INPUT_FILES/paper1/data_FileCASmatched.csv')
        self.df_H17 = pd.read_csv(fpath, header=0)
        self.df_H17 = self.df_H17.replace('null', np.nan)
        for key in self.df_H17.keys():
            if key != 'objID' and key != 'specobjID':
                self.df_H17[key] = self.df_H17[key].astype(float) 
        self.df_H17['n_SN'] = self.df_H17['n_SN'].astype(bool)
        self.df_H17['objID'] = self.df_H17['objID'].astype(str)

        #Read extra data for the hosts used in Heringer+ 2017.
        fpath = ('./../../INPUT_FILES/paper1/hosts_info.csv')
        self.df_hostH17 = pd.read_csv(fpath, header=0)
        self.df_hostH17 = self.df_hostH17.replace('null', np.nan)
        self.df_hostH17['host_objID'] = self.df_hostH17['host_objID'].astype(str)

        #Read extra data for the spec hosts used in Maoz+ 2012.
        fpath = ('./../../INPUT_FILES/Maoz_file/hosts_info_spec.csv')
        self.df_hostM12 = pd.read_csv(fpath, header=0)
        self.df_hostM12 = self.df_hostM12.replace('null', np.nan)

    def compare_host_IDS(self):
        host_M12 = self.df_M12['n_SN'].values
        host_H17 = self.df_H17['n_SN'].values
        
        self.host_ID_M12 = self.df_M12['objID'].values[host_M12].astype(str)
        self.host_ID_G13 = self.df_G13['objID'].values.astype(str)
        self.host_ID_H17 = self.df_H17['objID'].values[host_H17].astype(str)
        self.host_all_H17 = self.df_hostH17['host_objID'].values.astype(str)

        ctrl_ID_M12 = self.df_M12['objID'].values.astype(str)
        ctrl_ID_H17 = self.df_H17['objID'].values.astype(str)
        

        print 'M12, G13 and H17 # of hosts:', len(self.host_ID_M12), len(self.host_ID_G13), len(self.host_ID_H17)
        print 'M12 and H17 # of ctrl:', len(ctrl_ID_M12), len(ctrl_ID_H17)
        print 'Hosts in common:', len(set(self.host_ID_M12).intersection(self.host_ID_H17))
        print 'Ctrl in common:', len(set(ctrl_ID_M12).intersection(ctrl_ID_H17))
        print '\n'
        print 'H17 hosts in M12s whole sample:', len(set(self.host_ID_H17).intersection(ctrl_ID_M12))
        print 'M12 hosts in H17s whole sample:', len(set(self.host_ID_M12).intersection(ctrl_ID_H17))        
        print 'G13 hosts in H17s whole sample:', len(set(self.host_ID_G13).intersection(ctrl_ID_H17)) 
        print 'G13 hosts in M12s whole sample:', len(set(self.host_ID_G13).intersection(ctrl_ID_M12)) 
        #print np.sort(host_ID_G13.astype(str))

    def check_positions(self):

        #Check that the hosts from H17 are present in finSNe1a_SDSS.cat.
        #common_hosts = list(set(self.host_all_H17).intersection(self.host_ID_H17))
        #print len(common_hosts)
        
        common = list(set(self.host_ID_M12).intersection(self.host_ID_H17))
        H17_notin_M12 = list(set(self.host_ID_H17) - set(self.host_ID_M12))
        M12_notin_H17 = list(set(self.host_ID_M12) - set(self.host_ID_H17))
        diff = list(set(self.host_ID_M12).symmetric_difference(self.host_ID_H17)) #not useful
        
        common_in_H17 = self.df_H17[self.df_H17['objID'].isin(list(common))]
        common_in_M12 = self.df_M12[self.df_M12['objID'].isin(list(common))]

        diff_in_H17 = self.df_H17[self.df_H17['objID'].isin(list(H17_notin_M12))]
        diff_in_M12 = self.df_M12[self.df_M12['objID'].isin(list(M12_notin_H17))]

        for i, (ra_H17,dec_H17) in enumerate(zip(diff_in_H17['ra'].values,diff_in_H17['dec'].values)):
            for j, (ra_M12,dec_M12) in enumerate(zip(diff_in_M12['ra'].values,diff_in_M12['dec'].values)):
                if abs(ra_H17 - ra_M12) < 1. and abs(dec_H17 - dec_M12) < .1:
                    pass
                    #print ra_H17, dec_H17, diff_in_H17['objID'].values[i], diff_in_H17['z'].values[i], ra_M12, dec_M12, diff_in_M12['objID'].values[j], diff_in_M12['redshift'].values[j], diff_in_M12['z'].values[j]

    def check_host_IDS(self):


        spec_hostH17 = self.df_hostH17[self.df_hostH17['host_objID'].isin(list(self.host_ID_H17))]
        SNID_H17 = np.array([ID[2::] for ID in spec_hostH17['SDSS_SN_ID'].values]).astype(str)
        SNID_M12 = self.df_hostM12['SDSS_SN_ID'].values.astype(str)
        unique_hostsM12 = list(set(SNID_M12) - set(SNID_H17))
        unique_hostsH17 = list(set(SNID_H17) - set(SNID_M12))
        common_SNe = list(set(SNID_H17).intersection(SNID_M12))
        print '\n'
        print 'SNe (spec) in M12', len(SNID_M12)
        print 'SNe In M12 but not in H17:', len(unique_hostsM12)
        print 'SNe In H17 but not in M12:', len(unique_hostsH17)
        print 'common SNe', len(common_SNe)
        
        print 'unique in M12', unique_hostsM12, '\n'
        print 'unique in H17', unique_hostsH17
            
    def run_check(self):
        self.read_data()
        self.compare_host_IDS()
        self.check_positions()
        self.check_host_IDS()

if __name__ == '__main__':
    Check_Hosts()
 
