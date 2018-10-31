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
        #Read data used in Gao+ 2013.
        #fpath = './../INPUT_FILES/G13/Table_1_Gao2013.csv'
        #self.df_G13 = pd.read_csv(fpath, header=0, low_memory=False)
        #self.df_G13 = self.df_G13.replace('null', np.nan)
        #self.df_G13['objID'] = self.df_G13['objID'].astype(str)
        #self.host_ID_G13 = self.df_G13['objID'].values.astype(str)

               
        #Read data used in Heringer+ 2017.
        eng_SNe = ['2004ia', '2004il', '2004hu']
        fpath = './../OUTPUT_FILES/RUNS/H17/data_merged.csv'
        self.df_H17 = pd.read_csv(fpath, header=0, low_memory=False, dtype='str')
        self.df_H17 = self.df_H17.replace('null', np.nan)
        self.H17_objID = self.df_H17['objID'].values
        host_cond = ((self.df_H17['is_host'] == 'True')
                     & (self.df_H17['IAUName'] != '2004ia')
                     & (self.df_H17['IAUName'] != '2004il')
                     & (self.df_H17['IAUName'] != '2004hu')).values
        self.H17_host = self.H17_objID[host_cond]
        self.H17_hname = self.df_H17['IAUName'].values[host_cond]

        #Read data used in Maoz+ 2012.
        fpath = './../OUTPUT_FILES/RUNS/M12/data_merged.csv'
        self.df_M12 = pd.read_csv(fpath, header=0, low_memory=False, dtype='str')
        self.df_M12 = self.df_M12.replace('null', np.nan)
        self.M12_objID = self.df_M12['objID'].values
        host_cond = (self.df_M12['is_host'] == 'True').values
        self.M12_host = self.M12_objID[host_cond]
        self.M12_hname = self.df_M12['IAUName'].values[host_cond]
        
        spec_cond = ((self.df_M12['is_host'] == 'True')
                     & (self.df_M12['Classification'] == 'SNIa')).values
        self.M12_host_s = self.M12_objID[spec_cond]
        self.M12_hname_s = self.df_M12['IAUName'].values[spec_cond] 

        z_cond = ((self.df_M12['is_host'] == 'True')
                     & (self.df_M12['Classification'] == 'zSNIa')).values
        self.M12_host_z = self.M12_objID[z_cond]
        self.M12_hname_z = self.df_M12['IAUName'].values[z_cond] 
                
        #Read data used in Sako+ 2018.
        fpath = './../INPUT_FILES/S18/formatted_hosts.csv'
        self.df_S18 = pd.read_csv(fpath, header=0, low_memory=False, dtype='str')
        self.df_S18 = self.df_S18.replace('null', np.nan)
        self.S18_host = self.df_S18['objID'].values
        self.S18_hname = self.df_S18['IAUName'].values
 
        spec_cond = (self.df_S18['Classification'] == 'SNIa').values
        self.S18_host_s = self.S18_host[spec_cond]
        self.S18_hname_s = self.S18_hname[spec_cond] 

        z_cond = (self.df_S18['Classification'] == 'zSNIa').values
        self.S18_host_z = self.S18_host[z_cond]
        self.S18_hname_z = self.S18_hname[z_cond]         
                        
    def compare_host_IDS(self):
        print 'M12, H17 and S18 # of hosts:', len(self.M12_host), len(self.H17_host), len(self.S18_host)
        print 'M12, H17 and S18 # of spec hosts:', len(self.M12_host_s), len(self.H17_host), len(self.S18_host_s)
        print 'M12, H17 and S18 # of zSNIa hosts:', len(self.M12_host_z), 0, len(self.S18_host_z)
        print 'M12 and H17 # of ctrl:', len(self.M12_objID), len(self.H17_objID)
        print '\n'
        
        print 'Hosts in common between H17 and M12:', len(set(self.M12_host).intersection(self.H17_host))
        print 'Hosts in common between H17 and S18:', len(set(self.S18_host).intersection(self.H17_host))
        print 'Hosts in common between M12 and S18:', len(set(self.S18_host).intersection(self.M12_host))
        print '\n'

        print 'Spec hosts in common between H17 and M12:', len(set(self.M12_host_s).intersection(self.H17_host))
        print 'Spec hosts in common between H17 and S18:', len(set(self.S18_host_s).intersection(self.H17_host))
        print 'Spec hosts in common between M12 and S18:', len(set(self.S18_host_s).intersection(self.M12_host_s))
        print '\n'

        print 'SNe in common between H17 and M12:', len(set(self.M12_hname).intersection(self.H17_hname))
        print 'SNe in common between H17 and S18:', len(set(self.S18_hname).intersection(self.H17_hname))
        print 'SNe in common between M12 and S18:', len(set(self.S18_hname).intersection(self.M12_hname))
        print '\n'

        print 'Spec SNe in common between H17 and M12:', len(set(self.M12_hname_s).intersection(self.H17_hname))
        print 'Spec SNe in common between H17 and S18:', len(set(self.S18_hname_s).intersection(self.H17_hname))
        print 'Spec SNe in common between M12 and S18:', len(set(self.S18_hname_s).intersection(self.M12_hname_s))
        print '\n'

        print 'Ctrl in common:', len(set(self.M12_objID).intersection(self.H17_objID))
        print 'H17 hosts in M12s whole sample:', len(set(self.H17_host).intersection(self.M12_objID))
        print 'M12 hosts in H17s whole sample:', len(set(self.M12_host).intersection(self.H17_objID))        


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
        
        #unique_hostsM12 = list(set(self.M12_hname) - set(self.H17_hname))
        unique_hostsM12 = list(set(self.M12_host) - set(self.H17_host))
        
        
        
        #print unique_hostsM12, len(unique_hostsM12)
        #print list(self.M12_hname), list(self.H17_hname)
        #unique_hostsH17 = list(set(SNID_H17) - set(SNID_M12))
        #common_SNe = list(set(SNID_H17).intersection(SNID_M12))
        #print '\n'
        #print 'SNe (spec) in M12', len(SNID_M12)
        #print 'SNe In M12 but not in H17:', len(unique_hostsM12)
        #print 'SNe In H17 but not in M12:', len(unique_hostsH17)
        #print 'common SNe', len(common_SNe)
        
        #print 'unique in M12', unique_hostsM12, '\n'
        #print 'unique in H17', unique_hostsH17
            
    def run_check(self):
        self.read_data()
        self.compare_host_IDS()
        #self.check_positions()
        #self.check_host_IDS()

if __name__ == '__main__':
    Check_Hosts()
 
