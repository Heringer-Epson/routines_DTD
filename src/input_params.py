#!/usr/bin/env python

import numpy as np
from astropy import units as u

class Input_Parameters(object):
    """
    Code Description
    ----------    
    The user may use this file to set the input parameters and the datasets
    to be used in applying the ssnral method, which is called under the
    'master.py' routine. All parameters will be set as the default ones,
    except for those which are changed under a given 'case'.

    Parameters
    ---------- 
    hosts_from : ~str
        'M12', 'H17' or 'S18'. Indicates from which data sample the hosts are
        to be colelcted from. Default is 'S18'.
    host_class : ~list
        List containing a combination of 'SNIa' (spectroscopically confirmed
        and the host galaxy has spectra taken), 'zSNIa' (photometrically typed
        and the host galaxy has spectra taken) and 'pSNIa' (photometrically
        typed but the host galaxy does not have spectra taken).
        Default is ['SNIa', 'zSNIa'].
    hosts_from_2004 : boolean
        Whether or not to include SN observed in 2004. H17 used those, but
        they are likely to be unreliable. Default is False.
    host_peculiar : boolean
        Whether or not to include galaxies which have hosted peculiar SN Ia.
        Only enforced if this information is available (i.e. information about
        host galaxies are retrieved from S18). Default is False

    kcorr_type : str
        Mode to perform Kcorrections. Options are 'complete' (where SDSS
        photometry is properly converted to maggies) or 'simple' (where
        corrections are prrformed as in H17). Default is 'complete'.
    z_ref : float
        Which redshift to use as a reference for Kcorrection (see Eq. 1 in
        Wyder+ 2007). Default is 0.0.
    Q_factor : float
        Evolutionary correction factor, as in Eq. 1 in Wyder+ 2007.
        Default is 1.6.

    data_Drange : str
        Whether or not to trim the range of Delta colors (e.g. D(g-r)).
        Options are 'limited' (if Dcolor is to be trimmed according to the
        red sequence fittting--as used in H17). Or 'full' (if no Dcolor cut
        is required--as in H19, after updating the ssnral package).
        If 'full', then the model_Drange variable must be 'extended'.
        Default is 'full'. 
    model_Drange : str
        Whether or not to use ssnral-Dcolor relationships that cover an
        extended range of Dcolors that is only valid in a Dcolor range.
        Options are 'reduced' (as in H17) or 'extended' (as in H19, after
        updating the ssnral package). Default is 'extended'.
    
    ra_min : ~float
        Minimum RA accepted. Default is 309. (360 - 51).
    ra_max : ~float
        Maximum RA accepted. Default is 57.
    dec_min : ~float
        Minimum DEC accepted. Default is -1.25.
    dec_max : ~float
        Maximum DEC accepted. Default is 1.25.
    redshift_min : ~float
        Minimum redshift accepted. Default is 0.01.
    redshift_max : ~float
        Maximum redshift accepted. Default is 0.2.
    petroMag_u_min : ~float
        Minimum SDSS u-band apparent petrosian magnitude accepted.
        Default is -1000. (i.e. unconstrained).        
    petroMag_u_max : ~float
        Maximum SDSS u-band apparent petrosian magnitude accepted.
        Default is 1000. (i.e. unconstrained). 
    petroMag_g_min : ~float
        Minimum SDSS g-band apparent petrosian magnitude accepted.
        Default is -1000. (i.e. unconstrained).        
    petroMag_g_max : ~float
        Maximum SDSS g-band apparent petrosian magnitude accepted.
        Default is 22.2.               
    ext_r_min : ~float
        Minimum SDSS r-band apparent petrosian magnitude accepted, after
        galactic extinction corrections have been performed. Default is 14.        
    ext_r_max : ~float
        Maximum SDSS r-band apparent petrosian magnitude accepted, after
        galactic extinction corrections have been performed. Default is 17.77.                 
    uERR_max : ~float
        Maximum SDSS u-band apparent petrosian magnitude uncertainty accepted.
        Default is 1000. (i.e. unconstrained). 
    gERR_max : ~float
        Maximum SDSS g-band apparent petrosian magnitude uncertainty accepted.
        Default is 0.05.
    rERR_max : ~float
        Maximum SDSS r-band apparent petrosian magnitude uncertainty accepted.
        Default is 0.05.
    Dcolor_min : ~float
        Sets the lower Dcolor limit, below which galaxies are not taken into
        account. This was used in H17. Only valid if the variable 'data_Drange'
        is 'limited' ##Check. Default is None. If a Dcolor cut is imposed, the
        upper limit will be taken as 2 times the standard deviation of the 
        red sequence fit.         

    Z : ~str
        Choice of metallicity for the fsps simulations. Default is '0.0190'.
        Different choices of metallicity will require FSPS to be available.
    imf_type : ~str
        Choice of initial mass function for the fsps simulations.
        Accepts: 'Salpeter', 'Chabrier' or 'Kroupa'. Default is 'Kroupa'.
    sfh_type : ~str
        Choice of star formation history for the fsps simulations.
        Accepts: 'exponential' (sfh propto exp(-t/tau)) or
        'delayed-exponential' (sfh propto t*exp(-t/tau).
        Default is 'exponential'.
    t_onset : ~astropy float (unit of time)
        Sets the time past which SN may occur. Determined by the lifetime of
        stars that may contribute to the SN rate. Default is .1*u.Gyr.
    t_cutoff : ~astropy float (unit of time)
        Sets the time at which the slope of the DTD may change. Determined by
        theoretical models. Default is 1.*u.Gyr.
    f2 : ~str
        f1 and f2 determine the color to be used as
        (f2 - f1). A list of available filters can be shown by
        calling fsps.list_filters() under run_fsps.py. Only g and r are
        currently supported. Default is 'g'.
    f1 : ~str
        Same as above. Default is 'r'.
    f0 : ~str
        The band used to for the luminosity unit. Default is 'r'.
    spec_lib : ~str
        Which spectral library to be used for FSPS simulations. Options are
        'BASEL' or 'MILES', depending on the availability of FSPS files.
        Default is 'BASEL'.
    isoc_lib_lib : ~str
        Which isochrone library to be used for FSPS simulations. Options are
        'PADOVA' or 'MIST', depending on the availability of FSPS files.
        Default is 'PADOVA'.        
    fbhb : float
        Fraction of blue horizontal branch stars. Options are 0.0 and 0.2, 
        depending on the availability of FSPS files. Default is 0.0.
  
    Dcolor_range : list
        Used to fit the red sequence only. Determines the Dcolor range of the
        red sequence that is to be fitted, so that the shape of the histogram
        of galaxies can be fitted by a gaussian. [Dcolor_min, Dcolor_max].
        Default is [-0.08, .1] (for f1='r' and f2='g').
    bin_size : float
        Size of the Dcolor bin used to build the histogram of galaxies, so
        that the red sequence can be fitted. Default is 0.005 (for f1='r'
        and f2='g').
    bin_range : list
        Used to bin the Dcolor data when fitting the red sequence. Format of 
        the list is [Dcolor_min, Dcolor_max]. Default is [-.8, .4] (for
        f1='r' and f2='g').

    visibility_flag : boolean
        Whether or not to correct rates acording to the redshift of the
        galaxies, as in M12. Default is True, was False in H17.
    tau_list : ~astropy array (unit of time)
        List containing the tau timescales (in yr) for the selected SFH. e.g.:
        tau_list = np.array([1., 1.5, 2., 3., 4., 5., 7., 10.]) * u.Gyr
    slopes : ~np.array
        Numpy array containing which DTD slopes to use to compute likelihoods.
        This package adopts the same array for slopes pre and post cutoff time.
        Default is np.arange(-3., 0.0001, 0.02).

    data_dir : ~str
        Relative path to the directory from which the control data should be
        retrieved. Does not need to be initialized, but must be defined in 
        each 'case'.
    subdir_fullpath : ~str
        Relative path to the  sub-directory where the outputs will be stored.
        For organization purposes only. Does not need to be initialized.

    custom_pars : ~tuple
        Tuple containing a set of variables to be set in a given case. The
        sequence of variables needs to be coded in each 'case'. For instance,
        under the case 'custom', this tuple contains
        (ctrl_samp, hosts_samp, host_class, redshift_max). This allows for a
        serie of simulations without needing to defined a series of cases.
        Does not need to be initialized.
    show_fig : boolean
        Whether or not to show the figures being made during the run.
        Default is False.
    save_fig : boolean
        Whether or not to save the figures being made during the run.
        Default is True.

    References:
    -----------
    Maoz+ 2012 (M12): http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
    Gao & Pritchet 2013 (G13): http://adsabs.harvard.edu/abs/2013AJ....145...83G
    Heringer+ 2017 (H17): http://adsabs.harvard.edu/abs/2017ApJ...834...15H
    Sako+ 2018 (S18): http://adsabs.harvard.edu/abs/2018PASP..130f4002S
    Wyder+ 2007: http://adsabs.harvard.edu/abs/2007ApJS..173..293W
    """
    def __init__(
      self, case, hosts_from='S18', host_class=['SNIa', 'zSNIa'],
      hosts_from_2004=False, host_peculiar=False,
      kcorr_type='complete', z_ref=0.0, Q_factor=1.6,
      data_Drange='full', model_Drange='extended',
      ra_min=309., ra_max=57., dec_min=-1.25, dec_max=1.25, redshift_min=0.01,
      redshift_max=0.2, petroMag_u_min=-1000., petroMag_u_max=1000.,
      petroMag_g_min=-1000., petroMag_g_max=1000., ext_r_min=-1000.,
      ext_r_max=1000., uERR_max=1000., gERR_max=0.1, rERR_max=0.1,
      Dcolor_min=None,  
      Z='0.0190', imf_type='Kroupa', sfh_type='exponential', t_onset=.1*u.Gyr,
      t_cutoff=1.*u.Gyr, f2='g', f1='r', f0='r', spec_lib='BASEL',
      isoc_lib='PADOVA', fbhb=0.0, dust='0',
      Dcolor_range=[-0.08, .1],
      bin_size=0.005, bin_range=[-.8, .4],
      visibility_flag=True,
      tau_list=np.array([1., 1.5, 2., 3., 4., 5., 7., 10.]) * u.Gyr,
      slopes=np.arange(-3., 0.0001, 0.01),
      data_dir=None, subdir_fullpath=None,
      custom_pars=None, show_fig=False, save_fig=True):

        self.case = case

        #Data set parameters.
        self.hosts_from = hosts_from
        self.host_class = host_class
        self.hosts_from_2004 = hosts_from_2004
        self.host_peculiar = host_peculiar

        #Kcorrection parameters.
        self.kcorr_type = kcorr_type
        self.z_ref = z_ref
        self.Q_factor = Q_factor

        #Data trimming parameters.
        self.data_Drange = data_Drange
        self.model_Drange = model_Drange
        self.ra_min, self.ra_max = ra_min, ra_max
        self.dec_min, self.dec_max = dec_min, dec_max
        self.redshift_min, self.redshift_max = redshift_min, redshift_max
        self.petroMag_u_min, self.petroMag_u_max = petroMag_u_min, petroMag_u_max
        self.petroMag_g_min, self.petroMag_g_max = petroMag_g_min, petroMag_g_max
        self.ext_r_min, self.ext_r_max = ext_r_min, ext_r_max
        self.uERR_max = uERR_max
        self.gERR_max = gERR_max
        self.rERR_max = rERR_max
        self.Dcolor_min = Dcolor_min
        
        
        #Stellar population parameters.
        self.Z = Z
        self.imf_type = imf_type
        self.sfh_type = sfh_type
        self.t_onset = t_onset
        self.t_cutoff = t_cutoff
        self.f2 = f2
        self.f1 = f1
        self.f0 = f0
        self.spec_lib = spec_lib
        self.isoc_lib = isoc_lib
        self.fbhb = fbhb
        self.dust = dust

        #Red sequence fitting parameters
        self.Dcolor_range = Dcolor_range
        self.bin_size = bin_size
        self.bin_range = bin_range
        
        #Model processing parameters.
        self.visibility_flag = visibility_flag 
        self.tau_list = tau_list   
        self.slopes = slopes
        self.slopes = np.arange(-3., 0.0001, 0.01) #Coarse for testing.        
        #self.slopes = np.arange(-3., 0.0001, 0.05) #Coarse for testing.        
        
        self.custom_pars = custom_pars
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.data_dir = None
        self.subdir_fullpath = None
        
        self.set_params()
                
    def set_params(self):
   
        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-= H17 tests =-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        
        if self.case == 'H17':   #Same data set as in Heringer+ 2017.
            subdir = 'H17/'   #and reproduces Fig. 9.  
            self.data_dir = './../INPUT_FILES/H17/'
            self.matching = 'File'
            self.hosts_from = 'H17'
            self.host_class = ['SNIa']
            self.imf_type = 'Chabrier'
            self.t_onset = 1.e8 * u.yr
            self.kcorr_type = 'simple'
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.petroMag_g_max=22.2
            self.ext_r_min=14.
            self.ext_r_max=17.77
            self.gERR_max=0.05
            self.rERR_max=0.05

            self.visibility_flag = False
            self.hosts_from_2004 = True
            self.host_peculiar = True
            self.data_Drange = 'limited'
            self.model_Drange = 'reduced'
            self.Dcolor_min = -0.4

        #=-=-=-=-=-=-=-=-=-=-=-=-Sample size tests=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        elif self.case == 'M12': #Same data set as in Maoz+ 2012.  
            subdir = 'M12/original/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'M12'
            self.host_class = ['SNIa', 'zSNIa']
            self.kcorr_type = 'none'
            self.t_onset = 40.e6 * u.yr
            self.redshift_min, self.redshift_max = 0., 0.4
            self.gERR_max = 1000.
            self.rERR_max = 1000.

        elif self.case == 'M12_zlim': #Same data set as in Maoz+ 2012.  
            subdir = 'M12/zlim/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'M12'
            self.host_class = ['SNIa', 'zSNIa']
            self.kcorr_type = 'none'
            self.t_onset = 40.e6 * u.yr
            self.gERR_max = 1000.
            self.rERR_max = 1000.

        elif self.case == 'M12_zunclim': #Same data set as in Maoz+ 2012.  
            subdir = 'M12/zunclim/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'M12'
            self.host_class = ['SNIa', 'zSNIa']
            self.kcorr_type = 'none'
            self.t_onset = 40.e6 * u.yr

        elif self.case == 'M12_zunclim_S18': #Same data set as in Maoz+ 2012.  
            subdir = 'M12/zunclim_S18/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.kcorr_type = 'none'
            self.t_onset = 40.e6 * u.yr

        elif self.case == 'M12_zunclim_S18': #Same data set as in Maoz+ 2012.  
            subdir = 'M12/zunclim_S18/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.kcorr_type = 'none'
            self.t_onset = 40.e6 * u.yr

        elif self.case == 'M12_zunclim_S18pec': #Same data set as in Maoz+ 2012.  
            subdir = 'M12/zunclim_S18pec/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.kcorr_type = 'none'
            self.t_onset = 40.e6 * u.yr
            self.host_peculiar = True

        elif self.case == 'M12_zunclim_S18SNIa': #Same data set as in Maoz+ 2012.  
            subdir = 'M12/zunclim_S18SNIa/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa']
            self.kcorr_type = 'none'
            self.t_onset = 40.e6 * u.yr

        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        elif self.case == 'default':  
            subdir = 'default/standard/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']

        #For SN counting only.
        elif self.case == 'default_pechost':  
            subdir = 'default/standard_pechost/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.host_peculiar = True

        #For SN counting only.
        elif self.case == 'default_nozSNIa':  
            subdir = 'default/standard_nozSNIa/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa']

        elif self.case == 'default_40':  
            subdir = 'default/standard_40/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.t_onset = 40.e6 * u.yr

        elif self.case == 'default_40_24':  
            subdir = 'default/standard_40_24/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.t_onset = 40.e6 * u.yr
            self.t_cutoff = 2.4 * u.Gyr
            
        elif self.case == 'default_40_042':  
            subdir = 'default/standard_40_042/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.t_onset = 40.e6 * u.yr
            self.t_cutoff = 0.42 * u.Gyr

        elif self.case == 'default_test':  
            subdir = 'default/standard_test/'  
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']
            self.slopes = np.arange(-3., 0.0001, 0.05) #Coarse for testing.
            self.f1 = 'i'        
            self.f0 = 'r'     

        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        elif self.case == 'H17_updated_model':
            #Same data set as in Heringer+ 2017, but updated methods.
            #Updates: KCORRECT, extended model Dcd, effective visibility time,
            #IMF is Kroupa.
            subdir = 'H17_updated_model/'   #and reproduces Fig. 9.  
            self.data_dir = './../INPUT_FILES/H17/'
            self.matching = 'File'
            self.hosts_from = 'H17'
            self.host_class = ['SNIa']
            self.t_onset = 1.e8 * u.yr
            self.kcorr_type = 'complete'
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.visibility_flag = True
            self.hosts_from_2004 = True
            self.host_peculiar = True
            self.data_Drange = 'limited'
            self.model_Drange = 'extended'

        elif self.case == 'H17_interpolation':
            #Same as h17, but model interpolation is extended
            subdir = 'H17_interp/'  
            self.data_dir = './../INPUT_FILES/H17/'
            self.matching = 'File'
            self.hosts_from = 'H17'
            self.host_class = ['SNIa']
            self.imf_type = 'Chabrier'
            self.t_onset = 1.e8 * u.yr
            self.kcorr_type = 'simple'
            self.ra_min, self.ra_max = 360. - 60., 60.
            self.visibility_flag = False
            self.hosts_from_2004 = True
            self.data_Drange = 'limited'
            self.model_Drange = 'extended'
            
        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= CUSTOM =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #Uses all default parameters, except those specifiec here.
        elif self.case == 'datasets':
            ctrl_samp, hosts_samp, self.host_class, z = self.custom_pars

            self.data_dir = './../INPUT_FILES/' + ctrl_samp + '/'
            self.redshift_max = float(z)

            if hosts_samp == 'native':
                self.hosts_from = ctrl_samp
            else:
                self.hosts_from = hosts_samp
            
            if ctrl_samp == 'M12':
                self.matching = 'Table'
            elif ctrl_samp == 'H17':
                self.matching = 'View'

            subdir = (
              'datasets/' + ctrl_samp + '_' + self.hosts_from + '_'\
               + self.host_class[-1] + '_' + z + '/') 

        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-= SYSTEMATICS =-=-=-=-=-=-=-=-=-=-=-=-=-=-
        elif self.case == 'sys':
            Q, t_o, t_c, sfh, imf, Z, fbhb, dust, spec_lib, isoc_lib, f2, f1, f0 =\
              self.custom_pars

            self.Q = float(Q)
            self.spec_lib = spec_lib
            self.isoc_lib = isoc_lib
            self.imf_type = imf
            self.sfh_type = sfh
            self.Z = Z
            self.fbhb = fbhb
            self.dust = dust
            self.t_cutoff = float(t_c) * u.Gyr
            self.t_onset = float(t_o) * u.Myr
            self.f2 = f2        
            self.f1 = f1        
            self.f0 = f0   

            #Default parameters for the analysis of systematic uncertainties. 
            self.matching = 'Table'
            self.data_dir = './../INPUT_FILES/M12/'
            self.hosts_from = 'S18'
            self.host_class = ['SNIa', 'zSNIa']

            subdir = (
              'sys/' + imf + '_' + sfh + '_' + Z + '_' + str(fbhb) + '_' + dust + '_'
              + isoc_lib + '_' + spec_lib + '_' + t_o + '_' + t_c + '_' + Q
              + '_' + f2 + f1 + f0 + '/')        

        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-= HOST CUSTOM =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #Similar to above, but this will not apply the default photometry cuts.
        #This is relevant for understanding the differences in host samples.
        elif self.case == 'hosts':
            ctrl_samp, self.hosts_from, self.matching, PC = self.custom_pars

            self.data_dir = './../INPUT_FILES/' + ctrl_samp + '/'
            self.host_peculiar = True

            #Whether to make photometry cuts (as in H17) or not:
            if not PC:
                self.petroMag_g_min, self.petroMag_g_max = -1000., 1000.
                self.ext_r_min, self.ext_r_max = -1000., 1000.
                self.gERR_max = 1000.
                self.rERR_max = 1000.
                PC_str = 'noPC'
            else:
                PC_str = 'PC'
                
            subdir = ('hosts/' + ctrl_samp + '_' + self.hosts_from + '_'
                      + self.matching + '_' + PC_str + '/')

        else:
            raise ValueError('Case "%s" is not defined.\n\n' %(self.case))            
            
        #Define the path to the RUN directory after subdir is set above.
        self.subdir_fullpath = './../OUTPUT_FILES/RUNS/' + subdir

        #Define from which directory to retrieve fsps files.
        self.fsps_path = (
          self.imf_type + '_' + self.sfh_type + '_' + self.Z + '_'
          + str(self.fbhb) + '_' + self.dust + '_' + self.spec_lib + '_'
          + self.isoc_lib + '/')
                
        #To avoid divergent values when s+1=0, replace s=-1 with s=-0.9999.
        self.slopes[abs(self.slopes + 1.) < 1.e-6] = -0.9999
    
