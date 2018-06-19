#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.32)
Q_factor = 1.6
z_ref = 0.

#http://www.sdss.org/dr12/algorithms/magnitudes/
b_u, b_g, b_r, b_i, b_z = 1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10
quadErr_u, quadErr_g, quadErr_r, quadErr_i, quadErr_z = .05, .02, .02, .02, .03 

def mag2maggy(mag, magErr, b, quadErr):
    """
    maggies[b,*]=2.D*bvalues[b]*sinh(-alog(bvalues[b])-0.4D*alog(10.D)*lups[b,*])

    maggies_err[b,*]=2.D*bvalues[b]*cosh(-alog(bvalues[b])- $
                                       0.4D*alog(10.D)*lups[b,*])* $
    0.4*alog(10.)*lups_err[b,*]

    """
    try:
        _magErr = np.sqrt(magErr**2. + quadErr**2.)
        maggy = 2. * b * np.sinh(-np.log(b) - 0.4 * np.log(10.) * mag)
        
        maggyErr = (2. * b * np.cosh(-np.log(b) - 0.4 * np.log(10.) * mag)
                    * 0.4 * np.log(10.) * _magErr)
        invvar = maggyErr**-2. 
    except:
        maggy, invvar = np.nan, np.nan
    return maggy, invvar

def kcorr2mag(maggy, maggy_kcorr):
    return -2.5 * np.log10(maggy / maggy_kcorr)


# maggies = 2 b sinh(-lnb -0.4 ln10 luptitudes)
#maggies_err = 2 b cosh(-lnb - 0.4 ln10 luptitudes) (0.4 ln10) luptitudes_err
#Warning: k_sdssfix adds errors in quadrature sigma(ugriz)=[0.05, 0.02, 0.02, 0.02, 0.03] to account for calibration uncertainties. 


class Process_Data(object):
    """Compute absolute magnitudes and necessary corrections to the data used
    Maox+2012 (http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M). This data
    was provided in priv. comm.
    
    Dependencies: To compute Kcorrection, the package KCORRECT
    (Blanton & Roweis 2007) and a Python wrapper are required.
    """
    def __init__(self, _inputs):
        self._inputs = _inputs
        self.df = None
        self.perform_run()

    def read_data(self):
        fpath = self._inputs.data_dir + 'data_matched.csv'
        self.df = pd.read_csv(fpath, header=0)

    '''
    def perform_redshift_check(self):
        fail_cond = (self.df.redshift - self.df.z > 1.e-3)
        if len(self.df[fail_cond]) > 0:
            error = 'Error: The photometry matched to Maoz data may be wrong,'\
                    ' as the given and retrieved redshifts do not agree.'
            raise ValueError(error)
    '''    

    def extinction_correction(self):
        """Compute extinction corrections. This is simply done by using the
        values retrieved from SDSS - to check if these corrections are valid
        only for 'Modelmags' or also for 'petromags'. This correction needs
        to be done before computing Kcorrections. Error not propragated here.
        Subtraction sign is correct; e.g. see quantity 'dered_u' under DR7,
        Views, SpecPhoto, http://skyserver.sdss.org/CasJobs/SchemaBrowser.aspx
        """
        for fltr in ['u', 'g', 'r', 'i', 'z']:
            self.df['ext_' + fltr] = (
              self.df['petroMag_' + fltr] - self.df['extinction_' + fltr])

    def trim_data(self):
        """Perform data cuts similar to the original publication."""

        self.df = self.df[(self.df['ra'] > 360. - 57.) | (self.df['ra'] < 57.0)]
        self.df = self.df[(self.df['z'] >= self._inputs.redshift_min) &
                          (self.df['z'] <= self._inputs.redshift_max)]
        self.df = self.df[(self.df['ext_u'] >= self._inputs.ext_u_min) &
                          (self.df['ext_u'] < self._inputs.ext_u_max)]   
        self.df = self.df[(self.df['ext_g'] >= self._inputs.ext_g_min) &
                          (self.df['ext_g'] < self._inputs.ext_g_max)]
        self.df = self.df[(self.df['ext_r'] >= self._inputs.ext_r_min) &
                          (self.df['ext_r'] < self._inputs.ext_r_max)]                                  
        self.df = self.df[(self.df['petroMagErr_u'] <= self._inputs.ext_uERR_max)]
        self.df = self.df[(self.df['petroMagErr_g'] <= self._inputs.ext_gERR_max)]
        self.df = self.df[(self.df['petroMagErr_r'] <= self._inputs.ext_rERR_max)]

    def make_Kcorrections(self):
        """Make the appropriate kcorrections using the package KCORRECT by
        Blanton & Roweis 2007. Code is available at http://kcorrect.org/ (v4_3)
        Note that a Python wrap around is also used; developed by nirinA and
        mantained at https://pypi.org/project/kcorrect_python. (v2017.07.05)         
        """
        kcorrect.load_templates()
        kcorrect.load_filters(f='sdss_filters.dat', band_shift=z_ref)
        
        def compute_kcorrection(redshift, u, g, r, i, z,
                                uErr, gErr, rErr, iErr, zErr):           
            
            _u, _uI = mag2maggy(u, uErr, b_u, quadErr_u)
            _g, _gI = mag2maggy(g, gErr, b_g, quadErr_g)
            _r, _rI = mag2maggy(r, rErr, b_r, quadErr_r)
            _i, _iI = mag2maggy(i, iErr, b_i, quadErr_i)
            _z, _zI = mag2maggy(z, zErr, b_z, quadErr_z)
            
            inp_array = np.array(
              [redshift, _u, _g, _r, _i, _z, _uI, _gI, _rI, _iI, _zI])
            out_coeffs = kcorrect.fit_coeffs(inp_array)
            out_kcorrection = kcorrect.reconstruct_maggies(
              out_coeffs, redshift=z_ref)
            
            #Convert kcorrections for maggies to magnitudes.
            u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr = [
              kcorr2mag(inp_maggy, kcorr) for (inp_maggy, kcorr) in\
              zip(inp_array[1:6], out_kcorrection[1::])]
            
            return u_kcorr, g_kcorr, r_kcorr, i_kcorr, z_kcorr
        
        self.df['kcorr_u'], self.df['kcorr_g'], self.df['kcorr_r'],\
        self.df['kcorr_i'], self.df['kcorr_z'] =\
          np.vectorize(compute_kcorrection)(
          self.df['z'], self.df['ext_u'], self.df['ext_g'], 
          self.df['ext_r'], self.df['ext_i'], self.df['ext_z'],
          self.df['petroMagErr_u'], self.df['petroMagErr_g'],
          self.df['petroMagErr_r'], self.df['petroMagErr_i'],
          self.df['petroMagErr_z'])    

    def compute_abs_mags(self):
        """Compute absolute magnitudes. Note that the extinction correction
        is done here but shouldn't because it impacts the kcorr calculation.
        Note that the kcorr should be subtracted, as explicit in:
        http://cosmo.nyu.edu/blanton/kcorrect/kcorrect_help.html#SDSS_KCORRECT
        """
        def abs_mag(z, m, kcorr):
            #lum_dist is returned in Mpc, as needed for the calculation below.
            lum_dist = cosmo.luminosity_distance(z).value                        
            M = (m - 5. * np.log10(lum_dist) - 25. - kcorr
                 + (z - z_ref) * Q_factor)
            return M
                        
        for fltr in ['u', 'g', 'r', 'i', 'z']:
            self.df['abs_' + fltr] = np.vectorize(abs_mag)(
              self.df['z'], self.df['ext_' + fltr],
              self.df['kcorr_' + fltr])        

    def save_output(self):
        fpath = self._inputs.subdir_fullpath + 'data_absmag.csv'
        self.df.to_csv(fpath)

    def perform_run(self):
        import kcorrect
        self.read_data()
        self.extinction_correction()
        self.trim_data()
        self.make_Kcorrections()
        self.compute_abs_mags()
        self.save_output()

'''
class Plot_CMD(object):
    
    def __init__(self, _df):
        
        self._df = _df
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        
        self.make_plot()
        
    def set_fig_frame(self):
        
        f1 = 'r'
        f2 = 'g'

        x_label = r'$' +f1 + '$'
        y_label = r'$' + f2 + '-' +f1 + '$'
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xlim(10., 25.)
        self.ax.set_ylim(-1., 3.)
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        #self.ax.yaxis.set_minor_locator(MultipleLocator(100))
        #self.ax.yaxis.set_major_locator(MultipleLocator(500))  

    def retrieve_data(self):
        self.df = pd.read_csv(self._inputs.fpath, header=0)
        
        #Make rough cut to exclude spurious objects.
        self.df = self.df[(self.df.ext_r > 10.) & (self.df.ext_r < 25.0)]        
        f1 = self.df['ext_r'].values
        f2 = self.df['ext_g'].values
        self.mag = f1
        self.color = f2 - f1  
        self.hosts = self.df['n_SN'].values
    
    def plot_quantities(self):

        #For the legend.
        self.ax.plot(self.mag, self.color, ls='None', marker=',', color='k', 
                     label='Control')
        
        #N_ctrl = str(len(self.x_acc) + len(self.x_rej))
        ##N_hosts = str(len(self.x_hosts))
        #plt.title(
        #  r'$\mu = ' + str(format(self.mu, '.3f')) + ', \sigma = '
        #  + str(format(self.std, '.3f')) + ', \mathrm{N_{ctrl}} = ' + N_ctrl
        #  + ', \mathrm{N_{host}} = ' + N_hosts + '$', fontsize=20.)
        
        self.ax.legend(frameon=True, fontsize=20., numpoints=1, loc=2)

    def manage_output(self):
        if self._inputs.save_fig:
            fpath = self._inputs.subdir_fullpath + 'FIGURES/CMD.pdf'
            plt.savefig(fpath, format='pdf')
        if self._inputs.show_fig:
            plt.show()

    def make_plot(self):
        self.set_fig_frame()
        self.retrieve_data()
        self.plot_quantities()
        self.manage_output()             
        plt.close(self.fig)    
'''        
