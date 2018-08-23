#!/usr/bin/env python
from astropy import units as u
import numpy as np

"""
Set of functions which can be used to determine the effective efficiency of the
SDSS DR7 supernova survey.

References:
-----------
Maoz+ 2012 (M12): http://adsabs.harvard.edu/abs/2012MNRAS.426.3282M
"""

def visibility_time(redshift):
    """The total amount of time (in days and in the rest frame) during which
    the SDSS supernova survey scanned the sky. As in M12"""
    survey_duration = 269. * u.day
    survey_duration = survey_duration.to(u.year).value
    vistime = np.ones(len(redshift)) * survey_duration        
    vistime = np.divide(vistime,(1. + redshift)) #In the galaxy rest frame.
    return vistime

def detection_eff(redshift):
    """The approximate effiency for detecting SNe Ia as a function of redshidft.
    As in M12."""
    detection_eff = np.piecewise(
      redshift, [redshift < 0.175, redshift >= 0.175],
      [0.72, lambda x: -3.2 * x + 1.28])   
    return detection_eff 
