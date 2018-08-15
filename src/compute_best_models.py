#!/usr/bin/env python

import numpy as np

class Compute_Best(object):
    """
    Description:
    ------------
    This code will compute the most likely DTD and its 68% and 95% confidence
    range under two assumptions: s1=-1 and s1=s2. It requires a fine DTD model
    grid (slope steps of 0.01). These fine grids are not currently being
    computed to save simulation time.

    Parameters:
    -----------
    _inputs : ~instance
        Instance of the Input_Parameters class defined in input_params.py.
     
    Outputs:
    --------
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihood_cont.csv
    ./../../OUTPUT_FILES/RUNS/$RUN_DIR/likelihood_def.csv
    """      
    def __init__(self, _inputs):
        self._inputs = _inputs
        self.contour_list = [0.95, 0.68]  
        self.write_scenarios()
    
    def write_scenarios(self):
        fpath = self._inputs.subdir_fullpath + 'DTD_scenarios.dat'
        out = open(fpath, 'w')
        out.write('---DTD analysis given an assumption---')

        conditions = ['cont', 'def']
        labels = ['s1=s2', 's1=-1']
        
        for (cond,label) in zip(conditions,labels):


            fpath = self._inputs.subdir_fullpath + 'likelihood_' + cond + '.csv'
            slopes_1, slopes_2, ln_L = np.loadtxt(
              fpath, delimiter=',', skiprows=7, usecols=(0,1,3), unpack=True)           
              
            #Get how many slopes there is s1 and s2 
            slopes = np.unique(slopes_2)
            N_s = len(slopes)

            #multiplicative factor to make exponentials small. Otherwise difficult
            #to handle e^-1000.
            _L = ln_L - min(ln_L) 
            #Then normalize in linear scale.
            _L = np.exp(_L)
            L = _L / sum(_L)

            aux_L = np.copy(L)
            aux_L /= sum(aux_L) #Renormalise.
            _L_sort = sorted(aux_L, reverse=True)
            _L_hist_cum = np.cumsum(_L_sort)
            
            most_likely_s2 = slopes_2[aux_L.argmax()]

            out.write('\n\nIf ' + label + ':\n')
            out.write('  Most likely late time slope: ' + str(most_likely_s2))

            for contour in self.contour_list:
                _L_hist_diff = [abs(value - contour) for value in _L_hist_cum]
                diff_at_contour, idx_at_contour = min((val,idx) for (idx,val)
                                                      in enumerate(_L_hist_diff))
                
                m = abs(aux_L - _L_sort[idx_at_contour])
                
                L_at_sigma = _L_sort[idx_at_contour]
                left_L = aux_L[0:aux_L.argmax()]
                left_slopes = slopes_2[0:aux_L.argmax()]
                left_idx = abs(left_L - L_at_sigma).argmin()
                left_slope_at_sigma = left_slopes[left_idx]
                left_unc = left_slope_at_sigma - most_likely_s2

                L_at_sigma = _L_sort[idx_at_contour]
                right_L = aux_L[aux_L.argmax():]
                right_slopes = slopes_2[aux_L.argmax():]
                right_idx = abs(right_L - L_at_sigma).argmin()
                right_slope_at_sigma = right_slopes[right_idx] 
                right_unc = right_slope_at_sigma - most_likely_s2               

                out.write(
                  '\n  At ' + str(contour) + '% cf: -' + str(left_unc)
                  + ' to ' + str(right_unc))
                
                if diff_at_contour > 0.1:
                    UserWarning(str(contour * 100.) + '% contour not constrained.')	            
        out.close()
