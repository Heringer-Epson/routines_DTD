# routines_DTD
Routines used to make models of supernova rates as a function of
color with respect to a fitted red sequence. The models may then
be used to compute the likelihood of parametrized delay time
distributions (DTD) and to make figures containing the relevant
physical quantities.

Documentation is available at ./doc/manual.pdf

QUICK START:
>>python setup.py build_ext --inplace

Input parameters may be set in the 'input_params.py' file and
several standard routines may be executed via

>>python master.py (at $INSTALLATION_DIR/codes/)

The default settings in master.py and input_params should
allow for a simple execution case, which will reproduce
Fig. 9 of H17 and generate several other figures
containing the relationship between age, color, star
formation history and SN rates. Outputs will be stored under
'INSTALLATION_DIR/OUTPUT_FILES/RUNS/example1/'.

H17: http://adsabs.harvard.edu/abs/2017ApJ...834...15H 
