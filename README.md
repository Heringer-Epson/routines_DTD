# routines_DTD
Routines used in the project for delay time distributions.

ENVIRONMENT SPECS:
  For rhis project a conda environment is used and is created
  according to the instructions in the env_specs.txt file.
  OS is Ubuntu 17.10. and the gfortran compiller version 6.4.0
  has been installed in order to run PythonFSPS (which
  conflicts with the default fortran installation (7.2.0)).

  -gfortran-6
    -sudo apt-get install gfortran-6
    -Then change the $SPS_HOME/src/Makefile so that
     F90 = gfortran -> F90 = gfortran-6

IMPORTANT:
  The codes here can be used as standalone routines, but they
  may depend on data produced by FSPS (v3.0). These data files
  are stored under INPUT_FILES/STELLAR_POP/ which have been
  uploaded to this repository. However, if one wishes to
  re-create these files by running 'run_fsps.py', then
  the FSPS and PythonFSPS packages need to be installed. See
  the 'INSTALLATION NOTES' below. 

INSTALLATION NOTES:

  Some of the codes here depend on external packages being installed
  which are NOT present in this git rep. e.g.: FSPS and PythonFSPS.
  Typically, I fork the repositories mentioned below. 

  -FSPS.
    -Source code available at [https://github.com/cconroy20/fsps].
      -Installation requires setting the SPS_HOME variable at the
       bash file. e.g.: export SPS_HOME='/home/heringer/Programs/fsps'
       and typing 'make' at the src directory.
      -IMPORATNT NOTE: PythonFSPS could not be installed properly for
       me unless the following line is changed in src/Makefile
       F90FLAGS = -O -cpp -> F90FLAGS = -O -cpp -fPIC

  -PythonFSPS.
    -Source code available at [https://github.com/dfm/python-fsps].
    -In the main directory, type python setup.py install

  Before compiling FSPS (by typing 'make'):
  -The quantities computed in the original paper
   [http://adsabs.harvard.edu/abs/2017ApJ...834...15H]
   adopted the BaSeL spectral library, which needs to be changed
   at $SPS_HOME/src/sps_vars.f90 before make.
   #define MILES 1 #define BASEL 0 -> #define MILES 0 #define BASEL 1

RUNNING NOTES:
  -Running SN_rate_gen_tests.py with the profile decorator requires the
   code to be executed as kernprof -l -v SN_rate_gen_tests.py

