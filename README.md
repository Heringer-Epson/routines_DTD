# routines_DTD
Codes used in a project on delay time distributions.

INSTALLATION NOTES:
  Using a conda environment created using the env_specs.txt file.
  OS is Ubuntu 17.10. gfortran compiller version 6.4.0
  Note that PythonFSPS did not work with 7.2.0.

  -gfortran-6
    -sudo apt-get install gfortran-6
    -Then change the $SPS_HOME/src/Makefile so that
     F90 = gfortran -> F90 = gfortran-6

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

BEFORE COMPILLING (make) FSPS:
  -The quantities computed in the original paper
   [http://adsabs.harvard.edu/abs/2017ApJ...834...15H]
   adopted the BaSeL spectral library, which needs to be changed
   at $SPS_HOME/src/sps_vars.f90 before make.
   #define MILES 1 #define BASEL 0 -> #define MILES 0 #define BASEL 1



