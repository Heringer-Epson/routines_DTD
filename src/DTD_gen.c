#include <math.h>
#include <stdio.h>

/*
Description:
------------
The routines here are used compute SN rates, given a DTD form and an
exponential SFH. The DTD and SFH are convolved here and integrated in 
SN_rate.py to determine rates. This is roughly a factor of 10 faster using
ctypes compared to DTD_gen_outdated.py.
*/

double DTD_func(double t, double A, double B, double s1, double s2,
                double t_ons, double t_cut){
    if (t < t_ons)
        return 1.E-40;
    else if ((t >= t_ons) & (t < t_cut)) 
        return A * pow(t, s1);
    else if (t >= t_cut) 
        return B * pow(t, s2);
}

double SFR_func(double t, double tau, double norm){
    return norm * exp(-t / tau);    
}

double conv_sSNR(int n, double args[n]){
    return DTD_func(
      args[0], args[3], args[9], args[4], args[5],args[6], args[7])
      * SFR_func(args[1] - args[0], args[2], args[8]);
}
