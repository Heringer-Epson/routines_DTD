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

/* Parameters
   ----------
   arg[0] = t' (integration variable)
   arg[1] = t
   arg[2] = tau
   arg[3] = s1
   arg[4] = s2
   arg[5] = t_ons
   arg[6] = t_bre
   arg[7] = sfr_norm
   arg[8] = B
*/

double DTD_func(double t, double B, double s1, double s2,
                double t_ons, double t_bre){
    //DTD propto (t/1Gyr)^s
    if (t < t_ons)
        return 1.E-40;
    else if ((t >= t_ons) & (t < t_bre)) 
        return pow(t, s1);
    else if (t >= t_bre) 
        return B * pow(t, s2);
}

double SFR_exponential_func(double t, double tau, double norm){
    return norm * exp(-t / tau);    
}

double conv_exponential_sSNR(int n, double args[n]){
    return DTD_func(
      args[0], args[8], args[3], args[4], args[5], args[6])
      * SFR_exponential_func(args[1] - args[0], args[2], args[7]);
}

double SFR_delayed_exponential_func(double t, double tau, double norm){
    return norm * t * exp(-t / tau);    
}

double conv_delayed_exponential_sSNR(int n, double args[n]){
    return DTD_func(
      args[0], args[8], args[3], args[4], args[5], args[6])
      * SFR_delayed_exponential_func(args[1] - args[0], args[2], args[7]);
}
