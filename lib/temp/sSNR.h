#include <math.h>
#include <stdio.h>

static union
{
    double d;
    struct
    {

#ifdef LITTLE_ENDIAN
    int j, i;
#else
    int i, j;
#endif
    } n;
} _eco;

#define EXP_A (1048576/M_LN2) /* use 1512775 for integer version */
#define EXP_C 60801            /* see text for choice of c values */
#define EXP(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)

/*
Description:
------------
The routines here are used compute SN rates, given a DTD form and an
exponential SFH. The DTD and SFH are convolved here and integrated in 
SN_rate.py to determine rates. This is roughly a factor of 10 faster using
ctypes compared to DTD_gen_outdated.py.
*/

double DTD_func(double t, double B, double s1, double s2,
                double t_ons, double t_bre){
    if (t < t_ons)
        return 1.E-40;
    else if ((t >= t_ons) & (t < t_bre)) 
        return pow(t / t_ons, s1);
    else if (t >= t_bre) 
        return B * pow(t / t_ons, s2);
}

double sfrexp(double t, double tau, double norm){
    //return norm * exp(-t / tau);    
    return norm * EXP(-t / tau);    
}

double sfrdelexp(double t, double tau, double norm){
    //return norm * t * exp(-t / tau);    
    return norm * t * EXP(-t / tau);    
}

double conv_exponential_sSNR(
  double tprim, double t, double tau, double s1, double s2,
  double t_ons, double t_bre, double sfr_norm, double B){
    return DTD_func(tprim, B, s1, s2, t_ons, t_bre) * sfrexp(t - tprim, tau, sfr_norm);
}

double conv_delayed_exponential_sSNR(
  double tprim, double t, double tau, double s1, double s2,
  double t_ons, double t_bre, double sfr_norm, double B){
    return DTD_func(tprim, B, s1, s2, t_ons, t_bre) * sfrdelexp(t - tprim, tau, sfr_norm);
}
