//From http://nealhughes.net/cython1/
#include <math.h>

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

double DTD_func(double t, double B, double s1, double s2,
                double t_ons, double t_bre){
    if (t < t_ons)
        return 1.E-40;
    else if ((t >= t_ons) & (t < t_bre)) 
        return pow(t / t_ons, s1);
    else if (t >= t_bre) 
        return B * pow(t / t_ons, s2);
}
