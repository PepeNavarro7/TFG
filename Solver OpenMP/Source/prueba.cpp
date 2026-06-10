#ifndef PRUEBA_CPP
#define PRUEBA_CPP

#include <omp.h>
#include "prueba.h"
#include <cmath>

using namespace std;

// PROBLEMA 0

prueba::prueba(const int &nx_points){ 
    nx = nx_points;
    neqn = nx;
    dtx=1.0/nx;     // Compute Spatial step
    name = "Prueba manual";
}

// Initialize stage vector Y0 with neqn components
void prueba::init(double *Y0) const {
    for (int i=0;i<neqn;i++) { 
        Y0[i]=1.0;
    }
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void prueba::feval(const double &t, const double* Y, double* DY) const {

    #pragma omp single
    {
        DY[0] = 2 * Y[0] - 6.0;
    } // Barrera implicita    
}
double prueba::feval_i (const double &t, const double *Y, const int &i) const {
    double res=0;
    // Compute partially DY in inner points
    return res;
}
#endif