#ifndef PRUEBA_CPP
#define PRUEBA_CPP

#include <omp.h>
#include "prueba.h"
#include <cmath>

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 

prueba::prueba(const int &nx_points){ 
    nx = nx_points;
    neqn = nx;
    dtx=1.0/nx;     // Compute Spatial step
    dtx_doubled = 2.0 * dtx;
    dtx_squared = dtx*dtx;
    name = "1D_Simple Advection-Diffusion";
}

// Initialize stage vector Y0 with neqn components
void prueba::init(double *Y0) {
    for (int i=0;i<neqn;i++) { 
        Y0[i]=1.0;
    }
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void prueba::feval(const double &t, const double* Y, double* DY){

    #pragma omp single
    {
        DY[0] = 2.0 * Y[0] * t;
    } // Barrera implicita    
}
#endif