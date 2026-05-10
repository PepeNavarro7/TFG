#ifndef SIMPLE_AVD_DIFF_CPP
#define SIMPLE_AVD_DIFF_CPP

#include <omp.h>
#include "simpleadvdiff1d.h"
#include <cmath>

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 

simpleadvdiff1d::simpleadvdiff1d(const int &nx_points){ 
    nx = nx_points;
    neqn = nx;
    dtx=1.0/nx;     // Compute Spatial step
    dtx_doubled = 2.0 * dtx;
    dtx_squared = dtx*dtx;
    name = "1D_Simple Advection-Diffusion";
}

// Initialize stage vector Y0 with neqn components
void simpleadvdiff1d::init(double *Y0) {
    for (int i=0;i<neqn;i++) { 
        double x_i=(double)(i+1)*dtx;
        Y0[i]=sin(2.0*PI*x_i);
    }
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void simpleadvdiff1d::feval(const double &t, const double* Y, double* DY){
    // Compute partially DY in inner points
    #pragma omp for nowait
    for (int i = 1; i < nx - 1; i++){
        DY[i] = d * (Y[i + 1] - 2 * Y[i] + Y[i - 1]) / dtx_squared
              - a * (Y[i + 1]            - Y[i - 1]) / dtx_doubled;
    } // Nos saltamos la barrera
    #pragma omp single nowait
    {
        // Compute partially DY in boundary points (i=0 and i=nx-1)
        DY[0] = d * (Y[1] - 2 * Y[0] + Y[nx-1]) / dtx_squared
              - a * (Y[1]            - Y[nx-1]) / dtx_doubled;
    } // Nos saltamos esta barrera tambien
    #pragma omp single
    {
        DY[nx - 1] = d * (Y[0] - 2 * Y[nx-1] + Y[nx-2]) / dtx_squared
                   - a * (Y[0]               - Y[nx-2]) / dtx_doubled;
    } // Barrera implicita    
}
#endif