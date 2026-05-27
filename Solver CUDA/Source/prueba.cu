#ifndef PRUEBA_CU
#define PRUEBA_CU

#include "prueba.h"
#include <cmath>

using namespace std;



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

__global__ void d_feval(const double &t, const double* Y, double* DY, const int &nx){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<nx){
        DY[i] = 2.0 * Y[i] * t;
    }
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void prueba::feval(const double &t, const double* Y, double* DY){
    d_feval<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t,Y,DY,nx);
}
#endif