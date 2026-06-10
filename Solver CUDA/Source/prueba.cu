#ifndef PRUEBA_CU
#define PRUEBA_CU

#include "prueba.h"
#include <cmath>

using namespace std;



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

__global__ void feval_prueba(const double t, const double* Y, double* DY, const int nx){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<nx){
        DY[i] = 2.0 * Y[i] - 6.0;
    }
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void prueba::feval(const double &t, const double* Y, double* DY) const {
    feval_prueba<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t,Y,DY,nx);
}
#endif