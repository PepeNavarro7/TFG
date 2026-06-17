#ifndef ADVDIFF1D_CPP
#define ADVDIFF1D_CPP

#include "advdiff1d.h"
#include <cmath>

using namespace std;

// Constructor of the class IVP_ODE_advdiff1d    
advdiff1d::advdiff1d(const int &nx_points){ 
    name = "1D_Advection-Diffusion";
    nx=nx_points;
    neqn = nx;
    dtx=1.0/nx;     // Compute Spatial step
    dtx_squared=dtx*dtx;
    dtx_quad=4.0*dtx;
}

// Initialize stage vector Y0 with neqn components
void advdiff1d::init(double *Y0) const {
    for (int i=0; i<neqn; ++i) { 
        double x_i = (double)(i+1)*dtx;
        Y0[i] = sin(2.0*PI*x_i);
    }
}

__global__ void feval_advdiff1d(const double t, const double *Y, double *DY, const int nx, const double dtx, const double a, const double d){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const double PI = 3.14159265358979,
        dtx_squared=dtx*dtx,
        dtx_quad=4.0*dtx;
    
    if(i>=1 && i<=(nx-2)){   // Compute partially DY in inner points
        DY[i] = d * (Y[i+1]    - 2*Y[i]   + Y[i-1]) / dtx_squared
              - a * (Y[i+1]*Y[i+1] - Y[i-1]*Y[i-1]) / dtx_quad;
    } else if(i==0){ // Primero
        DY[0]   = d * (Y[1]    - 2*Y[0]  + Y[nx-1]) / dtx_squared
                - a * (Y[1]*Y[1] - Y[nx-1]*Y[nx-1]) / dtx_quad;
    } else if(i==(nx-1)){ // Ultimo
        DY[nx-1]= d * (Y[0] - 2*Y[nx-1]  + Y[nx-2]) / dtx_squared
                - a * (Y[0]*Y[0] - Y[nx-2]*Y[nx-2]) / dtx_quad;
    } 
    
    // Complete the computation of DY
    if(i>=0 && i<=(nx-1)){
        const double x = (i+1)*dtx;
        const double pi2xpt=2.0*PI*x + t;
        const double c=cos(pi2xpt), s=sin(pi2xpt); 
        const double res = c + 2.0*a*PI* s*c + 4.0*d*PI*PI*s - s;

        DY[i] += Y[i] + res;
    }
}

void advdiff1d::feval (const double &t, const double *Y, double *DY) const {
    feval_advdiff1d<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t, Y, DY, nx, dtx, a, d);
}

/*double advdiff1d::f(const double &x, const double &t) const{ 
    const double pi2xpt=2.0*PI*x + t;
    const double c=cos(pi2xpt);
    const double s=sin(pi2xpt); 
    return( c + 2*a*PI* s*c + 4*d*PI*PI*s - s);
}*/

#endif