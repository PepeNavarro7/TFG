#ifndef ADVDIFF1D_CU
#define ADVDIFF1D_CU

#include "advdiff1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 2
// Class for the IVP-ODE representing a 1D Advection-Diffusion model
struct Params_advdiff1d {
    double PI;
    int neqn;
    double dtx;
    double dtx_sq_inv;
    double dtx_4_inv;
    double a;
    double d;
};

// Variable en memoria constante (vive en la GPU)
__constant__ Params_advdiff1d cte2;

// Definicion de los valores constantes para el kernel
void advdiff1d::updateConstants() const {
    Params_advdiff1d aux;

    aux.PI = 3.14159265358979;
    aux.neqn = neqn;
    aux.dtx = dtx;
    aux.dtx_sq_inv = 1.0 / dtx_sq; 
    aux.dtx_4_inv = 1.0 / dtx_4; 
    aux.a = a;
    aux.d = d;

    cudaMemcpyToSymbol(cte2, &aux, sizeof(Params_advdiff1d));
}

// Initialize stage vector Y0 with neqn components
void advdiff1d::init(double *Y0) const {
    for (int i=0; i<neqn; ++i) { 
        double x_i = (double)(i+1)*dtx;
        Y0[i] = sin(2.0*PI*x_i);
    }
}

__global__ void feval_advdiff1d(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    const int ult = cte2.neqn-1;
    const int i_ant = i==0 ? ult : i-1,
              i_pst = i==ult ? 0 : i+1;
    const double v_ant = Y[i_ant],
                 valor = Y[i],
                 v_pst = Y[i_pst];
    
    DY[i] = cte2.d * (v_pst - 2.0 * valor + v_ant) * cte2.dtx_sq_inv
          - cte2.a * (v_pst * v_pst - v_ant * v_ant) * cte2.dtx_4_inv;

    const double x = (i+1) * cte2.dtx;
    const double pi2xpt = 2.0 * cte2.PI * x + t;
    const double c=cos(pi2xpt), s=sin(pi2xpt); 
    const double res = c + 2.0 * cte2.a * cte2.PI * s * c 
                     + 4.0 * cte2.d * cte2.PI * cte2.PI * s - s;

    DY[i] += Y[i] + res;
    
    /*
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
        */
}

void advdiff1d::feval (const double &t, const double *Y, double *DY) const {
    feval_advdiff1d<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t, Y, DY);
}


/*double advdiff1d::f(const double &x, const double &t) const{ 
    const double pi2xpt=2.0*PI*x + t;
    const double c=cos(pi2xpt);
    const double s=sin(pi2xpt); 
    return( c + 2*a*PI* s*c + 4*d*PI*PI*s - s);
}*/

#endif