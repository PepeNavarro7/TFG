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
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int ult = cte2.neqn-1;

    if(thread<cte2.neqn){
        const int i_ant = thread==0   ? ult : thread-1,
                  i_pst = thread==ult ?  0  : thread+1;
        const double val_ant = Y[i_ant],
                     valor   = Y[thread],
                     val_pst = Y[i_pst];
        
        DY[thread] = cte2.d * (val_pst - 2.0 * valor + val_ant) * cte2.dtx_sq_inv
                   - cte2.a * (val_pst * val_pst - val_ant * val_ant) * cte2.dtx_4_inv;

        const double x = (thread+1) * cte2.dtx;
        const double pi2xpt = 2.0 * cte2.PI * x + t;
        const double c=cos(pi2xpt), s=sin(pi2xpt); 
        const double res = c + 2.0 * cte2.a * cte2.PI * s * c 
                         + 4.0 * cte2.d * cte2.PI * cte2.PI * s - s;

        DY[thread] += Y[thread] + res;
    }
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