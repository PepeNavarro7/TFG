#ifndef SIMPLE_AVD_DIFF_CPP
#define SIMPLE_AVD_DIFF_CPP

#include "simpleadvdiff1d.h"
#include <cmath>

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 

struct Params_simpleadvdiff1d {
    int n;
    double dtx_2_inv;
    double dtx_sq_inv;
    double a;
    double d;
};

// Variable en memoria constante (vive en la GPU)
__constant__ Params_simpleadvdiff1d cte1;


// Initialize stage vector Y0 with neqn components
void simpleadvdiff1d::init(double *Y0) const {
    for (int i=0;i<neqn;i++) { 
        double x_i=(double)(i+1)*dtx;
        Y0[i]=sin(2.0*PI*x_i);
    }
}

void simpleadvdiff1d::updateConstants() const {
    Params_simpleadvdiff1d aux;

    aux.n = neqn;
    aux.dtx_2_inv  = 1.0 / dtx_doubled;
    aux.dtx_sq_inv = 1.0 / dtx_squared;
    aux.a = a;
    aux.d = d;

    cudaMemcpyToSymbol(cte1, &aux, sizeof(Params_simpleadvdiff1d));
}


__global__ void feval_simpleadvdiff1d(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<cte1.n){
        const int ult = cte1.n-1;
        const int i_ant = i ==  0  ? ult : i-1, // Calculamos indices vecinos
                  i_pst = i == ult ?  0  : i+1;

        const double v_ant = Y[i_ant], // Acceso a los indices
                     valor = Y[i],
                     v_pst = Y[i_pst];
        
        DY[i] = cte1.d * (v_pst - 2.0*valor + v_ant) * cte1.dtx_sq_inv // Calculo del valor feval
              - cte1.a * (v_pst             - v_ant) * cte1.dtx_2_inv;    
    }
        
    
    
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void simpleadvdiff1d::feval(const double &t, const double* Y, double* DY) const {
    feval_simpleadvdiff1d<<<NUM_BLOCKS, THREADSPERBLOCK>>>(t, Y, DY);
}

#endif