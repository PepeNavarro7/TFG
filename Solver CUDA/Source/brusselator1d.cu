#ifndef BRUSSELATOR1D_CU
#define BRUSSELATOR1D_CU

#include "brusselator1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 
struct Params_brusselator1d {
    int neqn;
    int nx;
    double A;
    double B;
    double DD;
};

// Variable en memoria constante (vive en la GPU)
__constant__ Params_brusselator1d cte3;

// Definicion de los valores constantes para el kernel
void brusselator1d::updateConstants() const {
    Params_brusselator1d aux;

    aux.neqn = neqn;
    aux.nx = nx;
    aux.A = A;
    aux.B = B;
    aux.DD = DD;

    cudaMemcpyToSymbol(cte3, &aux, sizeof(Params_brusselator1d));
}

void brusselator1d::init(double *Y0) const { 
    for (int i=0;i<nx;i++){  
        double x_i=(double)(i+1)*dtx;
        Y0[idx(i,0)]=A+sin(2*PI*x_i);
        Y0[idx(i,1)]=B;
    }  
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
__global__ void kernel_brusselator1d (const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<cte3.neqn){
        const double C[2]={cte3.A, cte3.B};
        const int ult_x = cte3.nx-1,
            id_x = i/2,
            id_z = i%2;

        const double v_ant = id_x == 0 ? C[id_z] : Y[i-2],
                     valor = Y[i],
                     v_pst = id_x==ult_x ? C[id_z] : Y[i+2];

        DY[i] = cte3.DD * (v_pst - 2.0 * valor + v_ant);

        if(id_z==0){ // pares == x0
            const double ui = valor, vi=Y[i+1];
            const double u2v=ui*ui*vi;
            DY[i] += cte3.A + u2v - ( cte3.B + 1 ) * ui;
        } else{ // impares == x1
            const double vi = valor, ui=Y[i-1];
            const double u2v=ui*ui*vi;
            DY[i] += cte3.B * ui - u2v; 
        }
    }
}

void brusselator1d::feval (const double &t, const double *Y, double *DY) const {
    kernel_brusselator1d<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t,Y,DY);
}
  
#endif   