#ifndef BRUSSELATOR1D_CU
#define BRUSSELATOR1D_CU

#include "brusselator1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 

__constant__ Params_brusselator1d cte3;
__constant__ double cte_t;


// Definicion de los valores constantes para el kernel
void brusselator1d::updateConstants() const {
    Params_brusselator1d aux;

    aux.neqn = this->neqn;
    aux.nx = this->nx;
    aux.A = this->A;
    aux.B = this->B;
    aux.DD = this->DD;

    cudaMemcpyToSymbol(cte3, &aux, sizeof(Params_brusselator1d));
}

void brusselator1d::init(double *Y0) const { 
    for (int i=0;i<nx;i++){  
        double x_i=(double)(i+1)*dtx;
        Y0[idx(i,0)]=A+sin(2*PI*x_i);
        Y0[idx(i,1)]=B;
    }  
}

// Kernel para paralelizar con CUDA el feval del problema
__global__ void kernel_brusselator1d (const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int ult_x = cte3.nx-1, 
        id_x = thread/2, // Identificamos coordenada x
        id_z = thread%2;  // Coordenada z

    if(thread < cte3.neqn){
        const double C[2]={cte3.A, cte3.B};
        const double val_ant = id_x == 0   ? C[id_z] : Y[thread-2],
                     valor   = Y[thread],
                     val_pst = id_x==ult_x ? C[id_z] : Y[thread+2],
                     val_pareja = id_z==0 ? Y[thread+1] : Y[thread-1];

        DY[thread] = cte3.DD * (val_pst - 2.0 * valor + val_ant);

        const double ui = id_z==0 ? valor : val_pareja,
                     vi = id_z==1 ? valor : val_pareja;
        const double u2v=ui*ui*vi;

        if(id_z==0){ // pares == x0
            DY[thread] += cte3.A + u2v - ( cte3.B + 1 ) * ui;
        } else{ // impares == x1
            DY[thread] += cte3.B * ui - u2v; 
        }
    }
}

__global__ void graph_brusselator1d (const double offset, const double* __restrict__ Y, double* __restrict__ DY){
    double t = cte_t + offset;

    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int ult_x = cte3.nx-1, 
        id_x = thread/2, // Identificamos coordenada x
        id_z = thread%2;  // Coordenada z

    if(thread < cte3.neqn){
        const double C[2]={cte3.A, cte3.B};
        const double val_ant = id_x == 0   ? C[id_z] : Y[thread-2],
                     valor   = Y[thread],
                     val_pst = id_x==ult_x ? C[id_z] : Y[thread+2],
                     val_pareja = id_z==0 ? Y[thread+1] : Y[thread-1];

        DY[thread] = cte3.DD * (val_pst - 2.0 * valor + val_ant);

        const double ui = id_z==0 ? valor : val_pareja,
                     vi = id_z==1 ? valor : val_pareja;
        const double u2v=ui*ui*vi;

        if(id_z==0){ // pares == x0
            DY[thread] += cte3.A + u2v - ( cte3.B + 1 ) * ui;
        } else{ // impares == x1
            DY[thread] += cte3.B * ui - u2v; 
        }
    }
}

// Feval sin graph
void brusselator1d::feval (const double &t, const double *Y, double *DY) const {
    kernel_brusselator1d<<<this->grid,this->block>>>(t,Y,DY);
}

//Feval con graph
void brusselator1d::feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const {
    graph_brusselator1d<<<this->grid, this->block, 0, stream>>>(offset, Y, DY);
}
  
#endif   