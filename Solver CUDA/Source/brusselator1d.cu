#ifndef BRUSSELATOR1D_CU
#define BRUSSELATOR1D_CU

#include "brusselator1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 

// Variables en memoria constante (viven en la GPU)
__constant__ Params_brusselator1d cte_br1d;
__constant__ double cte_br1d_t;


// Definicion de los valores constantes para el kernel
void brusselator1d::updateConstants() const {
    Params_brusselator1d aux;

    aux.neqn = this->neqn;
    aux.nx = this->nx;
    aux.A = this->A;
    aux.B = this->B;
    aux.DD = this->DD;

    cudaMemcpyToSymbol(cte_br1d, &aux, sizeof(Params_brusselator1d));
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
    const int ult_x = cte_br1d.nx-1, 
        id_x = thread/2, // Identificamos coordenada x
        id_z = thread%2;  // Coordenada z

    if(thread < cte_br1d.neqn){
        const double C[2]={cte_br1d.A, cte_br1d.B};
        const double val_izq = id_x == 0   ? C[id_z] : Y[thread-2],
                     valor   = Y[thread],
                     val_der = id_x==ult_x ? C[id_z] : Y[thread+2],
                     val_par = id_z==0 ? Y[thread+1] : Y[thread-1];

        DY[thread] = cte_br1d.DD * (val_der - 2.0 * valor + val_izq);

        const double ui = id_z==0 ? valor : val_par,
                     vi = id_z==1 ? valor : val_par;
        const double u2v=ui*ui*vi;

        DY[thread] += id_z==0 ? (cte_br1d.A + u2v - ( cte_br1d.B + 1 ) * ui) : (cte_br1d.B * ui - u2v);
    }
}

__global__ void graph_brusselator1d (const double offset, const double* __restrict__ Y, double* __restrict__ DY){
    //const double t = cte_br1d_t + offset;

    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int ult_x = cte_br1d.nx-1, 
        id_x = thread/2, // Identificamos coordenada x
        id_z = thread%2;  // Coordenada z

    if(thread < cte_br1d.neqn){
        const double C[2]={cte_br1d.A, cte_br1d.B};
        const double val_izq = id_x == 0   ? C[id_z] : Y[thread-2],
                     valor   = Y[thread],
                     val_der = id_x==ult_x ? C[id_z] : Y[thread+2],
                     val_par = id_z==0 ? Y[thread+1] : Y[thread-1];

        DY[thread] = cte_br1d.DD * (val_der - 2.0 * valor + val_izq);

        const double ui = id_z==0 ? valor : val_par,
                     vi = id_z==1 ? valor : val_par;
        const double u2v=ui*ui*vi;

        DY[thread] += id_z==0 ? (cte_br1d.A + u2v - ( cte_br1d.B + 1 ) * ui) : (cte_br1d.B * ui - u2v);
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

const void* brusselator1d::get_t() const { 
    return (const void*)&cte_br1d_t; 
}

void brusselator1d::archivo(const string &filename, const double *Y) const { 
    archivo2(filename,Y); 
}

int brusselator1d::idx(const int &i, const int &j) const { 
    return i * 2 + j; 
}
  
#endif   