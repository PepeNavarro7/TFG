#ifndef BRUSSELATOR1D_SHUFFLE_CU
#define BRUSSELATOR1D_SHUFFLE_CU

#include "brusselator1d_shuffle.h"
#include <cmath>
#include <iostream>

using namespace std;

// Variable en memoria constante de brusselator1d
__constant__ Params_brusselator1d_shuffle cte3_sh;

// Definicion de los valores constantes para el kernel
void brusselator1d_shuffle::updateConstants() const {
    Params_brusselator1d_shuffle aux;

    aux.neqn = this->neqn;
    aux.nx = this->nx;
    aux.A = this->A;
    aux.B = this->B;
    aux.DD = this->DD;

    cudaMemcpyToSymbol(cte3_sh, &aux, sizeof(Params_brusselator1d_shuffle));
}

void brusselator1d_shuffle::init(double *Y0) const { 
    for (int i=0;i<nx;i++){  
        double x_i=(double)(i+1)*dtx;
        Y0[idx(i,0)]=A+sin(2*PI*x_i);
        Y0[idx(i,1)]=B;
    }  
}

// Version del Kernel en la que usamos shuffle
__global__ void kernel_brusselator1d_shuffle (const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int ult_x = cte3_sh.nx-1, // Ultimo valor de la coordenada X
        id_x = thread >> 1,      // Identificamos coordenada X -> th/2
        id_z = threadIdx.x & 1,       // Coordenada Z -> th%2
        lane = threadIdx.x & 31; // Posicion en el warp -> th%32

    if(thread < cte3_sh.neqn){
        const double C = id_z==0 ? cte3_sh.A : cte3_sh.B;
        const unsigned mask = 0xFFFFFFFF;
        const double valor = Y[thread];
        
        // Hacemos los shuffles con todos los hilos, incluidos los no validos
        double val_ant  = __shfl_up_sync  (mask, valor, 2), // th(n-2) -> th(n)
               val_pst  = __shfl_down_sync(mask, valor, 2), // th(n) <- th(n+2)
               par_up   = __shfl_up_sync  (mask, valor, 1), // th(n-1) -> th(n)
               par_down = __shfl_down_sync(mask, valor, 1); // th(n) <- th(n+1)
        
        // Reescribimos los shuffles que no son correctos
        if(lane<=1 || id_x==0) // lanes 0 y 1 no pueden shuffle, idx==0 es frontera
            val_ant =   id_x==0   ? C : Y[thread-2];
        if(lane>=30 || id_x==ult_x) // lanes 30 y 31 no pueden shuffle, idx==ult es frontera
            val_pst = id_x==ult_x ? C : Y[thread+2];

        const double val_pareja = id_z==0 ? par_down : par_up; // Nos quedamos el valor pareja apropiado

        DY[thread] = cte3_sh.DD * (val_pst - 2.0 * valor + val_ant);

        const double ui = id_z==0 ? valor : val_pareja,
                     vi = id_z==1 ? valor : val_pareja;
        const double u2v=ui*ui*vi;
        DY[thread] += id_z==0 ? (cte3_sh.A + u2v - (cte3_sh.B+1)*ui) : (cte3_sh.B * ui - u2v);
    }
}

void brusselator1d_shuffle::feval (const double &t, const double *Y, double *DY) const {
    kernel_brusselator1d_shuffle<<<this->grid,this->block>>>(t,Y,DY);
}
void brusselator1d_shuffle::feval (const double &h, const double *Y, double* DY, cudaStream_t stream) const {
    kernel_brusselator1d_shuffle<<<this->grid, this->block, 0, stream>>>(0.0, Y, DY);
}
  
#endif   