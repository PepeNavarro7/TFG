#ifndef ADVDIFF1D_SHUFFLE_CU
#define ADVDIFF1D_SHUFFLE_CU

#include "advdiff1d_shuffle.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 2
// Variables en memoria constante (viven en la GPU)
__constant__ Params_advdiff1d cte_avd1s; // Estructura de datos constantes para los kernel
__constant__ double cte_avd1s_t; // Constante que utilizará el graph

// Definicion de los valores constantes para el kernel
void advdiff1d_shuffle::updateConstants() const {
    Params_advdiff1d aux;

    aux.PI = get_PI();
    aux.neqn = get_neqn();
    aux.dtx = get_dtx();
    aux.dtx_sq_inv = 1.0 / get_dtx_sq(); 
    aux.dtx_4_inv = 1.0 / get_dtx_4(); 
    aux.a = get_a();
    aux.d = get_d();

    cudaMemcpyToSymbol(cte_avd1s, &aux, sizeof(Params_advdiff1d));
}

// Version del Kernel en la que usamos shuffle
__global__ void kernel_advdiff1d_shuffle(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x,
        ult = cte_avd1s.neqn-1, // Ultimo valor
        lane = threadIdx.x & 31; // Posicion en el warp -> th%32

    if(thread<cte_avd1s.neqn){
        const unsigned mask = 0xFFFFFFFF;
        const double valor = Y[thread];

        // Hacemos los shuffles con todos los hilos, incluidos los erroneos
        double val_ant = __shfl_up_sync  (mask, valor, 1), // th(n-1) -> th(n)
               val_pst = __shfl_down_sync(mask, valor, 1); // th(n) <- th(n+1)

        // Corregimos los shuffles erróneos & los valores frontera
        if(lane==0 || thread==0)
            val_ant =  thread==0  ? Y[ult] : Y[thread-1];
        if(lane==31 || thread==ult)
            val_pst = thread==ult ?  Y[0]  : Y[thread+1]; 
        
        DY[thread] = cte_avd1s.d * (val_pst -    2.0 * valor    + val_ant) * cte_avd1s.dtx_sq_inv
                   - cte_avd1s.a * (val_pst * val_pst - val_ant * val_ant) * cte_avd1s.dtx_4_inv;

        const double x = (thread+1) * cte_avd1s.dtx;
        const double pi2xpt = 2.0 * cte_avd1s.PI * x + t;
        const double c=cos(pi2xpt), s=sin(pi2xpt); 
        const double res = c + 2.0 * cte_avd1s.a * cte_avd1s.PI * s * c 
                         + 4.0 * cte_avd1s.d * cte_avd1s.PI * cte_avd1s.PI * s - s;

        DY[thread] += Y[thread] + res;
    }
}
__global__ void graph_advdiff1d_shuffle(const double offset, const double* __restrict__ Y, double* __restrict__ DY){
    const double t = cte_avd1s_t + offset;
    const int thread = blockDim.x * blockIdx.x + threadIdx.x,
        ult = cte_avd1s.neqn-1, // Ultimo valor
        lane = threadIdx.x & 31; // Posicion en el warp -> th%32

    if(thread<cte_avd1s.neqn){
        const unsigned mask = 0xFFFFFFFF;
        const double valor = Y[thread];

        // Hacemos los shuffles con todos los hilos, incluidos los erroneos
        double val_ant = __shfl_up_sync  (mask, valor, 1), // th(n-1) -> th(n)
               val_pst = __shfl_down_sync(mask, valor, 1); // th(n) <- th(n+1)

        // Corregimos los shuffles erróneos & los valores frontera
        if(lane==0 || thread==0)
            val_ant =  thread==0  ? Y[ult] : Y[thread-1];
        if(lane==31 || thread==ult)
            val_pst = thread==ult ?  Y[0]  : Y[thread+1]; 
        
        DY[thread] = cte_avd1s.d * (val_pst -    2.0 * valor    + val_ant) * cte_avd1s.dtx_sq_inv
                   - cte_avd1s.a * (val_pst * val_pst - val_ant * val_ant) * cte_avd1s.dtx_4_inv;

        const double x = (thread+1) * cte_avd1s.dtx;
        const double pi2xpt = 2.0 * cte_avd1s.PI * x + t;
        const double c=cos(pi2xpt), s=sin(pi2xpt); 
        const double res = c + 2.0 * cte_avd1s.a * cte_avd1s.PI * s * c 
                         + 4.0 * cte_avd1s.d * cte_avd1s.PI * cte_avd1s.PI * s - s;

        DY[thread] += Y[thread] + res;
    }
}

void advdiff1d_shuffle::feval (const double &t, const double *Y, double *DY) const {
    kernel_advdiff1d_shuffle<<<get_grid(), get_block()>>>(t, Y, DY);
}
void advdiff1d_shuffle::feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const {
    graph_advdiff1d_shuffle<<<get_grid(), get_block(), 0, stream>>>(offset, Y, DY);
}

#endif