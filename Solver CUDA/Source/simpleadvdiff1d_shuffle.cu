#ifndef SIMPLE_AVD_DIFF_SHUFFLE_CU
#define SIMPLE_AVD_DIFF_SHUFFLE_CU

#include "simpleadvdiff1d_shuffle.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model

// Variables en memoria constante (viven en la GPU)
__constant__ Params_simpleadvdiff1d cte_savds; // Estructura de datos constantes para los kernel
__constant__ double cte_savds_t; // Constante que utilizará el graph

void simpleadvdiff1d_shuffle::updateConstants() const {
    Params_simpleadvdiff1d aux;

    aux.neqn = get_neqn();
    aux.dtx_2_inv  = 1.0 / get_dtx_2();
    aux.dtx_sq_inv = 1.0 / get_dtx_sq();
    aux.a = get_a();
    aux.d = get_d();

    cudaMemcpyToSymbol(cte_savds, &aux, sizeof(Params_simpleadvdiff1d));
}

// Version del Kernel en la que usamos shuffle
__global__ void kernel_simpleadvdiff1d_shuffle(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x,
        ult = cte_savds.neqn-1, // Último valor
        lane = threadIdx.x & 31; // Posicion en el warp -> th%32
    if(thread<cte_savds.neqn){
        const unsigned mask = 0xFFFFFFFF;
        const double valor = Y[thread];

        // Hacemos los shuffles con todos los hilos, incluidos los no validos
        double val_ant = __shfl_up_sync  (mask, valor, 1), // th(n-1) -> th(n)
               val_pst = __shfl_down_sync(mask, valor, 1); // th(n) <- th(n+1)

        // Corregimos los shuffles erróneos & los valores frontera
        if(lane==0 || thread==0)
            val_ant =  thread==0  ? Y[ult] : Y[thread-1];
        if(lane==31 || thread==ult)
            val_pst = thread==ult ?  Y[0]  : Y[thread+1];        
        
        DY[thread] = cte_savds.d * (val_pst - 2.0*valor + val_ant) * cte_savds.dtx_sq_inv // Calculo del valor feval
                   - cte_savds.a * (val_pst             - val_ant) * cte_savds.dtx_2_inv;    
    }    
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void simpleadvdiff1d_shuffle::feval(const double &t, const double* Y, double* DY) const {
    kernel_simpleadvdiff1d_shuffle<<<get_grid(), get_block()>>>(t, Y, DY);
}
void simpleadvdiff1d_shuffle::feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const {
    kernel_simpleadvdiff1d_shuffle<<<get_grid(), get_block(), 0, stream>>>(offset, Y, DY);
}

#endif