#ifndef ADVDIFF1D_CU
#define ADVDIFF1D_CU

#include "advdiff1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 2
// Variables en memoria constante (viven en la GPU)
 __constant__ Params_advdiff1d cte_avd1; // Estructura de datos constantes para los kernel
 __constant__ double cte_avd1_t; // Constante que utilizará el graph

// Definicion de los valores constantes para el kernel
void advdiff1d::updateConstants() const {
    Params_advdiff1d aux;

    aux.PI = get_PI();
    aux.neqn = get_neqn();
    aux.dtx = get_dtx();
    aux.dtx_sq_inv = 1.0 / get_dtx_sq(); 
    aux.dtx_4_inv = 1.0 / get_dtx_4(); 
    aux.a = get_a();
    aux.d = get_d();

    cudaMemcpyToSymbol(cte_avd1, &aux, sizeof(Params_advdiff1d));
}

// Initialize stage vector Y0 with neqn components
void advdiff1d::init(double *Y0) const {
    for (int i=0; i<get_neqn(); ++i) { 
        double x_i = (double)(i+1)*get_dtx();
        Y0[i] = sin(2.0*get_PI()*x_i);
    }
}

// Kernel para paralelizar con CUDA el feval del problema
__global__ void kernel_advdiff1d(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x,
            ult = cte_avd1.neqn-1;

    if(thread<cte_avd1.neqn){
        // Indice de los vecinos + frontera
        const int i_ant =  thread==0  ? ult : thread-1,
                  i_pst = thread==ult ?  0  : thread+1;
        // Obtenemos los valores
        const double val_ant = Y[i_ant],
                     valor   = Y[thread],
                     val_pst = Y[i_pst];
        
        // Realizamos los calculos
        DY[thread] = cte_avd1.d * (val_pst -    2.0 * valor    + val_ant) * cte_avd1.dtx_sq_inv
                   - cte_avd1.a * (val_pst * val_pst - val_ant * val_ant) * cte_avd1.dtx_4_inv;

        const double x = (thread+1) * cte_avd1.dtx;
        const double pi2xpt = 2.0 * cte_avd1.PI * x + t;
        const double c=cos(pi2xpt), s=sin(pi2xpt); 
        const double res = c + 2.0 * cte_avd1.a * cte_avd1.PI * s * c 
                         + 4.0 * cte_avd1.d * cte_avd1.PI * cte_avd1.PI * s - s;

        DY[thread] += Y[thread] + res;
    }
}

__global__ void graph_advdiff1d(const double offset, const double* __restrict__ Y, double* __restrict__ DY){
    const double t = cte_avd1_t + offset;
    const int thread = blockDim.x * blockIdx.x + threadIdx.x,
            ult = cte_avd1.neqn-1;

    if(thread<cte_avd1.neqn){
        // Indice de los vecinos + frontera
        const int i_ant =  thread==0  ? ult : thread-1,
                  i_pst = thread==ult ?  0  : thread+1;
        // Obtenemos los valores
        const double val_ant = Y[i_ant],
                     valor   = Y[thread],
                     val_pst = Y[i_pst];
        
        // Realizamos los calculos
        DY[thread] = cte_avd1.d * (val_pst -    2.0 * valor    + val_ant) * cte_avd1.dtx_sq_inv
                   - cte_avd1.a * (val_pst * val_pst - val_ant * val_ant) * cte_avd1.dtx_4_inv;

        const double x = (thread+1) * cte_avd1.dtx;
        const double pi2xpt = 2.0 * cte_avd1.PI * x + t;
        const double c=cos(pi2xpt), s=sin(pi2xpt); 
        const double res = c + 2.0 * cte_avd1.a * cte_avd1.PI * s * c 
                         + 4.0 * cte_avd1.d * cte_avd1.PI * cte_avd1.PI * s - s;

        DY[thread] += Y[thread] + res;
    }
}

void advdiff1d::feval (const double &t, const double *Y, double *DY) const {
    kernel_advdiff1d<<<get_grid(), get_block()>>>(t, Y, DY);
}
void advdiff1d::feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const {
    graph_advdiff1d<<<get_grid(), get_block(), 0, stream>>>(offset, Y, DY);
}


/*double advdiff1d::f(const double &x, const double &t) const{ 
    const double pi2xpt=2.0*PI*x + t;
    const double c=cos(pi2xpt);
    const double s=sin(pi2xpt); 
    return( c + 2*a*PI* s*c + 4*d*PI*PI*s - s);
}*/

#endif
