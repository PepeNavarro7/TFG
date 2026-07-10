#ifndef ADVDIFF1D_CU
#define ADVDIFF1D_CU

#include "advdiff1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 2
// Variable en memoria constante (vive en la GPU)
__constant__ Params_advdiff1d cte2;

// Definicion de los valores constantes para el kernel
void advdiff1d::updateConstants() const {
    Params_advdiff1d aux;

    aux.PI = 3.14159265358979;
    aux.neqn = this->neqn;
    aux.dtx = this->dtx;
    aux.dtx_sq_inv = 1.0 / this->dtx_sq; 
    aux.dtx_4_inv = 1.0 / this->dtx_4; 
    aux.a = this->a;
    aux.d = this->d;

    cudaMemcpyToSymbol(cte2, &aux, sizeof(Params_advdiff1d));
}

// Initialize stage vector Y0 with neqn components
void advdiff1d::init(double *Y0) const {
    for (int i=0; i<neqn; ++i) { 
        double x_i = (double)(i+1)*dtx;
        Y0[i] = sin(2.0*PI*x_i);
    }
}

// Kernel para paralelizar con CUDA el feval del problema
__global__ void kernel_advdiff1d(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int ult = cte2.neqn-1;

    if(thread<cte2.neqn){
        // Indice de los vecinos + frontera
        const int i_ant =  thread==0  ? ult : thread-1,
                  i_pst = thread==ult ?  0  : thread+1;
        // Obtenemos los valores
        const double val_ant = Y[i_ant],
                     valor   = Y[thread],
                     val_pst = Y[i_pst];
        
        // Realizamos los calculos
        DY[thread] = cte2.d * (val_pst -    2.0 * valor    + val_ant) * cte2.dtx_sq_inv
                   - cte2.a * (val_pst * val_pst - val_ant * val_ant) * cte2.dtx_4_inv;

        const double x = (thread+1) * cte2.dtx;
        const double pi2xpt = 2.0 * cte2.PI * x + t;
        const double c=cos(pi2xpt), s=sin(pi2xpt); 
        const double res = c + 2.0 * cte2.a * cte2.PI * s * c 
                         + 4.0 * cte2.d * cte2.PI * cte2.PI * s - s;

        DY[thread] += Y[thread] + res;
    }
}

// Version del Kernel en la que usamos shuffle
__global__ void kernel2_advdiff1d(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x,
        ult = cte2.neqn-1, // Ultimo valor
        lane = threadIdx.x & 31; // Posicion en el warp -> th%32

    if(thread<cte2.neqn){
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
        
        DY[thread] = cte2.d * (val_pst -    2.0 * valor    + val_ant) * cte2.dtx_sq_inv
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
    kernel_advdiff1d<<<this->grid,this->block>>>(t, Y, DY);
}
void advdiff1d::feval (const double &h, const double *Y, double* DY, cudaStream_t stream) const {
    kernel_advdiff1d<<<this->grid, this->block, 0, stream>>>(0.0, Y, DY);
}


/*double advdiff1d::f(const double &x, const double &t) const{ 
    const double pi2xpt=2.0*PI*x + t;
    const double c=cos(pi2xpt);
    const double s=sin(pi2xpt); 
    return( c + 2*a*PI* s*c + 4*d*PI*PI*s - s);
}*/

#endif