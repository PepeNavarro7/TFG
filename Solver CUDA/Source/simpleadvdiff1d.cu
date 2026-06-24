#ifndef SIMPLE_AVD_DIFF_CU
#define SIMPLE_AVD_DIFF_CU

#include "simpleadvdiff1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
struct Params_simpleadvdiff1d {
    int neqn;
    double dtx_2_inv;
    double dtx_sq_inv;
    double a;
    double d;
};

// Variable en memoria constante (vive en la GPU)
__constant__ Params_simpleadvdiff1d cte1;

// Definicion de los valores constantes para el kernel
void simpleadvdiff1d::updateConstants() const {
    Params_simpleadvdiff1d aux;

    aux.neqn = neqn;
    aux.dtx_2_inv  = 1.0 / dtx_2;
    aux.dtx_sq_inv = 1.0 / dtx_sq;
    aux.a = a;
    aux.d = d;

    cudaMemcpyToSymbol(cte1, &aux, sizeof(Params_simpleadvdiff1d));
}

// Initialize stage vector Y0 with neqn components
void simpleadvdiff1d::init(double *Y0) const {
    for (int i=0;i<neqn;i++) { 
        double x_i=(double)(i+1)*dtx;
        Y0[i]=sin(2.0*PI*x_i);
    }
}

// Kernel para paralelizar con CUDA el feval del problema
__global__ void kernel_simpleadvdiff1d(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<cte1.neqn){
        const int ult = cte1.neqn-1;
        const int i_ant = i ==  0  ? ult : i-1, // Calculamos indices vecinos
                  i_pst = i == ult ?  0  : i+1;

        const double v_ant = Y[i_ant], // Acceso a los indices
                     valor = Y[i],
                     v_pst = Y[i_pst];
        
        DY[i] = cte1.d * (v_pst - 2.0*valor + v_ant) * cte1.dtx_sq_inv // Calculo del valor feval
              - cte1.a * (v_pst             - v_ant) * cte1.dtx_2_inv;    
    }    
}

// Version del Kernel en la que usamos shuffle
__global__ void kernel2_simpleadvdiff1d(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x,
        ult = cte1.neqn-1, // Último valor
        lane = threadIdx.x & 31; // Posicion en el warp -> th%32
    if(thread<cte1.neqn){
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
        
        DY[thread] = cte1.d * (val_pst - 2.0*valor + val_ant) * cte1.dtx_sq_inv // Calculo del valor feval
                   - cte1.a * (val_pst             - val_ant) * cte1.dtx_2_inv;    
    }    
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void simpleadvdiff1d::feval(const double &t, const double* Y, double* DY) const {
    kernel_simpleadvdiff1d<<<grid, block>>>(t, Y, DY);
}

#endif