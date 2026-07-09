#ifndef PRUEBA_CU
#define PRUEBA_CU

#include "prueba.h"
#include <cmath>

using namespace std;

// Variable en memoria constante (vive en la GPU)
__constant__ Params_prueba cte0;

// Definicion de los valores constantes para el kernel
void prueba::updateConstants() const {
    Params_prueba aux;

    aux.neqn = neqn;

    cudaMemcpyToSymbol(cte0, &aux, sizeof(Params_prueba));
}

// Initialize stage vector Y0 with neqn components
void prueba::init(double *Y0) const {
    for (int i=0;i<neqn;i++) { 
        Y0[i]=i;
    }
}

__global__ void kernel_prueba(const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cte0.neqn){
        DY[thread] = 2.0 * Y[thread] - 6.0;
    }
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void prueba::feval(const double &t, const double* Y, double* DY) const {
    kernel_prueba<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t,Y,DY);
}
#endif