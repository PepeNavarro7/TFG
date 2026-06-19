#ifndef METODO_CU
#define METODO_CU

#include "Metodo.h"

using namespace std;

__global__ void kernel_escalarPorVector(const double esc, const double* __restrict__ X, double* __restrict__ Y, const int neqn){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<neqn){
        Y[i]+=X[i]*esc;
    }
}

void Metodo::escalarPorVector(const double &esc, const double *X, double *Y) const {
    kernel_escalarPorVector<<<NUM_BLOCKS,THREADSPERBLOCK>>>(esc, X, Y, neqn);
}

#endif