#ifndef METODO_CU
#define METODO_CU

#include "Metodo.h"

using namespace std;

__global__ void d_escalarPorVector(const double &esc, const double *X, double *Y, const int &neqn){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<neqn){
        Y[i]+=X[i]*esc;
    }
}

void Metodo::escalarPorVector(const double &esc, const double *X, double *Y){
    d_escalarPorVector<<<NUM_BLOCKS,THREADSPERBLOCK>>>(esc, X, Y, neqn);
}

#endif