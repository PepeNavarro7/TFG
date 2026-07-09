#ifndef METODO_CU
#define METODO_CU

#include "Metodo.h"

using namespace std;

// Variable en memoria constante (vive en la GPU)
__constant__ Params_Metodo cteMet;

// Definicion de los valores constantes para el kernel
void Metodo::update() const {
    Params_Metodo aux;

    aux.num = this->neqn;

    cudaMemcpyToSymbol(cteMet, &aux, sizeof(Params_Metodo));
}

// Y[i] += esc * X[i] 
__global__ void kernel_escalarPorVector(const double esc, const double* __restrict__ X, double* __restrict__ Y){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<cteMet.num){
        Y[i]+=X[i]*esc;
    }
}

// Yf[i] = Y0[i] + esc * X[i]
__global__ void kernel_escalarSumaMult(const double* __restrict__ Y0, const double esc, const double* __restrict__ X, double* __restrict__ Yf){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<cteMet.num){
        Yf[i] = Y0[i] + X[i]*esc;
    }
}

// Yf[i] = Y0[i] + esc1 * X[i] + esc2 * Z[i]
__global__ void kernel_escalarSuma2Mult(const double* __restrict__ Y0, const double esc1, const double* __restrict__ X, const double esc2, const double* __restrict__ Z, double* __restrict__ Yf){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<cteMet.num){
        Yf[i] = Y0[i] + X[i]*esc1 + Z[i]*esc2;
    }
}

#endif