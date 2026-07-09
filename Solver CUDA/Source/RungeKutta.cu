#ifndef RUNGE_KUTTA_CU
#define RUNGE_KUTTA_CU

#include <iostream>
#include <fstream>
#include <cmath>
#include "RungeKutta.h"

using namespace std;

extern __global__ void kernel_escalarPorVector(const double esc, const double* __restrict__ X, double* __restrict__ Y);
extern __global__ void kernel_escalarSumaMult(const double* __restrict__ Y0, const double esc, const double* __restrict__ X, double* __restrict__ Yf);
extern __global__ void kernel_escalarSuma2Mult(const double* __restrict__ Y0, const double esc1, const double* __restrict__ X, const double esc2, const double* __restrict__ Z, double* __restrict__ Yf);

// Variable en memoria constante (vive en la GPU)
__constant__ Params_RungeKutta cteRK;

// Definicion de los valores constantes para el kernel
void RungeKutta::updateConstants(const int &neqn, const double &h) const {
    Params_RungeKutta aux;

    aux.neqn = neqn;
    aux.h = h;
    aux.h2 = h/2.0;
    aux.h6 = h/6.0;

    cudaMemcpyToSymbol(cteRK, &aux, sizeof(Params_RungeKutta));
}

__global__ void kernel_sumatoriaRK4(double* __restrict__ Yn, const double* __restrict__ K1, const double* __restrict__ K2, const double* __restrict__ K3, const double* __restrict__ K4) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cteRK.neqn){ // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
        Yn[thread] += cteRK.h6 * (K1[thread] + 2.0*K2[thread] + 2.0*K3[thread] + K4[thread]);
    }
}

__global__ void kernel_sumatoriaRK3(double* __restrict__ Yn, const double* __restrict__ K1, const double* __restrict__ K2, const double* __restrict__ K3) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cteRK.neqn){ // Yn+1 = Yn + h/6 * (K1 + 4*K2 + K3)
        Yn[thread] += cteRK.h6 * (K1[thread] + 4.0*K2[thread] + K3[thread]);
    }
}

__global__ void kernel_sumatoriaRK2(double* __restrict__ Yn, const double* __restrict__ K1, const double* __restrict__ K2) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cteRK.neqn){ // Yn+1 = Yn + h/2 * (K1 + K2)
        Yn[thread] += cteRK.h2 * (K1[thread] + K2[thread]);
    }
}

void RungeKutta::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    // creamos y alojamos memoria para los arrays en el device
    double *Yn, // Vector con la Y en cada iteracion
        *K1, *K2, *K3, *K4, // Vectores de cada paso
        *Yaux; // Vector auxiliar
    cudaMalloc((void**)&Yn,this->bytes);
    cudaMalloc((void**)&K1,this->bytes);
    cudaMalloc((void**)&K2,this->bytes);
    cudaMalloc((void**)&K3,this->bytes);
    cudaMalloc((void**)&K4,this->bytes);
    cudaMalloc((void**)&Yaux,this->bytes);

    // Constantes para los kernel, tanto de RK como del problema
    this->updateConstants(neqn, h); 
    problema->updateConstants();
    const double h2 = h/2.0, h_n = -1*h, h_2=2.0*h;
    
    cudaMemcpy(Yn, Y0, this->bytes, cudaMemcpyHostToDevice); // Y0 -> Yn, para primera iteración
    for(double tn=t0; tn<tf; tn+=h){ // Todas las llamadas han de hacerse de forma secuencial, ya que cada una necesita de la anterior
        switch(orden){
            case 1:
                // K1 = feval(tn, Yn)
                problema->feval(tn,Yn,K1);        // Definimos K1

                // Yn+1 = Yn + h*K1
                kernel_escalarPorVector<<<this->num_blocks,this->tam_blocks>>>(h, K1, Yn);      // Yn += K1*h
            break;
            case 2:
                // K1 = feval(tn, Yn)
                problema->feval(tn,Yn,K1);              // Definimos K1

                // K2 = feval(tn + h, Yn + h*K1) 
                kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h, K1, Yaux); // Yn + K1*h -> Yaux
                problema->feval(tn + h, Yaux, K2);  // Definimos K2

                // Yn+1 = Yn + h/2 * (K1 + K2)
                kernel_sumatoriaRK2<<<this->num_blocks,this->tam_blocks>>>(Yn, K1, K2);
            break;
            case 3:
                // K1 = feval(tn, Yn)
                problema->feval(tn,Yn,K1);              // Definimos K1

                // K2 = feval(tn + h/2, Yn + h/2 * K1) 
                kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h2, K1, Yaux); // Yaux = Yn + h/2*K1
                problema->feval(tn + h2, Yaux, K2);  // Definimos K2

                // K3 = feval(tn + h, Yn -h*K1 + 2*h*K2 ) 
                kernel_escalarSuma2Mult<<<this->num_blocks,this->tam_blocks>>>(Yn, h_n, K1, h_2, K2, Yaux); // Yaux = Yn -h*K1 +2h*K2
                problema->feval(tn + h, Yaux, K3);  // Definimos K3

                // Yn+1 = Yn + h/6 * (K1 + 4*K2 + K3)
                kernel_sumatoriaRK3<<<this->num_blocks,this->tam_blocks>>>(Yn, K1, K2, K3);
            break;
            case 4:
                // K1 = feval(tn, Yn)
                problema->feval(tn,Yn,K1);              // Definimos K1
                
                // K2 = feval(tn + h/2, Yn + K1*h/2)
                kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h2, K1, Yaux); // Yaux = Yn + h/2*K1
                problema->feval(tn + h2, Yaux, K2);  // Definimos K2

                // K3 = feval(tn + h/2, Yn + K2*h/2)
                kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h2, K2, Yaux); // Yaux = Yn + h/2*K2
                problema->feval(tn + h2, Yaux, K3);  // Definimos K3

                // K4 = feval(tn + h, Y0 + h*K3)
                kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h, K3, Yaux); // Yaux = Yn + h*K3
                problema->feval(tn+h, Yaux, K4);        // Definimos K4

                // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
                kernel_sumatoriaRK4<<<this->num_blocks,this->tam_blocks>>>(Yn, K1, K2, K3, K4);
            break;        // Tras las sumas, el vector Yn ahora es Yn+1
        } // Fin del switch
    } // Fin del bucle for

    // Yn es el valor que arrastramos de la ultima iteracion
    cudaMemcpy(Yf, Yn, this->bytes, cudaMemcpyDeviceToHost); // Yn -> Yf
    cudaFree(Yn);
    cudaFree(K1);
    cudaFree(K2);
    cudaFree(K3);
    cudaFree(K4);
    cudaFree(Yaux);
}

// Aplicamos una sola vez Runge-Kutta de Orden 4
void RungeKutta::aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf) const {
    // creamos y alojamos memoria para los arrays en el device
    double *Yn, // vector con la Y en la iteracion
        *K1, *K2, *K3, *K4, // Vectores de cada paso
        *Yaux; // Vector auxiliar

    cudaMalloc((void**)&Yn,bytes);
    cudaMalloc((void**)&K1,bytes);
    cudaMalloc((void**)&K2,bytes);
    cudaMalloc((void**)&K3,bytes);
    cudaMalloc((void**)&K4,bytes);
    cudaMalloc((void**)&Yaux,bytes);
    
    cudaMemcpy(Yn, Y0, bytes, cudaMemcpyHostToDevice); // Y0 -> Yn
    const double h2=h/2.0;
    
    // K1 = feval(tn, Yn)
    problema->feval(t0,Yn,K1);              // Definimos K1
    
    // K2 = feval(tn + h/2, Yn + K1*h/2)
    kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h2, K1, Yaux); // Yaux = Yn + h/2*K1
    problema->feval(t0 + h2, Yaux, K2);  // Definimos K2

    // K3 = feval(tn + h/2, Yn + K2*h/2)
    kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h2, K2, Yaux); // Yaux = Yn + h/2*K2
    problema->feval(t0 + h2, Yaux, K3);  // Definimos K3

    // K4 = feval(tn + h, Y0 + h*K3)
    kernel_escalarSumaMult<<<this->num_blocks,this->tam_blocks>>>(Yn, h, K3, Yaux); // Yaux = Yn + h*K3
    problema->feval(t0+h, Yaux, K4);        // Definimos K4

    // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
    kernel_sumatoriaRK4<<<num_blocks,tam_blocks>>>(Yn, K1, K2, K3, K4);
    // Tras las sumas, el vector Yn ahora es Yn+1

    // Yn es el valor que arrastramos de la ultima iteracion
    cudaMemcpy(Yf, Yn, bytes, cudaMemcpyDeviceToHost); // Yn -> Yf
    cudaFree(Yn);
    cudaFree(K1);
    cudaFree(K2);
    cudaFree(K3);
    cudaFree(K4);
    cudaFree(Yaux);
}


#endif