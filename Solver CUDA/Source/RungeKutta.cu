#ifndef RUNGE_KUTTA_CU
#define RUNGE_KUTTA_CU

#include <iostream>
#include <fstream>
#include <cmath>
#include "RungeKutta.h"

using namespace std;

__global__ void d_sumatoriaRK(double *Yn, const double h, const double *K1, const double *K2, const double *K3, const double *K4, const int neqn) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<neqn){
        Yn[i] += (h/6.0) * (K1[i] + 2.0*K2[i] + 2.0*K3[i] + K4[i]);
    }
}

void RungeKutta::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    // creamos y alojamos memoria para los arrays en el device
    double *Yn, // vector con la Y en cada iteracion
        *K1, *K2, *K3, *K4, // Vectores de cada paso
        *Yaux; // Vector auxiliar
    cudaMalloc((void**)&Yn,NUM_BYTES);
    cudaMalloc((void**)&K1,NUM_BYTES);
    cudaMalloc((void**)&K2,NUM_BYTES);
    cudaMalloc((void**)&K3,NUM_BYTES);
    cudaMalloc((void**)&K4,NUM_BYTES);
    cudaMalloc((void**)&Yaux,NUM_BYTES);
    
    cudaMemcpy(Yn, Y0, NUM_BYTES, cudaMemcpyHostToDevice); // Y0 -> Yn, para primera iteración
    
    for(double tn=t0; tn<tf; tn+=h){ // Todas las llamadas han de hacerse de forma secuencial, ya que cada una necesita de la anterior
        // K1 = feval(tn, Yn)
        problema->feval(tn,Yn,K1);              // Definimos K1

        // K2 = feval(tn + h/2, Yn + K1*h/2)
        cudaMemcpy(Yaux, Yn, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn -> Yaux
        escalarPorVector(0.5*h, K1, Yaux);      // Yaux += K1*h/2
        problema->feval(tn + 0.5*h, Yaux, K2);  // Definimos K2

        // K3 = feval(tn + h/2, Yn + K2*h/2)
        cudaMemcpy(Yaux, Yn, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn -> Yaux
        escalarPorVector(0.5*h, K2, Yaux);      // Yaux += K2*h/2
        problema->feval(tn + 0.5*h, Yaux, K3);  // Definimos K3

        // K4 = feval(tn + h, Y0 + h*K3)
        cudaMemcpy(Yaux, Yn, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn -> Yaux
        escalarPorVector(h, K3, Yaux);          // Yaux += h*K3
        problema->feval(tn+h, Yaux, K4);        // Definimos K4


        // Yn+1 = Yn + K1*h/6 + K2*h/3 + K3*h/3 + K4*h/6
        d_sumatoriaRK<<<NUM_BLOCKS,THREADSPERBLOCK>>>(Yn, h, K1, K2, K3, K4, neqn);
        // Tras las sumas, el vector Yn ahora es Yn+1
    }
    
    // Yn es el valor que arrastramos de la ultima iteracion
    cudaMemcpy(Yf, Yn, NUM_BYTES, cudaMemcpyDeviceToHost); // Yn -> Yf
    cudaFree(Yn);
    cudaFree(K1);
    cudaFree(K2);
    cudaFree(K3);
    cudaFree(K4);
    cudaFree(Yaux);
}

void RungeKutta::aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf) const {
    double *K1, *K2, *K3, *K4, // Vectores de cada paso
        *Yaux, *Yn0; // Vector auxiliar
    cudaMalloc((void**)&K1,NUM_BYTES);
    cudaMalloc((void**)&K2,NUM_BYTES);
    cudaMalloc((void**)&K3,NUM_BYTES);
    cudaMalloc((void**)&K4,NUM_BYTES);
    cudaMalloc((void**)&Yaux,NUM_BYTES);
    cudaMalloc((void**)&Yn0,NUM_BYTES);

    cudaMemcpy(Yn0, Y0, NUM_BYTES, cudaMemcpyHostToDevice); // Y0 -> Yn0
      
    // K1 = feval(tn, Yn)
    problema->feval(t0,Yn0,K1);              // Definimos K1

    // K2 = feval(tn + h/2, Yn + K1*h/2)
    cudaMemcpy(Yaux, Yn0, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn0 -> Yaux
    escalarPorVector(0.5*h, K1, Yaux);      // Yaux += K1*h/2
    problema->feval(t0 + 0.5*h, Yaux, K2);  // Definimos K2

    // K3 = feval(tn + h/2, Yn + K2*h/2)
    cudaMemcpy(Yaux, Yn0, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn0 -> Yaux
    escalarPorVector(0.5*h, K2, Yaux);      // Y1 += K2*h/2
    problema->feval(t0 + 0.5*h, Yaux, K3);  // Definimos K3

    // K4 = feval(tn + h, Y0 + h*K3)
    cudaMemcpy(Yaux, Yn0, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn0 -> Yaux
    escalarPorVector(h, K3, Yaux);          // Y1 += h*K3
    problema->feval(t0+h, Yaux, K4);        // Definimos K4

    // Yn+1 = Yn + K1*h/6 + K2*h/3 + K3*h/3 + K4*h/6
    cudaMemcpy(Yaux, Yn0, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn0 -> Yaux
    d_sumatoriaRK<<<NUM_BLOCKS,THREADSPERBLOCK>>>(Yaux, h, K1, K2, K3, K4, neqn);
    
    cudaMemcpy(Yf, Yaux, NUM_BYTES, cudaMemcpyDeviceToHost); // Yaux -> Yf
    
    cudaFree(K1);
    cudaFree(K2);
    cudaFree(K3);
    cudaFree(K4);
    cudaFree(Yaux);
    cudaFree(Yn0);
}


#endif