#ifndef ADAMS_BASHFORD_CU
#define ADAMS_BASHFORD_CU

#include "AdamsBashford.h"
#include <iostream>

using namespace std;

struct Params_AdamsBashford {
    int neqn;
    double h;
    double h2;
    double h12;
    double h24;
};

// Variable en memoria constante (vive en la GPU)
__constant__ Params_AdamsBashford cteAB;

// Definicion de los valores constantes para el kernel
void AdamsBashford::updateConstants(const int &neqn, const double &h) const {
    Params_AdamsBashford aux;

    aux.neqn = neqn;
    aux.h = h;
    aux.h2 = h/2.0;
    aux.h12 = h/12.0;
    aux.h24 = h/24.0;

    cudaMemcpyToSymbol(cteAB, &aux, sizeof(Params_AdamsBashford));
}

__global__ void kernel_sumatoriaAB4(double* __restrict__ Yn4, const double* __restrict__ Yn3, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAB.neqn){ // Yn4 = Yn3 + h/24 * (55*Fn3 - 59*Fn2 + 37*Fn1 - 9*Fn0)
        Yn4[thread] = Yn3[thread] + cteAB.h24 * (55.0*Fn3[thread] - 59.0*Fn2[thread] + 37.0*Fn1[thread] - 9.0*Fn0[thread]);
    }
}

__global__ void kernel_sumatoriaAB3(double* __restrict__ Yn3, const double* __restrict__ Yn2, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAB.neqn){ // Yn3 = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
        Yn3[thread] = Yn2[thread] + cteAB.h12 * (23.0*Fn2[thread] - 16.0*Fn1[thread] + 5.0*Fn0[thread]);
    }
}

__global__ void kernel_sumatoriaAB2(double* __restrict__ Yn2, const double* __restrict__ Yn1, const double* __restrict__ Fn1, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAB.neqn){ // Yn2 = Yn1 + h/2 * (3*Fn1 - Fn0)
        Yn2[thread] = Yn1[thread] + cteAB.h2 * (3.0 * Fn1[thread] - Fn0[thread]);
    }
}
__global__ void kernel_sumatoriaAB1(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAB.neqn){ // Yn1 = Yn0 + h * Fn0
        Yn1[thread] = Yn0[thread] + cteAB.h * Fn0[thread];
    }
}

// Aplicar Adams-Bashford 
void AdamsBashford::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0, *Yn1, *Yn2, *Yn3, *Yn4, // Vectores intermedios
           *Fn0, *Fn1, *Fn2, *Fn3; // vectores funcion

    cudaMalloc((void**)&Yn0,this->bytes);
    cudaMalloc((void**)&Yn1,this->bytes);
    cudaMalloc((void**)&Yn2,this->bytes);
    cudaMalloc((void**)&Yn3,this->bytes);
    cudaMalloc((void**)&Yn4,this->bytes);
    cudaMalloc((void**)&Fn0,this->bytes);
    cudaMalloc((void**)&Fn1,this->bytes);
    cudaMalloc((void**)&Fn2,this->bytes);
    cudaMalloc((void**)&Fn3,this->bytes);

    cudaMemcpy(Yn0, Y0, this->bytes, cudaMemcpyHostToDevice); // Y0 -> Yn0
    const double h_RK = h/100.0;

    ptr_runge->updateConstants(neqn, h_RK);
    switch(orden){ // Aplicamos orden-1 veces Runge-Kutta para obtener los primeros pasos
        case 1: break; // Orden 1 no necesita RK
        case 2: 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
        break;
        case 3: 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
        break;
        case 4: 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
            ptr_runge->aplicarUnidad(problema, t0+h_RK*2.0, h_RK, Yn2, Yn3);
        break;
    }
    
    switch(orden){ // Arranque generando las feval
        case 1:
            problema->feval(t0, Yn0, Fn0); // f(tn0,Yn0) -> Fn0
        break;
        case 2:
            problema->feval(t0,      Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK, Yn1, Fn1); // f(tn1,Yn1) -> Fn1
        break;
        case 3:
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
        break;
        case 4:
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
            problema->feval(t0+h_RK*3.0, Yn3, Fn3); // f(tn3,Yn3) -> Fn3
        break;
    } // Fin del switch de arranque
    
    updateConstants(neqn, h); // Constantes para los kernel
    // Ahora aplicamos Adams-Bashford del orden indicado
    switch(orden){ // Switch principal con el for que se trabaja
        case 1: 
            for(double tn = t0; tn<tf; tn+=h){
                // Yn1 = Yn0 + h * Fn0
                kernel_sumatoriaAB1<<<this->num_blocks,this->tam_blocks>>>(Yn1, Yn0, Fn0);

                // Ahora que tenemos Yn1, convertimos todos los Yn en Yn-1
                swap(Yn1, Yn0); // Yn1 -> Yn0
                // Tras los cambios, Yn1 contiene basura y será reescrito

                // Calculamos el nuevo Fn0 usando el Yn1 recién creado (que ahora es Yn0)
                problema->feval(tn+h, Yn0, Fn0); // f(tn1,Yn0') -> Fn0'
            } // Fin del bucle for iterativo
        break;
        case 2:
            for(double tn = t0+h_RK*1.0; tn<tf; tn+=h){
                // Yn2 = Yn1 + h/2 * (3*Fn1 - Fn0)
                kernel_sumatoriaAB2<<<this->num_blocks,this->tam_blocks>>>(Yn2, Yn1, Fn1, Fn0);

                // Ahora que tenemos Yn2, convertimos Yn2 a Yn1, y los Fn en Fn-1 para hacer la siguiente iteracion
                swap(Yn2, Yn1); // Yn2 -> Yn1
                swap(Fn1, Fn0); // Fn1 -> Fn0
                // Tras los cambios, Yn2 & Fn1 contienen basura y serán reescritos

                // Calculamos el nuevo Fn1 usando el Yn2 recién creado (que ahora es Yn1)
                problema->feval(tn+h, Yn1, Fn1); // f(tn2,Yn1') -> Fn1'
            } // Fin del bucle for iterativo
        break;
        case 3:
            for(double tn = t0+h_RK*2.0; tn<tf; tn+=h){
                // Yn3 = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
                kernel_sumatoriaAB3<<<this->num_blocks,this->tam_blocks>>>(Yn3, Yn2, Fn2, Fn1, Fn0);

                // Ahora que tenemos Yn3, convertimos Yn3 a Yn2, y los Fn en Fn-1 para hacer la siguiente iteracion
                swap(Yn3, Yn2); // Yn3 -> Yn2
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn2 -> Fn1
                // Tras los cambios, Yn3 & Fn2 contienen basura y serán reescritos

                // Calculamos el nuevo Fn2 usando el Yn3 recién creado (que ahora es Yn2)
                problema->feval(tn+h, Yn2, Fn2); // f(tn3,Yn2') -> Fn2'
            } // Fin del bucle for iterativo
        break;
        case 4:
            for(double tn = t0+h_RK*3.0; tn<tf; tn+=h){
                // Yn4 = Yn3 + h/24 * (55*Fn3 - 59*Fn2 + 37*Fn1 - 9*Fn0)
                kernel_sumatoriaAB4<<<this->num_blocks,this->tam_blocks>>>(Yn4, Yn3, Fn3, Fn2, Fn1, Fn0);
                
                // Ahora que tenemos Yn4, convertimos Yn4 a Yn3, y los Fn en Fn-1 para hacer la siguiente iteracion
                swap(Yn4, Yn3); // Yn4 -> Yn3
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn2 -> Fn1
                swap(Fn3, Fn2); // Fn3 -> Fn2
                // Tras los cambios, Yn4 & Fn3 contienen basura y serán reescritos

                // Calculamos el nuevo Fn3 usando el Yn4 recién creado (que ahora es Yn3)
                problema->feval(tn+h, Yn3, Fn3); // f(tn4,Yn3') -> Fn3'
            } // Fin del bucle for iterativo
        break;
    } // Fin del switch principal

    switch(orden){ // Switch para arrastrar el vector resultado
        case 1:
            cudaMemcpy(Yf, Yn0, this->bytes, cudaMemcpyDeviceToHost); // Yn0 -> Yf
        break;
        case 2:
            cudaMemcpy(Yf, Yn1, this->bytes, cudaMemcpyDeviceToHost); // Yn1 -> Yf
        break;
        case 3:
            cudaMemcpy(Yf, Yn2, this->bytes, cudaMemcpyDeviceToHost); // Yn2 -> Yf
        break;
        case 4:
            cudaMemcpy(Yf, Yn3, this->bytes, cudaMemcpyDeviceToHost); // Yn3 -> Yf
        break;
    } // Fin del switch resultado
    
    cudaFree(Yn0); cudaFree(Yn1); cudaFree(Yn2); cudaFree(Yn3); cudaFree(Yn4);
    cudaFree(Fn0); cudaFree(Fn1); cudaFree(Fn2); cudaFree(Fn3);
}

#endif