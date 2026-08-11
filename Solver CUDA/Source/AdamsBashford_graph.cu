#ifndef ADAMS_BASHFORD_GRAPH_CU
#define ADAMS_BASHFORD_GRAPH_CU

#include "AdamsBashford_graph.h"
#include <iostream>

using namespace std;

// Kernels de Adams-Bashford que utilizamos
extern __global__ void kernel_sumatoriaAB4(double* __restrict__ Yn4, const double* __restrict__ Yn3, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB3(double* __restrict__ Yn3, const double* __restrict__ Yn2, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB2(double* __restrict__ Yn2, const double* __restrict__ Yn1, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB1(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn0);

// Variable en memoria constante (vive en la GPU)
__constant__ Params_AdamsBashford cte_ABg;

// Definicion de los valores constantes para el kernel
void AdamsBashford_graph::updateConstants(const int &neqn, const double &h) const {
    Params_AdamsBashford aux;

    aux.neqn = neqn;
    aux.h = h;
    aux.h2 = h/2.0;
    aux.h12 = h/12.0;
    aux.h24 = h/24.0;

    cudaMemcpyToSymbol(cte_ABg, &aux, sizeof(Params_AdamsBashford));
}

__global__ void kernel_swapAB4(const double* __restrict__ Yn4, double* __restrict__ Yn3, const double* __restrict__ Fn4, double* __restrict__ Fn3, double* __restrict__ Fn2, double* __restrict__ Fn1, double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cte_ABg.neqn){
        Yn3[thread] = Yn4[thread]; // Yn4 -> Yn3
        Fn0[thread] = Fn1[thread]; // Fn1 -> Fn0
        Fn1[thread] = Fn2[thread]; // Fn2 -> Fn1
        Fn2[thread] = Fn3[thread]; // Fn3 -> Fn2
        Fn3[thread] = Fn4[thread]; // Fn4 -> Fn3
    }
}
__global__ void kernel_swapAB3(const double* __restrict__ Yn3, double* __restrict__ Yn2, const double* __restrict__ Fn3, double* __restrict__ Fn2, double* __restrict__ Fn1, double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cte_ABg.neqn){
        Yn2[thread] = Yn3[thread]; // Yn3 -> Yn2
        Fn0[thread] = Fn1[thread]; // Fn1 -> Fn0
        Fn1[thread] = Fn2[thread]; // Fn2 -> Fn1
        Fn2[thread] = Fn3[thread]; // Fn3 -> Fn2
    }
}
__global__ void kernel_swapAB2(const double* __restrict__ Yn2, double* __restrict__ Yn1, const double* __restrict__ Fn2, double* __restrict__ Fn1, double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cte_ABg.neqn){
        Yn1[thread] = Yn2[thread]; // Yn2 -> Yn1
        Fn0[thread] = Fn1[thread]; // Fn1 -> Fn0
        Fn1[thread] = Fn2[thread]; // Fn2 -> Fn1
    }
}
__global__ void kernel_swapAB1(const double* __restrict__ Yn1, double* __restrict__ Yn0, const double* __restrict__ Fn1, double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread<cte_ABg.neqn){
        Yn0[thread] = Yn1[thread]; // Yn1 -> Yn0
        Fn0[thread] = Fn1[thread]; // Fn1 -> Fn0
    }
}


// Aplicar Adams-Bashford 
void AdamsBashford_graph::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0, *Yn1, *Yn2, *Yn3, *Yn4, // Vectores intermedios
           *Fn0, *Fn1, *Fn2, *Fn3, *Fn4; // vectores funcion

    cudaMalloc((void**)&Yn0,get_bytes());
    cudaMalloc((void**)&Yn1,get_bytes());
    cudaMalloc((void**)&Yn2,get_bytes());
    cudaMalloc((void**)&Yn3,get_bytes());
    cudaMalloc((void**)&Yn4,get_bytes());
    cudaMalloc((void**)&Fn0,get_bytes());
    cudaMalloc((void**)&Fn1,get_bytes());
    cudaMalloc((void**)&Fn2,get_bytes());
    cudaMalloc((void**)&Fn3,get_bytes());
    cudaMalloc((void**)&Fn4,get_bytes());

    cudaMemcpy(Yn0, Y0, get_bytes(), cudaMemcpyHostToDevice); // Y0 -> Yn0

    // Constantes para los kernels, tanto de los metodos como del problema
    const double h_RK = h/100.0;
    get_ptr_runge()->updateConstants(get_neqn(), h_RK); // RK funciona con el h pequeño
    ptr_bash->updateConstants(get_neqn(), h); // Kernels de AB normal
    updateConstants(get_neqn(), h); // Kernels propios
    problema->updateConstants();
    

    switch(get_orden()){ // Aplicamos Runge-Kutta para obtener los primeros pasos
        case 1: break; // Orden 1 no necesita RK
        case 2: 
            get_ptr_runge()->aplicarUnidad(problema, t0, h_RK, Yn0, Yn1);
        break;
        case 3: 
            get_ptr_runge()->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            get_ptr_runge()->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
        break;
        case 4:
            get_ptr_runge()->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            get_ptr_runge()->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
            get_ptr_runge()->aplicarUnidad(problema, t0+h_RK*2.0, h_RK, Yn2, Yn3);
        break;
    }
    switch(get_orden()){ // Aplicamos los feval iniciales
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
    }

    // Creacion del stream y del graph, y lo capturamos
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    switch(get_orden()){ // Aplicamos Adams-Bashford una vez para el graph
        case 1: 
            kernel_sumatoriaAB1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn1, Yn0, Fn0); // Yn1 = Yn0 + h * Fn0
            problema->feval(h, Yn1, Fn1, stream); // f(tn1,Yn1) -> Fn1
            kernel_swapAB1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn1, Yn0, Fn1, Fn0);
        break;
        case 2:
            kernel_sumatoriaAB2<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn2, Yn1, Fn1, Fn0);// Yn2 = Yn1 + h/2 * (3*Fn1 - Fn0)
            problema->feval(h, Yn2, Fn2, stream); // f(tn2,Yn2) -> Fn2
            kernel_swapAB2<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn2, Yn1, Fn2, Fn1, Fn0);
        break;
        case 3:
            kernel_sumatoriaAB3<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn3, Yn2, Fn2, Fn1, Fn0); // Yn3 = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
            problema->feval(h, Yn3, Fn3, stream); // f(tn3,Yn3) -> Fn3
            kernel_swapAB3<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn3, Yn2, Fn3, Fn2, Fn1, Fn0);
        break;
        case 4:
            kernel_sumatoriaAB4<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn4, Yn3, Fn3, Fn2, Fn1, Fn0); // Yn4 = Yn3 + h/24 * (55*Fn3 - 59*Fn2 + 37*Fn1 - 9*Fn0)
            problema->feval(h, Yn4, Fn4, stream); // f(tn4,Yn4) -> Fn4
            kernel_swapAB4<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn4, Yn3, Fn4, Fn3, Fn2, Fn1, Fn0);
            break;
    }

    // Finalizamos la captura del graph
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    // Aqui tenemos el bucle principal en el que lanzamos el graph
    for (double tn = t0 + h_RK * (get_orden()-1); tn < tf; tn += h) {
        // Usamos memoria constante en device (c_t) y actualizarla antes del launch.
        cudaMemcpyToSymbolAsync(problema->get_t(), &tn, sizeof(double), 0, cudaMemcpyHostToDevice, stream);
        cudaGraphLaunch(instance, stream);
    }

    cudaStreamSynchronize(stream);     

    switch(get_orden()){ // Switch para arrastrar el vector resultado
        case 1:
            cudaMemcpy(Yf, Yn0, get_bytes(), cudaMemcpyDeviceToHost); // Yn0 -> Yf
        break;
        case 2:
            cudaMemcpy(Yf, Yn1, get_bytes(), cudaMemcpyDeviceToHost); // Yn1 -> Yf
        break;
        case 3:
            cudaMemcpy(Yf, Yn2, get_bytes(), cudaMemcpyDeviceToHost); // Yn2 -> Yf
        break;
        case 4:
            cudaMemcpy(Yf, Yn3, get_bytes(), cudaMemcpyDeviceToHost); // Yn3 -> Yf
        break;
    } // Fin del switch resultado

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    
    cudaFree(Yn0); cudaFree(Yn1); cudaFree(Yn2); cudaFree(Yn3); cudaFree(Yn4);
    cudaFree(Fn0); cudaFree(Fn1); cudaFree(Fn2); cudaFree(Fn3); cudaFree(Fn4);
}

#endif