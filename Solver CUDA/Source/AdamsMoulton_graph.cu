#ifndef ADAMS_MOULTON_GRAPH_CU
#define ADAMS_MOULTON_GRAPH_CU

#include "AdamsMoulton_graph.h"
#include "AdamsMoulton.h"
#include "AdamsBashford.h"
#include "AdamsBashford_graph.h"
#include <iostream>
#include <string>

using namespace std;

// Kernels de Adams-Bashford, Adams-Moulton, y Adams-Bashford_graph que reutilizamos
extern __global__ void kernel_sumatoriaAB1(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB2(double* __restrict__ Yn2, const double* __restrict__ Yn1, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB3(double* __restrict__ Yn3, const double* __restrict__ Yn2, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB4(double* __restrict__ Yn4, const double* __restrict__ Yn3, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAM1(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn1);
extern __global__ void kernel_sumatoriaAM2(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAM3(double* __restrict__ Yn2, const double* __restrict__ Yn1, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAM4(double* __restrict__ Yn3, const double* __restrict__ Yn2, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAM5(double* __restrict__ Yn4, const double* __restrict__ Yn3, const double* __restrict__ Fn4, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_swapAB1(const double* __restrict__ Yn1, double* __restrict__ Yn0, const double* __restrict__ Fn1, double* __restrict__ Fn0);
extern __global__ void kernel_swapAB2(const double* __restrict__ Yn2, double* __restrict__ Yn1, const double* __restrict__ Fn2, double* __restrict__ Fn1, double* __restrict__ Fn0);
extern __global__ void kernel_swapAB3(const double* __restrict__ Yn3, double* __restrict__ Yn2, const double* __restrict__ Fn3, double* __restrict__ Fn2, double* __restrict__ Fn1, double* __restrict__ Fn0);
extern __global__ void kernel_swapAB4(const double* __restrict__ Yn4, double* __restrict__ Yn3, const double* __restrict__ Fn4, double* __restrict__ Fn3, double* __restrict__ Fn2, double* __restrict__ Fn1, double* __restrict__ Fn0);


// Aplicar Adams-Moulton el numero necesario de veces
void AdamsMoulton_graph::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0, *Yn1, *Yn2, *Yn3, *Yn4, // Vectores intermedios
        *Fn0, *Fn1, *Fn2, *Fn3, *Fn4, // vectores funcion
        *Y_AB, *F_AB,   // Vector para aproximar Yn usando Adams-Bashford
        *Y_AM, *F_AM;    // Vector para aproximar Yn usando Adams-Moulton
    cudaMalloc((void**)&Yn0,  get_bytes());
    cudaMalloc((void**)&Yn1,  get_bytes());
    cudaMalloc((void**)&Yn2,  get_bytes());
    cudaMalloc((void**)&Yn3,  get_bytes());
    cudaMalloc((void**)&Yn4,  get_bytes());
    cudaMalloc((void**)&Fn0,  get_bytes());
    cudaMalloc((void**)&Fn1,  get_bytes());
    cudaMalloc((void**)&Fn2,  get_bytes());
    cudaMalloc((void**)&Fn3,  get_bytes());
    cudaMalloc((void**)&Fn4,  get_bytes());
    cudaMalloc((void**)&Y_AB, get_bytes());
    cudaMalloc((void**)&F_AB, get_bytes());
    cudaMalloc((void**)&Y_AM, get_bytes());
    cudaMalloc((void**)&F_AM, get_bytes());
    
    cudaMemcpy(Yn0, Y0, get_bytes(), cudaMemcpyHostToDevice); // Definimos Yn0

    // Constantes para los kernels, tanto de los metodos como del problema
    const double h_RK = h/100.0;
    get_ptr_runge()->updateConstants(get_neqn(), h_RK); // RK funciona con el h pequeño
    get_ptr_bashford()->updateConstants(get_neqn(), h); // Kernels de AB
    ptr_moulton->updateConstants(get_neqn(), h); // Kernels de AM
    ptr_abgraph->updateConstants(get_neqn(), h); // Kernels swap de ABg
    problema->updateConstants();

    switch(get_orden()){ // Aplicamos RK4 para obtener los primeros pasos
        case 1: case 2: break; // No necesitamos RK en orden 1 & 2
        case 3: // Definimos Yn1
            get_ptr_runge()->aplicarUnidad(problema, t0, h_RK, Yn0, Yn1);
        break;
        case 4: // Definimos Yn1 & Yn2 
            get_ptr_runge()->aplicarUnidad(problema, t0,      h_RK, Yn0, Yn1);
            get_ptr_runge()->aplicarUnidad(problema, t0+h_RK, h_RK, Yn1, Yn2);
        break;
        case 5: // Definimos Yn1, Yn2 & Yn3 
            get_ptr_runge()->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            get_ptr_runge()->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
            get_ptr_runge()->aplicarUnidad(problema, t0+h_RK*2.0, h_RK, Yn2, Yn3);
        break;
    } // Fin del switch de RK

    switch(get_orden()){ // Arrancamos los feval
        case 1: case 2:// Definimos el vector feval Fn0
            problema->feval(t0, Yn0, Fn0); // f(tn0,Yn0) -> Fn0
        break;
        case 3:// Definimos los vectores feval Fn0 & Fn1
            problema->feval(t0,      Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK, Yn1, Fn1); // f(tn1,Yn1) -> Fn1
        break;
        case 4:// Definimos los vectores feval Fn0, Fn1 & Fn2
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
        break;
        case 5:// Definimos los vectores feval Fn0, Fn1, Fn2 & Fn3
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
            problema->feval(t0+h_RK*3.0, Yn3, Fn3); // f(tn3,Yn3) -> Fn3
        break;
    } // Fin del switch de arranque

    // Creacion del stream y del graph, y lo capturamos
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    switch(get_orden()){
        case 1: // Yn+1 = Yn + h * Fn+1
            kernel_sumatoriaAB1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Y_AB, Yn0, Fn0); // Y_AB = Yn0 + h * Fn0
            problema->feval(h, Y_AB, F_AB, stream); // f(tn1, Y_AB) -> F_AB
            kernel_sumatoriaAM1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Y_AM, Yn0, F_AB);// Y_AM = Yn0 + h * F_AB
            problema->feval(h, Y_AM, F_AM, stream); // f(tn1, Y_AM) -> F_AM
            kernel_sumatoriaAM1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn1, Yn0, F_AM);
            problema->feval(h, Yn1, Fn1, stream); // f(tn1,Yn1) -> Fn1
            kernel_swapAB1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn1, Yn0, Fn1, Fn0);
        break;
        case 2: // Yn+1 = Yn + h/2 * (Fn+1 + Fn)
            kernel_sumatoriaAB1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Y_AB, Yn0, Fn0); // Y_AB = Yn0 + h * Fn0           
            problema->feval(h, Y_AB, F_AB, stream); // f(tn1, Y_AB) -> F_AB
            kernel_sumatoriaAM2<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Y_AM, Yn0, F_AB, Fn0);// Y_AM = Yn0 + h/2 * (F_AB + Fn0)
            problema->feval(h, Y_AM, F_AM, stream); // f(tn1, Y_AM) -> F_AM
            kernel_sumatoriaAM2<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn1, Yn0, F_AM, Fn0); // Yn1 = Yn0 + h/2 * (F_AM + Fn0)
            problema->feval(h, Yn1, Fn1, stream); // f(tn1,Yn1) -> Fn1
            kernel_swapAB1<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn1, Yn0, Fn1, Fn0);
        break;
        case 3: // Yn+1 = Yn + h/12 * (5Fn+1 + 8Fn - 1Fn-1)
            kernel_sumatoriaAB2<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Y_AB, Yn1, Fn1, Fn0); // Y_AB = Yn1 + h/2 * (3*Fn1 - Fn0)
            problema->feval(h, Y_AB, F_AB, stream); // f(tn2, Y_AB) -> F_AB
            kernel_sumatoriaAM3<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Y_AM, Yn1, F_AB, Fn1, Fn0); // Y_AM = Yn1 + h/12 * (5F_AB + 8Fn1 - 1Fn0)
            problema->feval(h, Y_AM, F_AM, stream); // f(tn2, Y_AM) -> F_AM
            kernel_sumatoriaAM3<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Yn2, Yn1, F_AM, Fn1, Fn0);// Yn2 = Yn1 + h/12 * (5Fn_AM + 8Fn1 - 1Fn0)
            problema->feval(h, Yn2, Fn2, stream); // f(tn2,Yn2) -> Fn2
            kernel_swapAB2<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn2, Yn1, Fn2, Fn1, Fn0);              
        break;
        case 4: // Yn+1 = Yn + h/24 * (9Fn+1 + 19Fn - 5Fn-1 + 1Fn-2)
            kernel_sumatoriaAB3<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Y_AB, Yn2, Fn2, Fn1, Fn0);// Y_AB = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
            problema->feval(h, Y_AB, F_AB, stream); // f(tn3, Y_AB) -> F_AB
            kernel_sumatoriaAM4<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Y_AM, Yn2, F_AB, Fn2, Fn1, Fn0);// Y_AM = Yn2 + h/24 * (9F_AB + 19Fn2 - 5Fn1 + 1Fn0)
            problema->feval(h, Y_AM, F_AM, stream); // f(tn3, Y_AM) -> F_AM
            kernel_sumatoriaAM4<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Yn3, Yn2, F_AM, Fn2, Fn1, Fn0);// Yn3 = Yn2 + h/24 * (9F_AM + 19Fn2 - 5Fn1 + 1Fn0)
            problema->feval(h, Yn3, Fn3, stream); // f(tn3,Yn3) -> Fn3
            kernel_swapAB3<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn3, Yn2, Fn3, Fn2, Fn1, Fn0);               
        break;
        case 5: // Yn+1 = Yn + h/720 * (251Fn+1 + 646Fn - 264Fn-1 + 106Fn-2 - 19Fn-3)
            kernel_sumatoriaAB4<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Y_AB, Yn3, Fn3, Fn2, Fn1, Fn0);// Y_AB = Yn3 + h/24 * (55Fn3 - 59Fn2 + 37Fn1 - 9Fn0)
            problema->feval(h, Y_AB, F_AB, stream); // f(tn4, Y_AB) -> F_AB
            kernel_sumatoriaAM5<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Y_AM, Yn3, F_AB, Fn3, Fn2, Fn1, Fn0); // Y_AM = Yn3 + h/720 * (251F_AB + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)
            problema->feval(h, Y_AM, F_AM, stream); // f(tn4, Y_AM) -> F_AM
            kernel_sumatoriaAM5<<<get_num_blocks(),get_tam_blocks(), 0, stream>>>(Yn4, Yn3, F_AM, Fn3, Fn2, Fn1, Fn0); // Yn4 = Yn3 + h/720 * (251F_AM + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)
            problema->feval(h, Yn4, Fn4, stream); // f(tn4,Yn4) -> Fn4
            kernel_swapAB4<<<get_num_blocks(), get_tam_blocks(), 0, stream>>>(Yn4, Yn3, Fn4, Fn3, Fn2, Fn1, Fn0);               
        break;        
    } // Fin del switch principal

    // Finalizamos la captura del graph
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    double tn = t0;
    tn += get_orden()<=2 ? 0 : h_RK * (get_orden()-2); // Alineamos tn con dónde lo deja el arranque con RK
    for (; tn < tf; tn += h) { // Aqui tenemos el bucle principal en el que lanzamos el graph
        // Usamos memoria constante en device (c_t) y actualizarla antes del launch.
        cudaMemcpyToSymbolAsync(problema->get_t(), &tn, sizeof(double), 0, cudaMemcpyHostToDevice, stream);
        cudaGraphLaunch(instance, stream);
    }
    cudaStreamSynchronize(stream);  
    
    switch(get_orden()){ // Arrastre de los valores finales
        case 1: case 2:
            cudaMemcpy(Yf, Yn0, get_bytes(), cudaMemcpyDeviceToHost);  // Yn0' -> Yf
        break;
        case 3:
            cudaMemcpy(Yf, Yn1, get_bytes(), cudaMemcpyDeviceToHost); // Yn1' -> Yf
        break;
        case 4:
            cudaMemcpy(Yf, Yn2, get_bytes(), cudaMemcpyDeviceToHost);  // Yn2' -> Yf
        break;
        case 5:
            cudaMemcpy(Yf, Yn3, get_bytes(), cudaMemcpyDeviceToHost); // Yn3' -> Yf
        break;
    } // Fin del switch de arrastre

    
    cudaFree(Yn0); cudaFree(Yn1); cudaFree(Yn2); cudaFree(Yn3); cudaFree(Yn4);
    cudaFree(Fn0); cudaFree(Fn1); cudaFree(Fn2); cudaFree(Fn3); cudaFree(Fn4);
    cudaFree(Y_AB); cudaFree(F_AB);
    cudaFree(Y_AM); cudaFree(F_AM);
}
#endif