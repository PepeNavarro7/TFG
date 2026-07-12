#ifndef RUNGE_KUTTA_GRAPH_CU
#define RUNGE_KUTTA_GRAPH_CU

#include <iostream>
#include <fstream>
#include <cmath>
#include "RungeKutta_graph.h"
#include "Problema.h"   // el cte_t

using namespace std;

extern __global__ void kernel_escalarPorVector(const double esc, const double* __restrict__ X, double* __restrict__ Y);
extern __global__ void kernel_escalarSumaMult(const double* __restrict__ Y0, const double esc, const double* __restrict__ X, double* __restrict__ Yf);
extern __global__ void kernel_escalarSuma2Mult(const double* __restrict__ Y0, const double esc1, const double* __restrict__ X, const double esc2, const double* __restrict__ Z, double* __restrict__ Yf);
extern __global__ void kernel_sumatoriaRK4(double* __restrict__ Yn, const double* __restrict__ K1, const double* __restrict__ K2, const double* __restrict__ K3, const double* __restrict__ K4);
extern __global__ void kernel_sumatoriaRK3(double* __restrict__ Yn, const double* __restrict__ K1, const double* __restrict__ K2, const double* __restrict__ K3);
extern __global__ void kernel_sumatoriaRK2(double* __restrict__ Yn, const double* __restrict__ K1, const double* __restrict__ K2);

void RungeKutta_graph::updateConstants(const int &neqn, const double &h) const {
    // Esta clase no tiene kernels (ni constantes propias por tanto) y llama al RK original
    ptr_runge->updateConstants(neqn, h);
}

void RungeKutta_graph::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    // Buffers en device
    double *Yn, *K1, *K2, *K3, *K4, *Yaux;
    cudaMalloc(&Yn,   this->bytes); // Vector con la Y en cada iteracion
    cudaMalloc(&K1,   this->bytes); // Vectores de cada paso
    cudaMalloc(&K2,   this->bytes);
    cudaMalloc(&K3,   this->bytes);
    cudaMalloc(&K4,   this->bytes);
    cudaMalloc(&Yaux, this->bytes); // Vector auxiliar

    // Constantes para los kernel, tanto de RK original como del problema
    this->updateConstants(neqn, h);
    problema->updateConstants();
    const double h2 = h/2.0, h_n=-1.0*h, h_2=2.0*h;

    cudaMemcpy(Yn, Y0, this->bytes, cudaMemcpyHostToDevice); // Y0 -> Yn, para primera iteración
    
    // Creacion del stream y del graph, y lo capturamos
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    switch(orden)
    {
        case 1:
            problema->feval(0.0, Yn, K1, stream);// K1 = feval(tn, Yn)
            kernel_escalarPorVector<<<num_blocks, tam_blocks, 0, stream>>>(h, K1, Yn);
        break;

        case 2:
            problema->feval(0.0, Yn, K1, stream);// K1 = feval(tn, Yn)
            kernel_escalarSumaMult<<<num_blocks, tam_blocks, 0, stream>>>(Yn, h, K1, Yaux);
            problema->feval(h, Yaux, K2, stream);// K2 = feval(tn + h, Yn + h*K1) 
            kernel_sumatoriaRK2<<<num_blocks, tam_blocks, 0, stream>>>(Yn, K1, K2);
        break;

        case 3:
            problema->feval(0.0, Yn, K1, stream);// K1 = feval(tn, Yn)
            kernel_escalarSumaMult<<<num_blocks, tam_blocks, 0, stream>>>(Yn, h2, K1, Yaux);
            problema->feval(h2, Yaux, K2, stream); // K2 = feval(tn + h/2, Yn + h/2 * K1) 
            kernel_escalarSuma2Mult<<<num_blocks, tam_blocks, 0, stream>>>(Yn, h_n, K1, h_2, K2, Yaux);
            problema->feval(h, Yaux, K3, stream);// K3 = feval(tn + h, Yn -h*K1 + 2*h*K2 ) 
            kernel_sumatoriaRK3<<<num_blocks, tam_blocks, 0, stream>>>(Yn, K1, K2, K3);
        break;

        case 4:
            problema->feval(0.0, Yn, K1, stream);// K1 = feval(tn, Yn)
            kernel_escalarSumaMult<<<num_blocks, tam_blocks, 0, stream>>>(Yn, h2, K1, Yaux);
            problema->feval(h2, Yaux, K2, stream);// K2 = feval(tn + h/2, Yn + K1*h/2)
            kernel_escalarSumaMult<<<num_blocks, tam_blocks, 0, stream>>>(Yn, h2, K2, Yaux);
            problema->feval(h2, Yaux, K3, stream);// K3 = feval(tn + h/2, Yn + K2*h/2)
            kernel_escalarSumaMult<<<num_blocks, tam_blocks, 0, stream>>>(Yn, h, K3, Yaux);
            problema->feval(h, Yaux, K4, stream);// K4 = feval(tn + h, Y0 + h*K3)
            kernel_sumatoriaRK4<<<num_blocks, tam_blocks, 0, stream>>>(Yn, K1, K2, K3, K4);
        break;
    }

    // Finalizamos la captura del graph
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    // Aqui tenemos el bucle principal en el que lanzamos el graph
    for (double tn = t0; tn < tf; tn += h) {
        // Usamos memoria constante en device (c_t) y actualizarla antes del launch.
        cudaMemcpyToSymbolAsync(problema->get_t(), &tn, sizeof(double), 0, cudaMemcpyHostToDevice, stream);
        cudaGraphLaunch(instance, stream);
    }

    cudaStreamSynchronize(stream);
    cudaMemcpy(Yf, Yn, bytes, cudaMemcpyDeviceToHost);

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    cudaFree(Yn);
    cudaFree(K1);
    cudaFree(K2);
    cudaFree(K3);
    cudaFree(K4);
    cudaFree(Yaux);
}


#endif