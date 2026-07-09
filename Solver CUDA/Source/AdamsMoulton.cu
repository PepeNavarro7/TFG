#ifndef ADAMS_MOULTON_CU
#define ADAMS_MOULTON_CU

#include "AdamsMoulton.h"
#include <iostream>
#include <string>

using namespace std;

extern __global__ void kernel_sumatoriaAB1(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB2(double* __restrict__ Yn2, const double* __restrict__ Yn1, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB3(double* __restrict__ Yn3, const double* __restrict__ Yn2, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);
extern __global__ void kernel_sumatoriaAB4(double* __restrict__ Yn4, const double* __restrict__ Yn3, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0);

// Variable en memoria constante (vive en la GPU)
__constant__ Params_AdamsMoulton cteAM;

// Definicion de los valores constantes para el kernel
void AdamsMoulton::updateConstants(const int &neqn, const double &h) const {
    Params_AdamsMoulton aux;

    aux.neqn = neqn;
    aux.h = h;
    aux.h2 = h/2.0;
    aux.h12 = h/12.0;
    aux.h24 = h/24.0;
    aux.h720 = h/720.0;

    cudaMemcpyToSymbol(cteAM, &aux, sizeof(Params_AdamsMoulton));
}

__global__ void kernel_sumatoriaAM1(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn1){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAM.neqn){ // Yn+1 = Yn + h * Fn+1
        Yn1[thread] = Yn0[thread] + cteAM.h * Fn1[thread];
    }
}

__global__ void kernel_sumatoriaAM2(double* __restrict__ Yn1, const double* __restrict__ Yn0, const double* __restrict__ Fn1, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAM.neqn){ // Yn+1 = Yn + h/2 * (Fn+1 + Fn)
        Yn1[thread] = Yn0[thread] + cteAM.h2 * (Fn1[thread] + Fn0[thread]);
    }
}

__global__ void kernel_sumatoriaAM3(double* __restrict__ Yn2, const double* __restrict__ Yn1, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAM.neqn){ // Yn+1 = Yn + h/12 * (5Fn+1 + 8Fn - 1Fn-1)
        Yn2[thread] = Yn1[thread] + cteAM.h12 * (5.0*Fn2[thread] + 8.0*Fn1[thread] - Fn0[thread]);
    }
}

__global__ void kernel_sumatoriaAM4(double* __restrict__ Yn3, const double* __restrict__ Yn2, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAM.neqn){ // Yn+1 = Yn + h/24 * (9Fn+1 + 19Fn - 5Fn-1 + 1Fn-2)
        Yn3[thread] = Yn2[thread] + cteAM.h24 * (9.0*Fn3[thread] + 19.0*Fn2[thread] - 5.0*Fn1[thread] + Fn0[thread]);
    }
}

__global__ void kernel_sumatoriaAM5(double* __restrict__ Yn4, const double* __restrict__ Yn3, const double* __restrict__ Fn4, const double* __restrict__ Fn3, const double* __restrict__ Fn2, const double* __restrict__ Fn1, const double* __restrict__ Fn0){
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < cteAM.neqn){ // Yn+1 = Yn + h/720 * (251Fn+1 + 646Fn - 264Fn-1 + 106Fn-2 - 19Fn-3)
        Yn4[thread] = Yn3[thread] + cteAM.h720 * (251.0*Fn4[thread] + 646.0*Fn3[thread] - 264.0*Fn2[thread] + 106.0*Fn1[thread] - 19.0*Fn0[thread]);
    }
}

// Aplicar Adams-Moulton el numero necesario de veces
void AdamsMoulton::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0, *Yn1, *Yn2, *Yn3, *Yn4, // Vectores intermedios
        *Fn0, *Fn1, *Fn2, *Fn3, // vectores funcion
        *Y_AB, *F_AB,   // Vector para aproximar Yn usando Adams-Bashford
        *Y_AM, *F_AM;    // Vector para aproximar Yn usando Adams-Moulton
    cudaMalloc((void**)&Yn0,this->bytes);
    cudaMalloc((void**)&Yn1,this->bytes);
    cudaMalloc((void**)&Yn2,this->bytes);
    cudaMalloc((void**)&Yn3,this->bytes);
    cudaMalloc((void**)&Yn4,this->bytes);
    cudaMalloc((void**)&Fn0,this->bytes);
    cudaMalloc((void**)&Fn1,this->bytes);
    cudaMalloc((void**)&Fn2,this->bytes);
    cudaMalloc((void**)&Fn3,this->bytes);
    cudaMalloc((void**)&Y_AB,this->bytes);
    cudaMalloc((void**)&F_AB,this->bytes);
    cudaMalloc((void**)&Y_AM,this->bytes);
    cudaMalloc((void**)&F_AM,this->bytes);
    
    cudaMemcpy(Yn0, Y0, this->bytes, cudaMemcpyHostToDevice); // Definimos Yn0

    // Constantes para los kernels, tanto de los metodos como del problema
    const double h_RK = h/100.0;
    ptr_runge->updateConstants(neqn, h_RK);
    ptr_bashford->updateConstants(neqn, h);
    this->updateConstants(neqn, h);
    problema->updateConstants();

    switch(orden){ // Aplicamos RK4 para obtener los primeros pasos
        case 1:
        case 2: break; // No necesitamos RK en orden 1 & 2
        case 3: // Definimos Yn1
            ptr_runge->aplicarUnidad(problema, t0, h_RK, Yn0, Yn1);
        break;
        case 4: // Definimos Yn1 & Yn2 
            ptr_runge->aplicarUnidad(problema, t0,      h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK, h_RK, Yn1, Yn2);
        break;
        case 5: // Definimos Yn1, Yn2 & Yn3 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
            ptr_runge->aplicarUnidad(problema, t0+h_RK*2.0, h_RK, Yn2, Yn3);
        break;
    } // Fin del switch de RK

    switch(orden){ // Arrancamos los feval
        case 1:
        case 2:// Definimos el vector feval Fn0
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

    switch(orden){
        case 1: // Yn+1 = Yn + h * Fn+1
            for (double tn=t0; tn<tf; tn+=h){
                // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn1 -> Y_AB
                kernel_sumatoriaAB1<<<this->num_blocks,this->tam_blocks>>>(Y_AB, Yn0, Fn0); // Y_AB = Yn0 + h * Fn0

                // Obtenemos el feval de Y_AB
                problema->feval(tn+h, Y_AB, F_AB); // f(tn1, Y_AB) -> F_AB
                
                // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn1 -> Y_AM
                kernel_sumatoriaAM1<<<this->num_blocks,this->tam_blocks>>>(Y_AM, Yn0, F_AB);// Y_AM = Yn0 + h * F_AB

                // Obtenemos el feval de Y_AM
                problema->feval(tn+h, Y_AM, F_AM); // f(tn1, Y_AM) -> F_AM

                // Ahora que tenemos una 2ª aproximación de Yn1, volvemos a aplicar A-M con ella, para obtener el Yn1 definitivo
                kernel_sumatoriaAM1<<<this->num_blocks,this->tam_blocks>>>(Yn1, Yn0, F_AM);

                // Ahora que tenemos Yn1, convertimos los Yn en Yn-1 para hacer la siguiente iteracion
                swap(Yn1, Yn0); // Yn1 -> Yn0
                // Tras los cambios, Yn1, Fn0, Y_AB, F_AB, Y_AM & F_AM contienen basura y serán reescritos

                // Calculamos el nuevo Fn0 usando el Yn1 recién creado (que ahora es Yn0)
                problema->feval(tn+h, Yn0, Fn0); // f(tn1,Yn0') -> Fn0'
            } // Fin del bucle for principal
        break;
        case 2: // Yn+1 = Yn + h/2 * (Fn+1 + Fn)
            for (double tn=t0; tn<tf; tn+=h){
                // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn1 -> Y_AB
                kernel_sumatoriaAB1<<<this->num_blocks,this->tam_blocks>>>(Y_AB, Yn0, Fn0); // Y_AB = Yn0 + h * Fn0           
                
                // Obtenemos el feval de Y_AB
                problema->feval(tn+h, Y_AB, F_AB); // f(tn1, Y_AB) -> F_AB
                
                // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn1 -> Y_AM
                kernel_sumatoriaAM2<<<this->num_blocks,this->tam_blocks>>>(Y_AM, Yn0, F_AB, Fn0);// Y_AM = Yn0 + h/2 * (F_AB + Fn0)

                // Obtenemos el feval de Y_AM
                problema->feval(tn+h, Y_AM, F_AM); // f(tn1, Y_AM) -> F_AM

                // Ahora que tenemos una 2ª aproximación de Yn1, volvemos a aplicar A-M con ella, para obtener el Yn1 definitivo
                kernel_sumatoriaAM2<<<this->num_blocks,this->tam_blocks>>>(Yn1, Yn0, F_AM, Fn0); // Yn1 = Yn0 + h/2 * (F_AM + Fn0)

                // Ahora que tenemos Yn1, convertimos los Yn en Yn-1 para hacer la siguiente iteracion
                swap(Yn1, Yn0); // Yn1 -> Yn0
                // Tras los cambios, Yn1, Fn0, Y_AB, F_AB, Y_AM & F_AM contienen basura y serán reescritos

                // Calculamos el nuevo Fn0 usando el Yn1 recién creado (que ahora es Yn0)
                problema->feval(tn+h, Yn0, Fn0); // f(tn1,Yn0') -> Fn0'
            } // Fin del bucle for principal
        break;
        case 3: // Yn+1 = Yn + h/12 * (5Fn+1 + 8Fn - 1Fn-1)
            for (double tn=t0+h_RK; tn<tf; tn+=h){
                //Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn2 -> Y_AB
                kernel_sumatoriaAB2<<<this->num_blocks,this->tam_blocks>>>(Y_AB, Yn1, Fn1, Fn0); // Y_AB = Yn1 + h/2 * (3*Fn1 - Fn0)         
                
                // Obtenemos el feval de Y_AB
                problema->feval(tn+h, Y_AB, F_AB); // f(tn2, Y_AB) -> F_AB
                
                // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn2 -> Y_AM
                kernel_sumatoriaAM3<<<this->num_blocks,this->tam_blocks>>>(Y_AM, Yn1, F_AB, Fn1, Fn0); // Y_AM = Yn1 + h/12 * (5F_AB + 8Fn1 - 1Fn0)

                // Obtenemos el feval de Y_AM
                problema->feval(tn+h, Y_AM, F_AM); // f(tn2, Y_AM) -> F_AM

                // Ahora que tenemos una 2ª aproximación de Yn2, volvemos a aplicar A-M con ella, para obtener el Yn2 definitivo
                kernel_sumatoriaAM3<<<this->num_blocks,this->tam_blocks>>>(Yn2, Yn1, F_AM, Fn1, Fn0);// Yn2 = Yn1 + h/12 * (5Fn_AM + 8Fn1 - 1Fn0)

                // Ahora que tenemos Yn2, convertimos Yn2 en Yn1, y los Fn en Fn-1 para hacer la siguiente iteracion
                swap(Yn2, Yn1); // Yn2 -> Yn1
                swap(Fn1, Fn0); // Fn1 -> Fn0
                // Tras los cambios, Yn2, Fn1, Y_AB, F_AB, Y_AM & F_AM contienen basura y serán reescritos

                // Calculamos el nuevo Fn1 usando el Yn2 recién creado (que ahora es Yn1)
                problema->feval(tn+h, Yn1, Fn1); // f(tn2,Yn1') -> Fn1'
            } // Fin del bucle for principal
        break;
        case 4: // Yn+1 = Yn + h/24 * (9Fn+1 + 19Fn - 5Fn-1 + 1Fn-2)
            for (double tn=t0+h_RK*2.0; tn<tf; tn+=h){
                // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn3 -> Y_AB
                kernel_sumatoriaAB3<<<this->num_blocks,this->tam_blocks>>>(Y_AB, Yn2, Fn2, Fn1, Fn0);// Y_AB = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
                
                // Obtenemos el feval de Y_AB
                problema->feval(tn+h, Y_AB, F_AB); // f(tn3, Y_AB) -> F_AB
                
                // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn3 -> Y_AM
                kernel_sumatoriaAM4<<<this->num_blocks,this->tam_blocks>>>(Y_AM, Yn2, F_AB, Fn2, Fn1, Fn0);// Y_AM = Yn2 + h/24 * (9F_AB + 19Fn2 - 5Fn1 + 1Fn0)

                // Obtenemos el feval de Y_AM
                problema->feval(tn+h, Y_AM, F_AM); // f(tn3, Y_AM) -> F_AM

                // Ahora que tenemos una 2ª aproximación de Yn3, volvemos a aplicar A-M con ella, para obtener el Yn3 definitivo
                kernel_sumatoriaAM4<<<this->num_blocks,this->tam_blocks>>>(Yn3, Yn2, F_AM, Fn2, Fn1, Fn0);// Yn3 = Yn2 + h/24 * (9F_AM + 19Fn2 - 5Fn1 + 1Fn0)

                // Ahora que tenemos Yn3, convertimos Yn3 en Yn2, y los Fn en Fn-1 para hacer la siguiente iteracion
                swap(Yn3, Yn2); // Yn3 -> Yn2
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn2 -> Fn1
                // Tras los cambios, Yn3, Fn2, Y_AB, F_AB, Y_AM & F_AM contienen basura y serán reescritos

                // Calculamos el nuevo Fn2 usando el Yn3 recién creado (que ahora es Yn2)
                problema->feval(tn+h, Yn2, Fn2); // f(tn3,Yn2') -> Fn2'
            } // Fin del bucle for principal
        break;
        case 5: // Yn+1 = Yn + h/720 * (251Fn+1 + 646Fn - 264Fn-1 + 106Fn-2 - 19Fn-3)
            for (double tn=t0+h_RK*3.0; tn<tf; tn+=h){
                // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn4 -> Y_AB
                kernel_sumatoriaAB4<<<this->num_blocks,this->tam_blocks>>>(Y_AB, Yn3, Fn3, Fn2, Fn1, Fn0);// Y_AB = Yn3 + h/24 * (55Fn3 - 59Fn2 + 37Fn1 - 9Fn0)
                
                // Obtenemos el feval de Y_AB
                problema->feval(tn+h, Y_AB, F_AB); // f(tn4, Y_AB) -> F_AB
                
                // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn4 -> Y_AM
                kernel_sumatoriaAM5<<<this->num_blocks,this->tam_blocks>>>(Y_AM, Yn3, F_AB, Fn3, Fn2, Fn1, Fn0); // Y_AM = Yn3 + h/720 * (251F_AB + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)

                // Obtenemos el feval de Y_AM
                problema->feval(tn+h, Y_AM, F_AM); // f(tn4, Y_AM) -> F_AM

                // Ahora que tenemos una 2ª aproximación de Yn4, volvemos a aplicar A-M con ella, para obtener el Yn4 definitivo
                kernel_sumatoriaAM5<<<this->num_blocks,this->tam_blocks>>>(Yn4, Yn3, F_AM, Fn3, Fn2, Fn1, Fn0); // Yn4 = Yn3 + h/720 * (251F_AM + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)

                // Ahora que tenemos Yn4, convertimos todos los Yn en Yn-1, y los Fn en Fn-1 para hacer la siguiente iteracion
                swap(Yn4, Yn3); // Yn4 -> Yn3
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn2 -> Fn1
                swap(Fn3, Fn2); // Fn3 -> Fn2
                // Tras los cambios, Yn4, Fn3, Y_AB, F_AB, Y_AM & F_AM contienen basura y serán reescritos

                // Calculamos el nuevo Fn3 usando el Yn4 recién creado (que ahora es Yn3)
                problema->feval(tn+h, Yn3, Fn3); // f(tn4,Yn3') -> Fn3'
            } // Fin del bucle for principal
        break;        
    } // Fin del switch principal

    switch(orden){ // Arrastre de los valores finales
        case 1:
        case 2:
            cudaMemcpy(Yf, Yn0, this->bytes, cudaMemcpyDeviceToHost);  // Yn0' -> Yf
        break;
        case 3:
            cudaMemcpy(Yf, Yn1, this->bytes, cudaMemcpyDeviceToHost); // Yn1' -> Yf
        break;
        case 4:
            cudaMemcpy(Yf, Yn2, this->bytes, cudaMemcpyDeviceToHost);  // Yn2' -> Yf
        break;
        case 5:
            cudaMemcpy(Yf, Yn3, this->bytes, cudaMemcpyDeviceToHost); // Yn3' -> Yf
        break;
    } // Fin del switch de arrastre

    
    cudaFree(Yn0); cudaFree(Yn1); cudaFree(Yn2); cudaFree(Yn3); cudaFree(Yn4);
    cudaFree(Fn0); cudaFree(Fn1); cudaFree(Fn2); cudaFree(Fn3);
    cudaFree(Y_AB); cudaFree(F_AB);
    cudaFree(Y_AM); cudaFree(F_AM);
}
#endif