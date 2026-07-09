#ifndef BRUSSELATOR2D_CU
#define BRUSSELATOR2D_CU

#include "brusselator2d.h"
#include <cmath>

using namespace std;

// Problema 4

// Variable en memoria constante (vive en la GPU)
__constant__ Params_brusselator2d cte4;

// Definicion de los valores constantes para el kernel
void brusselator2d::updateConstants() const {
    Params_brusselator2d aux;

    aux.neqn = this->neqn;
    aux.nx = this->nx;
    aux.A = this->A;
    aux.B = this->B;
    aux.dtx = this->dtx;
    aux.DD = this->DD;

    cudaMemcpyToSymbol(cte4, &aux, sizeof(Params_brusselator2d));
}

void brusselator2d::init(double* Y0) const {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x_i = (double)(i + 1) * dtx;
            double y_i = (double)(j + 1) * dtx;
            Y0[idx(i, j, 0)] = 22 * y_i * pow(1 - y_i, 1.5);
            Y0[idx(i, j, 1)] = 27 * x_i * pow(1 - x_i, 1.5);
        }
    }
}

// Kernel para paralelizar con CUDA el feval del problema
__global__ void kernel_brusselator2d(const double t, const double* __restrict__ Y, double* __restrict__ DY) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int tam_fila = cte4.nx*2, 
        ult = cte4.nx-1;
    const int id_x = thread / tam_fila, // thread == id_x * 2 * nx + id_y * 2 + id_z  
        id_y = (thread - id_x*tam_fila) / 2,
        id_z = thread % 2;

    if(thread < cte4.neqn){
        // Calculamos los indices de los vecinos    
        const int i_pareja = id_z==0 ? thread+1 : thread-1,
            i_arriba  =  id_x==0  ? (ult*tam_fila  + id_y*2 + id_z) : (thread-tam_fila),
            i_abajo  =  id_x==ult ? (     0        + id_y*2 + id_z) : (thread+tam_fila),
            i_izquierda = id_y==0 ? (id_x*tam_fila +  ult*2 + id_z) : (thread-2),
            i_derecha = id_y==ult ? (id_x*tam_fila +   0    + id_z) : (thread+2);

        // Calculamos los valores de los vecinos          
        const double valor = Y[thread], val_pareja = Y[i_pareja],
            val_arriba=Y[i_arriba], val_abajo=Y[i_abajo], val_izquierda=Y[i_izquierda], val_derecha=Y[i_derecha];

        const double u_ij = id_z==0 ? valor : val_pareja,
                     v_ij = id_z==1 ? valor : val_pareja;
        const double term = cte4.B * u_ij - u_ij * u_ij * v_ij;

        DY[thread] = cte4.DD * (val_arriba + val_izquierda - 4.0 * valor + val_abajo + val_derecha);
        if(id_z == 0){
            const double x = (id_x + 1) * cte4.dtx, y = (id_y + 1) * cte4.dtx;
            const double xmxc = x - 0.3, ymyc = y - 0.5;
            const double r = 0.1;
            const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;
            DY[thread]+= cte4.A - term - u_ij + result;
        } else { // id_z==1
            DY[thread]+= term;
        }   
    }
}

// Version del Kernel en la que usamos shuffle
__global__ void kernel2_brusselator2d(const double t, const double* __restrict__ Y, double* __restrict__ DY) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x, // thread == id_x * 2 * nx + id_y * 2 + id_z  
              tam_fila = cte4.nx*2, 
              ult = cte4.nx-1;
    const int id_x = thread / tam_fila, // Identificamos coordenada x
              id_y = (thread - id_x*tam_fila) >> 1, // Coordenada Y -> / 2
              id_z = threadIdx.x & 1, // Coordenada z -> %2
              lane = threadIdx.x & 31; // Posicion en el warp -> th%32

    if(thread < cte4.neqn){
        // Calculamos los indices y valores de los vecinos verticales
        const int i_arriba  =  id_x==0  ? (ult*tam_fila  + id_y*2 + id_z) : (thread-tam_fila),
                  i_abajo   = id_x==ult ? (     0        + id_y*2 + id_z) : (thread+tam_fila);
        const double valor = Y[thread], val_arriba=Y[i_arriba], val_abajo=Y[i_abajo];
        const unsigned mask = 0xFFFFFFFF;

        // Calculamos los shuffles para los vecinos horizontales
        double val_izq  = __shfl_up_sync  (mask, valor, 2), // th(n-2) -> th(n)
               val_der  = __shfl_down_sync(mask, valor, 2), // th(n) <- th(n+2) 
               par_up   = __shfl_up_sync  (mask, valor, 1), // th(n-1) -> th(n)
               par_down = __shfl_down_sync(mask, valor, 1); // th(n) <- th(n+1) 

        // Reescribimos los shuffles que no son correctos
        if(lane<=1 || id_y==0) // lanes 0 y 1 shuffle incorrecto, id_y==0 es frontera horizontal
            val_izq = id_y==0   ? Y[id_x*tam_fila +  ult*2 + id_z] : Y[thread-2];
        if(lane>=30 || id_y==ult) // lanes 30 y 31 shuffle incorrecto, id_y==ult es frontera horizontal
            val_der = id_y==ult ? Y[id_x*tam_fila +    0   + id_z] : Y[thread+2];
        const double val_pareja = id_z==0 ? par_down : par_up; // Nos quedamos el valor pareja apropiado

        const double u_ij = id_z==0 ? valor : val_pareja,
                     v_ij = id_z==1 ? valor : val_pareja;
        const double term = cte4.B * u_ij - u_ij * u_ij * v_ij;

        DY[thread] = cte4.DD * (val_arriba + val_izq - 4.0 * valor + val_abajo + val_der);
        
        const double x = (id_x + 1) * cte4.dtx, y = (id_y + 1) * cte4.dtx;
        const double xmxc = x - 0.3, ymyc = y - 0.5;
        const double r = 0.1;
        const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;

        DY[thread] += id_z==0 ? (cte4.A - term - u_ij + result) : term;
    }
}

void brusselator2d::feval(const double &t, const double* Y, double* DY) const {
    kernel2_brusselator2d<<<this->grid,this->block>>>(t, Y, DY);
}
void brusselator2d::feval (const double *Y, double* DY, cudaStream_t stream) const {
    kernel2_brusselator2d<<<this->grid, this->block, 0, stream>>>(0.0, Y, DY);
}
/* // Auxiliary function f
double brusselator2d::f(const int &i, const int &j, const double &t) const {
    const double x = (i + 1) * dtx, y = (j + 1) * dtx;
    const double xmxc = x - 0.3, ymyc = y - 0.5;
    const double r = 0.1;
    const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;
    return result;
}*/

#endif