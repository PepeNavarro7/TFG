#ifndef BRUSSELATOR2D_CU
#define BRUSSELATOR2D_CU

#include "brusselator2d.h"
#include <cmath>

using namespace std;

// Problema 4

// Variables en memoria constante (viven en la GPU)
__constant__ Params_brusselator2d cte_br2d;
__constant__ double cte_br2d_t;

// Definicion de los valores constantes para el kernel
void brusselator2d::updateConstants() const {
    Params_brusselator2d aux;

    aux.neqn = get_num_ODEs();
    aux.nx = nx;
    aux.A = A;
    aux.B = B;
    aux.dtx = get_dtx();
    aux.DD = DD;

    cudaMemcpyToSymbol(cte_br2d, &aux, sizeof(Params_brusselator2d));
}

void brusselator2d::init(double* Y0) const {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x_i = (double)(i + 1) * get_dtx();
            double y_i = (double)(j + 1) * get_dtx();
            Y0[idx(i, j, 0)] = 22 * y_i * pow(1 - y_i, 1.5);
            Y0[idx(i, j, 1)] = 27 * x_i * pow(1 - x_i, 1.5);
        }
    }
}

// Kernel para paralelizar con CUDA el feval del problema
__global__ void kernel_brusselator2d(const double t, const double* __restrict__ Y, double* __restrict__ DY) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int tam_fila = cte_br2d.nx*2, 
        ult = cte_br2d.nx-1;
    const int id_x = thread / tam_fila, // thread == id_x * 2 * nx + id_y * 2 + id_z  
        id_y = (thread - id_x*tam_fila) / 2,
        id_z = thread % 2;

    if(thread < cte_br2d.neqn){
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
        const double term = cte_br2d.B * u_ij - u_ij * u_ij * v_ij;

        DY[thread] = cte_br2d.DD * (val_arriba + val_izquierda - 4.0 * valor + val_abajo + val_derecha);
        
        const double x = (id_x + 1) * cte_br2d.dtx, y = (id_y + 1) * cte_br2d.dtx;
        const double xmxc = x - 0.3, ymyc = y - 0.5;
        const double r = 0.1;
        const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;

        DY[thread]+= id_z==0 ? (cte_br2d.A - term - u_ij + result) : term;  
    }
}

__global__ void graph_brusselator2d(const double offset, const double* __restrict__ Y, double* __restrict__ DY) {
    double t = cte_br2d_t + offset;

    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int tam_fila = cte_br2d.nx*2, 
        ult = cte_br2d.nx-1;
    const int id_x = thread / tam_fila, // thread == id_x * 2 * nx + id_y * 2 + id_z  
        id_y = (thread - id_x*tam_fila) / 2,
        id_z = thread % 2;

    if(thread < cte_br2d.neqn){
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
        const double term = cte_br2d.B * u_ij - u_ij * u_ij * v_ij;

        DY[thread] = cte_br2d.DD * (val_arriba + val_izquierda - 4.0 * valor + val_abajo + val_derecha);

        const double x = (id_x + 1) * cte_br2d.dtx, y = (id_y + 1) * cte_br2d.dtx;
        const double xmxc = x - 0.3, ymyc = y - 0.5;
        const double r = 0.1;
        const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;

        DY[thread]+= id_z==0 ? (cte_br2d.A - term - u_ij + result) : term;
    }
}

void brusselator2d::feval(const double &t, const double* Y, double* DY) const {
    kernel_brusselator2d<<<get_grid(),get_block()>>>(t, Y, DY);
}
void brusselator2d::feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const {
    graph_brusselator2d<<<get_grid(),get_block(), 0, stream>>>(offset, Y, DY);
}

#endif