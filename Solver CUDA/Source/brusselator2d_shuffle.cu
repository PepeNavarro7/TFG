#ifndef BRUSSELATOR2D_SHUFFLE_CU
#define BRUSSELATOR2D_SHUFFLE_CU

#include "brusselator2d_shuffle.h"
#include <cmath>

using namespace std;

// Problema 4

// Variables en memoria constante (viven en la GPU)
__constant__ Params_brusselator2d cte_br2ds; // Estructura de datos constantes para los kernel
__constant__ double cte_br2ds_t; // Constante que utilizará el graph

// Definicion de los valores constantes para el kernel
void brusselator2d_shuffle::updateConstants() const {
    Params_brusselator2d aux;

    aux.neqn = get_neqn();
    aux.nx = get_nx();
    aux.A = get_A();
    aux.B = get_B();
    aux.dtx = get_dtx();
    aux.DD = get_DD();

    cudaMemcpyToSymbol(cte_br2ds, &aux, sizeof(Params_brusselator2d));
}

// Version del Kernel en la que usamos shuffle
__global__ void kernel_brusselator2d_shuffle(const double t, const double* __restrict__ Y, double* __restrict__ DY) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x, // thread == id_x * 2 * nx + id_y * 2 + id_z  
              tam_fila = cte_br2ds.nx*2, 
              ult = cte_br2ds.nx-1;
    const int id_x = thread / tam_fila, // Identificamos coordenada x
              id_y = (thread - id_x*tam_fila) >> 1, // Coordenada Y -> / 2
              id_z = threadIdx.x & 1, // Coordenada z -> %2
              lane = threadIdx.x & 31; // Posicion en el warp -> th%32

    if(thread < cte_br2ds.neqn){
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
        const double term = cte_br2ds.B * u_ij - u_ij * u_ij * v_ij;

        DY[thread] = cte_br2ds.DD * (val_arriba + val_izq - 4.0 * valor + val_abajo + val_der);
        
        const double x = (id_x + 1) * cte_br2ds.dtx, y = (id_y + 1) * cte_br2ds.dtx;
        const double xmxc = x - 0.3, ymyc = y - 0.5;
        const double r = 0.1;
        const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;

        DY[thread] += id_z==0 ? (cte_br2ds.A - term - u_ij + result) : term;
    }
}

__global__ void graph_brusselator2d_shuffle(const double offset, const double* __restrict__ Y, double* __restrict__ DY) {
    const double t = cte_br2ds_t + offset;

    const int thread = blockDim.x * blockIdx.x + threadIdx.x, // thread == id_x * 2 * nx + id_y * 2 + id_z  
              tam_fila = cte_br2ds.nx*2, 
              ult = cte_br2ds.nx-1;
    const int id_x = thread / tam_fila, // Identificamos coordenada x
              id_y = (thread - id_x*tam_fila) >> 1, // Coordenada Y -> / 2
              id_z = threadIdx.x & 1, // Coordenada z -> %2
              lane = threadIdx.x & 31; // Posicion en el warp -> th%32

    if(thread < cte_br2ds.neqn){
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
        const double term = cte_br2ds.B * u_ij - u_ij * u_ij * v_ij;

        DY[thread] = cte_br2ds.DD * (val_arriba + val_izq - 4.0 * valor + val_abajo + val_der);
        
        const double x = (id_x + 1) * cte_br2ds.dtx, y = (id_y + 1) * cte_br2ds.dtx;
        const double xmxc = x - 0.3, ymyc = y - 0.5;
        const double r = 0.1;
        const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;

        DY[thread] += id_z==0 ? (cte_br2ds.A - term - u_ij + result) : term;
    }
}

void brusselator2d_shuffle::feval(const double &t, const double* Y, double* DY) const {
    kernel_brusselator2d_shuffle<<<get_grid(),get_block()>>>(t, Y, DY);
}
void brusselator2d_shuffle::feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const {
    graph_brusselator2d_shuffle<<<get_grid(),get_block(), 0, stream>>>(offset, Y, DY);
}

#endif