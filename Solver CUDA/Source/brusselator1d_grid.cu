#ifndef BRUSSELATOR1D_GRID_CU
#define BRUSSELATOR1D_GRID_CU

#include "brusselator1d_grid.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 
struct Params_brusselator1d_grid {
    int neqn;
    int nx;
    int ny;
    double A;
    double B;
    double DD;
};

// Variable en memoria constante (vive en la GPU)
__constant__ Params_brusselator1d_grid cte3_g;

// Definicion de los valores constantes para el kernel
void brusselator1d_grid::updateConstants() const {
    Params_brusselator1d_grid aux;

    aux.neqn = this->neqn;
    aux.nx = this->nx;
    aux.ny = this->ny;
    aux.A = this->A;
    aux.B = this->B;
    aux.DD = this->DD;

    cudaMemcpyToSymbol(cte3_g, &aux, sizeof(Params_brusselator1d_grid));
}

// Modificamos la inicialización para que se corresponda con los kernel posteriores, tal que
// En lugar de tener un Array of Structures -> [0u, 0v, 1u, 1v...n-1u, n-1v]
// ahora tendremos un Structure of Arrays -> [0u, 1u...n-1u, 0v, 1v...n-1v]
void brusselator1d_grid::init(double *Y0) const { 
    for(int z=0; z<2; ++z){
        for (int i=0;i<nx;i++){  
            double x_i=(double)(i+1)*dtx;
            Y0[z*nx+i]= z==0 ? A+sin(2*PI*x_i) : B;
        }  
    }
    
}

// Con esta ordenación, los u->z==0, y los v->z==1
__global__ void kernel_brusselator1d_grid (const double t, const double* __restrict__ Y, double* __restrict__ DY){
    const int id_x = blockIdx.x * blockDim.x + threadIdx.x, // Identificamos coordenada x
              id_z = threadIdx.y,  // Coordenada z
              lane = threadIdx.x & 31,
              ult = cte3_g.nx-1;
    const int thread = id_z*cte3_g.nx + id_x; // thread
    

    if(id_x < cte3_g.nx && id_z < 2 << thread<cte3_g.neqn){
        const double valor = Y[thread];
        const unsigned mask = 0xFFFFFFFF;
        const double C = id_z==0 ? cte3_g.A : cte3_g.B;

        // Hacemos los shuffles de los vecinos
        double val_izq  = __shfl_up_sync  (mask, valor, 1), // th(n-1) -> th(n)
               val_der  = __shfl_down_sync(mask, valor, 1); // th(n) <- th(n+1)

        // Reescribimos los shuffles que no son correctos
        if(lane==0 || id_x==0) // lane 0 no puede shuffle, x==0 es frontera
            val_izq = id_x==0 ? C : Y[thread-1];
        if(lane==31 || id_x==ult) // lane 31 no puede shuffle, x==ult es frontera
            val_der = id_x==ult ? C : Y[thread+1];

        const double val_pareja = id_z==0 ? Y[thread+cte3_g.nx] : Y[thread-cte3_g.nx];

        DY[thread] = cte3_g.DD * (val_der - 2.0 * valor + val_izq);

        const double ui = id_z==0 ? valor : val_pareja,
                     vi = id_z==1 ? valor : val_pareja;
        const double u2v=ui*ui*vi;
        DY[thread] += id_z==0 ? (cte3_g.A + u2v - (cte3_g.B+1)*ui) : (cte3_g.B * ui - u2v);
    }
}

void brusselator1d_grid::feval (const double &t, const double *Y, double *DY) const {
    kernel_brusselator1d_grid<<<this->grid,this->block>>>(t,Y,DY);
}
  
#endif   