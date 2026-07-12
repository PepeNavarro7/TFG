#ifndef BRUSSELATOR1D_SHUFFLE_H
#define BRUSSELATOR1D_SHUFFLE_H

#include "brusselator1d.h"

using namespace std;

extern __constant__ double cte_br1ds_t; // Constante que utilizará el graph
extern __constant__ Params_brusselator1d cte_br1ds; // Estructura de datos constantes para los kernel

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 
class brusselator1d_shuffle:public brusselator1d{
public:
    // Constructor of the class 
    brusselator1d_shuffle(const int &nx_points, const int &threads):
        brusselator1d(nx_points, threads, "Brusselator_1D shuffle", dim3(threads,1,1)){};

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;
        
    //vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override; 
    void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;

    inline const void* get_t() const override { return (const void*)&cte_br1ds_t; }
};
  
#endif 
  
  