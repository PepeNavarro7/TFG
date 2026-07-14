#ifndef ADVDIFF1D_SHUFFLE_H
#define ADVDIFF1D_SHUFFLE_H

#include "advdiff1d.h"

using namespace std;

// PROBLEMA 2

extern __constant__ Params_advdiff1d cte_avd1s; // Estructura de datos constantes para los kernel
extern __constant__ double cte_avd1s_t; // Constante que utilizará el graph

// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class advdiff1d_shuffle: public advdiff1d {
            
public:
    // Constructor of the class IVP_ODE_advdiff1d    
    advdiff1d_shuffle(const int &nx_points, const int &threads):
        advdiff1d(nx_points, threads, "1D_Advection-Diffusion shuffle") { };

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;
    
    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override; 
    void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;

    inline const void* get_t() const { return (const void*)&cte_avd1s_t; }
};
#endif