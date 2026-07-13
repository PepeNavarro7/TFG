#ifndef SIMPLE_AVD_DIFF_SHUFFLE_H
#define SIMPLE_AVD_DIFF_SHUFFLE_H

#include "simpleadvdiff1d.h"

using namespace std;

// PROBLEMA 1
extern __constant__ Params_simpleadvdiff1d cte_savds; // Estructura de datos constantes para los kernel
extern __constant__ double cte_savds_t; // Constante que utilizará el graph

// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class simpleadvdiff1d_shuffle: public simpleadvdiff1d {            
public:
    // Constructor of the class
    simpleadvdiff1d_shuffle(const int &nx_points, const int &threads):
        simpleadvdiff1d(nx_points, threads, "1D_Simple Advection-Diffusion shuffle") { };

    virtual void updateConstants() const override;

    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override;
    void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;

    inline const void* get_t() const { return (const void*)&cte_savds_t; }
};
#endif