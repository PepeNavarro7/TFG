#ifndef BRUSSELATOR2D_SHUFFLE_H
#define BRUSSELATOR2D_SHUFFLE_H

#include "brusselator2d.h"
#include <string>

using namespace std;

extern __constant__ Params_brusselator2d cte_br2ds; // Estructura de datos constantes para los kernel
extern __constant__ double cte_br2ds_t; // Constante que utilizará el graph

// Problema 4
// the Brusselator 2D model 
class brusselator2d_shuffle: public brusselator2d {
public:
    // Constructor of the class 
    brusselator2d_shuffle(const int &nx_points, const int &threads):
        brusselator2d(nx_points, threads, "Brusselator_2D shuffle"){};

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;

    //vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double* Y, double* DY) const override;
    void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;

    inline const void* get_t() const override { return (const void*)&cte_br2ds_t; }
};

#endif