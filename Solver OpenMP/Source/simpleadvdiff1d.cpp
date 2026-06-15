#ifndef SIMPLE_AVD_DIFF_CPP
#define SIMPLE_AVD_DIFF_CPP

#include <omp.h>
#include "simpleadvdiff1d.h"
#include <cmath>
#include <iostream>
#include <string>

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
// Initialize stage vector Y0 with neqn components
void simpleadvdiff1d::init(double *Y0) const {
    for (int i=0;i<neqn;i++) { 
        double x_i=(double)(i+1)*dtx;
        Y0[i]=sin(2.0*PI*x_i);
    }
}

// Vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void simpleadvdiff1d::feval(const double &t, const double* Y, double* DY) const {
    const int tot_th = omp_get_num_threads(), num_th = omp_get_thread_num();
    const int cantidad = neqn / tot_th; // Division entre enteros
    const int desde = num_th == 0 ? 1 : cantidad*num_th, // Check por si la primera hebra
              hasta = num_th==(tot_th-1) ? neqn-1 : cantidad*(num_th+1); // Check por si la ultima hebra


    // No hay PRAGMA OMP FOR porque hacemos el reparto a mano
    for(int i = desde; i<hasta; ++i){
        DY[i] = d * (Y[i + 1] - 2 * Y[i] + Y[i - 1]) / dtx_squared
              - a * (Y[i + 1]            - Y[i - 1]) / dtx_doubled;
    }
    #pragma omp single nowait
    {   // Compute partially DY in boundary points (i=0 and i=nx-1)
        DY[0] = d * (Y[1] - 2 * Y[0] + Y[nx-1]) / dtx_squared
              - a * (Y[1]            - Y[nx-1]) / dtx_doubled;
    } // Nos saltamos esta barrera también
    #pragma omp single
    {
        DY[nx - 1] = d * (Y[0] - 2 * Y[nx-1] + Y[nx-2]) / dtx_squared
                   - a * (Y[0]               - Y[nx-2]) / dtx_doubled;
    } // Barrera implícita
}

// Adaptación de la función feval para aplicarse a un único término
double simpleadvdiff1d::feval_i (const double &t, const double *Y, const int &i) const {
    double res;
    // Compute partially DY in inner points
    if (i >= 1 && i <= nx-2){
        res = d * (Y[i + 1] - 2 * Y[i] + Y[i - 1]) / dtx_squared
            - a * (Y[i + 1]            - Y[i - 1]) / dtx_doubled;
    } else if(i==0){
        res = d * (Y[1] - 2 * Y[0] + Y[nx-1]) / dtx_squared
            - a * (Y[1]            - Y[nx-1]) / dtx_doubled;
    } else if(i== nx-1){
        res = d * (Y[0] - 2 * Y[nx-1] + Y[nx-2]) / dtx_squared
            - a * (Y[0]               - Y[nx-2]) / dtx_doubled;
    } 
    return res;
}
#endif