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
void simpleadvdiff1d::init(double* __restrict Y0) const {
    for (int i=0;i<get_neqn();i++) { 
        double x_i=(double)(i+1)*get_dtx();
        Y0[i]=sin(2.0*get_PI()*x_i);
    }
}

// Vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void simpleadvdiff1d::feval(const double &t, const double* __restrict Y, double* __restrict DY) const {
    #pragma omp parallel for schedule(static) default(none) shared(Y, DY, a, d, dtx_sq_inv, dtx_2_inv)
    for(int i = 1; i<nx-1; ++i){
        DY[i] = d * (Y[i+1] - 2.0*Y[i] + Y[i-1]) * dtx_sq_inv
              - a * (Y[i+1]            - Y[i-1]) * dtx_2_inv;
    } // Nos saltamos esta barrera 
    // Compute partially DY in boundary points (i=0 and i=nx-1)
    DY[0]    = d * (Y[1] - 2.0*Y[0]    + Y[nx-1]) * dtx_sq_inv
             - a * (Y[1]               - Y[nx-1]) * dtx_2_inv;
    DY[nx-1] = d * (Y[0] - 2.0*Y[nx-1] + Y[nx-2]) * dtx_sq_inv
             - a * (Y[0]               - Y[nx-2]) * dtx_2_inv;
}

// Adaptación de la función feval para aplicarse a un único término
double simpleadvdiff1d::feval_i (const double &t, const double* __restrict Y, const int &i) const {
    double res;
    // Compute partially DY in inner points
    if (i >= 1 && i <= nx-2){
        res = d * (Y[i+1] - 2.0*Y[i] + Y[i-1])  * dtx_sq_inv
            - a * (Y[i+1]            - Y[i-1])  * dtx_2_inv;
    } else if(i==0){
        res = d * (Y[i+1] - 2.0*Y[i] + Y[nx-1]) * dtx_sq_inv
            - a * (Y[i+1]            - Y[nx-1]) * dtx_2_inv;
    } else if(i== nx-1){
        res = d * (Y[0]   - 2.0*Y[i] + Y[i-1])  * dtx_sq_inv
            - a * (Y[0]              - Y[i-1])  * dtx_2_inv;
    } 
    return res;
}
#endif