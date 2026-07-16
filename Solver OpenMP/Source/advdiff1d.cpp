#ifndef ADVDIFF1D_CPP
#define ADVDIFF1D_CPP

#include "advdiff1d.h"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <string>

using namespace std;

// Initialize stage vector Y0 with neqn components
void advdiff1d::init(double* __restrict Y0) const {
    for (int i=0; i<nx; ++i) { 
        double x_i = (double)(i+1)*get_dtx();
        Y0[i] = sin(2.0*get_PI()*x_i);
    }
}

// Vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void advdiff1d::feval (const double &t, const double* __restrict Y, double* __restrict DY) const {
    // Compute partially DY in inner points
    #pragma omp parallel default(none) shared(t, Y, DY, a, d, dtx_sq_inv, dtx_4_inv, nx)
    {
        #pragma omp for nowait schedule(static)
        for(int i=1; i<nx-1; i++){
            DY[i] = d * (Y[i+1]    - 2*Y[i]   + Y[i-1]) * dtx_sq_inv
                - a * (Y[i+1]*Y[i+1] - Y[i-1]*Y[i-1]) * dtx_4_inv;
        } // Nos saltamos la barrera
        #pragma omp single nowait // Compute partially DY in boundary points (i=0 and i=nx-1)
        {
            DY[0]   = d * (Y[1]    - 2*Y[0]  + Y[nx-1]) * dtx_sq_inv
                    - a * (Y[1]*Y[1] - Y[nx-1]*Y[nx-1]) * dtx_4_inv;
        } // Nos saltamos la barrera
        #pragma omp single
        {
            DY[nx-1]= d * (Y[0] - 2*Y[nx-1]  + Y[nx-2]) * dtx_sq_inv
                    - a * (Y[0]*Y[0] - Y[nx-2]*Y[nx-2]) * dtx_4_inv;
        } // Barrera implicita
        
        // Complete the computation of DY
        #pragma omp for schedule(static)
        for(int i=0;i<nx;i++){   
            DY[i] += Y[i] + f((i+1)*get_dtx(),t);
        }
    }
    
}

// Adaptación de la función feval para aplicarse a un único término
double advdiff1d::feval_i (const double &t, const double* __restrict Y, const int &i) const {
    double res=0;
    
    if(i>=1 && i<=nx-2){
        res = d * (Y[i+1]    - 2*Y[i]   + Y[i-1]) * dtx_sq_inv
            - a * (Y[i+1]*Y[i+1] - Y[i-1]*Y[i-1]) * dtx_4_inv;
    } else if(i==0){
        res = d * (Y[1]    - 2*Y[0]  + Y[nx-1]) * dtx_sq_inv
            - a * (Y[1]*Y[1] - Y[nx-1]*Y[nx-1]) * dtx_4_inv;
    } else if(i==nx-1){
        res = d * (Y[0] - 2*Y[nx-1]  + Y[nx-2]) * dtx_sq_inv
            - a * (Y[0]*Y[0] - Y[nx-2]*Y[nx-2]) * dtx_4_inv;
    } 
    
    res += Y[i] + f((i+1)*get_dtx(),t);
    
    return res;
}

double advdiff1d::f(const double &x, const double &t) const{ 
    const double pi2xpt=2.0*get_PI()*x + t;
    const double c=cos(pi2xpt);
    const double s=sin(pi2xpt); 
    return( c + 2*a*get_PI()* s*c + 4*d*get_PI()*get_PI()*s - s);
}

#endif