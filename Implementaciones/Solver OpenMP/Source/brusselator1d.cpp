#ifndef BRUSSELATOR1D_CPP
#define BRUSSELATOR1D_CPP

#include "brusselator1d.h"
#include <cmath>

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 

// Constructor of the class 
brusselator1d::brusselator1d(const int &nx_points){ 
    name = "Brusselator_1D";
    nx = nx_points;
    neqn = 2*nx; // Number of ODEs
    dtx = 1.0/(nx+1.0); // Compute Spatial step
    dtx_squared = dtx*dtx;
    DD = alpha/dtx_squared; 
 }


void brusselator1d::init(double *Y0) { 
    for (int i=0;i<nx;i++){  
        double x_i=(double)(i+1)*dtx;
        Y0[idx(i,0)]=A+sin(2*PI*x_i);
        Y0[idx(i,1)]=B;
    }  
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void brusselator1d::feval (const double &t, const double *Y, double *DY){
    const double C[2]={A,B};
    #pragma omp for nowait
    for (int i = 1; i < nx-1; i++) {
        for (int j = 0; j < 2; j++) {
            const int ij = idx(i,j); 
            DY[ij] = DD * (Y[ij+2] - 2.0 * Y[ij] + Y[ij-2]);
        }
    } // Nos saltamos la barrera

    #pragma omp for 
    for (int j = 0; j < 2; j++) {
        const int first = idx(0,j), last=idx(nx-1,j); 
        DY[first] = DD * (Y[first+2] - 2.0 * Y[first] + C[j]);
        DY[last]  = DD * ( C[j]- 2.0 * Y[last] + Y[last-2] );
    } 

    #pragma omp for
    for (int i = 0; i < nx; i++) {
        const int i0=idx(i,0), i1=idx(i,1);
        const double ui = Y[i0];
        const double vi = Y[i1];
        const double u2v=ui*ui*vi;
        DY[i0] += A+u2v-(B+1)*ui;
        DY[i1] += B*ui-u2v; 
        //DY[i0] = A+ ui*vi-(B+1)*ui;
        //DY[i1] = B*ui+vi; 
    }
}
  
#endif   