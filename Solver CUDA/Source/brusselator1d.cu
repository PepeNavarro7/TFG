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
__global__ void d_feval (const double &t, const double *Y, double *DY, const int &nx, const double &dtx){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const double alpha=1.0/50.0, A=1.0, B=3.0;
    const double DD = alpha/(dtx*dtx);
    const double C[2]={A,B};
    if(i>=1 && i<=nx-2) {
        for (int j = 0; j < 2; j++) {
            const int ij = i*2 + j; 
            DY[ij] = DD * (Y[ij+2] - 2.0 * Y[ij] + Y[ij-2]);
        }
    } else if (i==0){
        for (int j = 0; j < 2; j++) {
            const int first = i*2 + j; 
            DY[first] = DD * (Y[first+2] - 2.0 * Y[first] + C[j]);
        }
    } else if (i==nx-1){
        for (int j = 0; j < 2; j++) {
            const int last=i*2+j; 
            DY[last]  = DD * ( C[j]- 2.0 * Y[last] + Y[last-2] );
        } 
    }

    

    if(i>=0 && i<=nx-1) {
        const int i0=i*2+0, i1=i*2+1;
        const double ui = Y[i0];
        const double vi = Y[i1];
        const double u2v=ui*ui*vi;
        DY[i0] += A+u2v-(B+1)*ui;
        DY[i1] += B*ui-u2v; 
        //DY[i0] = A+ ui*vi-(B+1)*ui;
        //DY[i1] = B*ui+vi; 
    }
}

void brusselator1d::feval (const double &t, const double *Y, double *DY){
    d_feval<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t,Y,DY,nx,dtx);
}
  
#endif   