#ifndef BRUSSELATOR1D_CPP
#define BRUSSELATOR1D_CPP

#include "brusselator1d.h"
#include <cmath>
#include <iostream>

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 

// Constructor of the class 
brusselator1d::brusselator1d(const int &nx_points){ 
    name = "Brusselator_1D";
    nx = nx_points;
    neqn = 2*nx; // Number of ODEs
    dtx = 1.0/(nx+1.0); // Compute Spatial step
    cout << "dtx =" << dtx << endl;
    dtx_squared = dtx*dtx;
    DD = alpha/dtx_squared; 
 }


void brusselator1d::init(double *Y0) const { 
    cout << "dtx =" << dtx << endl;
    for (int i=0;i<nx;i++){  
        double x_i=(double)(i+1)*dtx;
        Y0[idx(i,0)]=A+sin(2*PI*x_i);
        Y0[idx(i,1)]=B;
    }  
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
__global__ void feval_brusselator1d (const double &t, const double *Y, double *DY, const int &neqn, const double &dtx){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const double alpha=1.0/50.0, A=1.0, B=3.0;
    const double DD = alpha/(dtx*dtx);
    //const double C[2]={A,B};

    // Tenemos i hebras >= neqn; neqn == 2*nx
    if(i>=2 && i<=neqn-3){
        DY[i] = DD * (Y[i+2] - 2.0 * Y[i] + Y[i-2]);
    } else if (i==0){
        DY[i] = DD * (Y[i+2] - 2.0 * Y[i] + A);
    } else if (i==1){
        DY[i] = DD * (Y[i+2] - 2.0 * Y[i] + B);
    } else if (i==neqn-2){
        DY[i] = DD * ( A     - 2.0 * Y[i] + Y[i-2] );
    } else if (i==neqn-1){
        DY[i] = DD * ( B     - 2.0 * Y[i] + Y[i-2] );
    }

    if(i>=0 && i<=neqn-1){
        if(i%2==0){ // pares == i0
            const double ui = Y[i], vi=Y[i+1];
            const double u2v=ui*ui*vi;
            DY[i] += A+u2v-(B+1)*ui;
        } else{ // impares == i1
            const double vi = Y[i], ui=Y[i-1];
            const double u2v=ui*ui*vi;
            DY[i] += B*ui-u2v; 
        }
    }
}

void brusselator1d::feval (const double &t, const double *Y, double *DY) const {
    feval_brusselator1d<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t,Y,DY,neqn,dtx);
}
  
#endif   