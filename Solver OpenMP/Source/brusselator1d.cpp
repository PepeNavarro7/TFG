#ifndef BRUSSELATOR1D_CPP
#define BRUSSELATOR1D_CPP

#include "brusselator1d.h"
#include <cmath>

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 

void brusselator1d::init(double* __restrict Y0) const { 
    for (int i=0;i<nx;i++){  
        double x_i=(double)(i+1)*get_dtx();
        Y0[idx(i,0)]=A+sin(2*get_PI()*x_i);
        Y0[idx(i,1)]=B;
    }  
}

//vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
void brusselator1d::feval (const double &t, const double* __restrict Y, double* __restrict DY) const {
    const double C[2]={A,B};
    
    #pragma omp parallel default(none) shared(Y, DY, nx, C, A, B, DD)
    {
        // Hacemos uso del nowait y collapse ya que no hay dependencia de datos
        #pragma omp for schedule(static) collapse(2) nowait
        for (int i = 1; i < nx-1; i++) {
            for (int j = 0; j < 2; j++) {
                const int ij = idx(i,j); 
                DY[ij] = DD * (Y[ij+2] - 2.0*Y[ij] + Y[ij-2]);
            }
        } // Nos saltamos la barrera
        #pragma omp single nowait 
        {
            for (int j = 0; j < 2; j++) {
                const int first = idx(0,j); 
                DY[first] = DD * (Y[first+2] - 2.0*Y[first] + C[j]);
            } 
        } // Nos saltamos la barrera
        #pragma omp single
        {
            for (int j = 0; j < 2; j++) {
                const int last=idx(nx-1,j); 
                DY[last]  = DD * (C[j]- 2.0*Y[last] + Y[last-2]);
            } 
        }// Barrera implicita

        #pragma omp for schedule(static)
        for (int i = 0; i < nx; i++) {
            const int i0=idx(i,0), i1=idx(i,1);
            const double ui = Y[i0], vi = Y[i1];
            const double u2v=ui*ui*vi;
            DY[i0] += A+u2v-(B+1)*ui;
            DY[i1] += B*ui-u2v; 
            //DY[i0] = A+ ui*vi-(B+1)*ui;
            //DY[i1] = B*ui+vi; 
        } // Barrera implicita
    }
    
}

double brusselator1d::feval_i (const double &t, const double* __restrict Y, const int &i) const {
    double res=0;
    const double C[2]={A,B};
    const int id_x = i/2,
              id_z = i%2;

    if (id_x >= 1 && id_x <= nx-2) {
        res = DD * (Y[i+2] - 2.0*Y[i] + Y[i-2]);
    } else if(id_x == 0) { // Primera fila
        res = DD * (Y[i+2] - 2.0*Y[i] + C[id_z]);
    } else if(id_x == nx-1) { // Última fila
        res = DD * (C[id_z]- 2.0*Y[i] + Y[i-2]);
    }

    const double ui = (id_z==0) ?  (Y[i]) : (Y[i-1]),
                 vi = (id_z==0) ? (Y[i+1]) : (Y[i]);
    const double u2v=ui*ui*vi;
    res += (id_z==0) ? (A+u2v-(B+1)*ui) : (B*ui-u2v);
    return res;
}
  
#endif   