#ifndef SIMPLE_AVD_DIFF_H
#define SIMPLE_AVD_DIFF_H

#include "Problema.h"

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class simpleadvdiff1d: public Problema {

private:
    const int nx; // number of grid points at each dimension
    const double dtx_2, // Spatial step times 2
                 dtx_sq; // Spatial step squared
    const double a=10.0, // Constant scalar representing the strength of advection
                 d=10.0; // Constant scalar representing the strength of diffusion
            
public:
    // Constructor of the class
    simpleadvdiff1d(const int &nx_points, const int &threads):
        Problema(nx_points, "1D_Simple Advection-Diffusion", (1.0/nx_points), dim3( (nx_points+threads-1)/threads,1,1 ), dim3(threads,1,1)),
        nx(nx_points), dtx_2(2.0*dtx), dtx_sq(dtx*dtx) { 
            updateConstants();
        };

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;
    
    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 

    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override;

    inline void archivo (const string &filename, const double *Y) const override { archivo1(filename,Y); };
};
#endif