#ifndef SIMPLE_AVD_DIFF_H
#define SIMPLE_AVD_DIFF_H

#include "Problema.h"

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class simpleadvdiff1d: public Problema {

private:
    const int nx; // number of grid points at each dimension
    const double dtx_doubled, // Spatial step doubled
        dtx_squared; // Spatial step squared
    const double a=10.0, // Constant scalar representing the strength of advection
        d=10.0;          // Constant scalar representing the strength of diffusion
            
public:
    // Constructor of the class
    simpleadvdiff1d(const int &nx_points): 
        Problema(nx_points,"1D_Simple Advection-Diffusion",(1.0/nx_points)), 
        nx(nx_points), dtx_doubled(2.0*dtx), dtx_squared(dtx*dtx) {};

    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 

    // vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override;

    // Funcion feval para un solo término
    double feval_i (const double &t, const double *Y, const int &i) const override;

    // Exportar los datos a un archivo txt
    inline void archivo (const string &filename, const double *Y) const override { archivo1(filename,Y); };
};
#endif