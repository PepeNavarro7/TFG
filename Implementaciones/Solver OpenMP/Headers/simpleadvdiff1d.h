#ifndef SIMPLE_AVD_DIFF_H
#define SIMPLE_AVD_DIFF_H

#include "Problema.h"

using namespace std;

// PROBLEMA 1
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class simpleadvdiff1d: public Problema {

private:
    int nx; // number of grid points at each dimension
    double dtx, // Spatial step, double & square
        dtx_doubled, // Spatial step doubled
        dtx_squared; // Spatial step squared
    const double a=1.0, // Constant scalar representing the strength of advection
        d=1.0;          // Constant scalar representing the strength of diffusion
            
public:
    // Constructor of the class
    simpleadvdiff1d(const int &nx_points); 

    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) override; 

    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) override;

    //inline void archivo (const string &filename, const double *Y, const double &t) override { archivo1(filename,Y,t); };
};
#endif