#ifndef ADVDIFF1D_H
#define ADVDIFF1D_H

#include "Problema.h"

using namespace std;

// PROBLEMA 2
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class advdiff1d: public Problema {

private:
    const int nx; // number of grid points at each dimension
    const double dtx_squared,     // Spatial step squared
           dtx_quad;        // Spatial step times 4
    const double a=10.0,    // Constant scalar representing the strength of advection 
                 d=1.0;     // Constant scalar representing the strength of diffusion
            
public:
    // Constructor of the class IVP_ODE_advdiff1d    
    advdiff1d(const int &nx_points):
        Problema(nx_points, "1D_Advection-Diffusion", 1.0/nx_points),
        nx(nx_points), dtx_squared(dtx*dtx), dtx_quad(4.0*dtx) {};
    
    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override;  

    // Vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override; 

    // Funcion feval para un solo término
    double feval_i (const double &t, const double *Y, const int &i) const override;

    // Exportar los datos a un archivo txt
    inline void archivo (const string &n, const double *Y) const override { archivo1(n,Y); };

private:
    // Auxiliary function f
    double f(const double &x, const double &t) const;
};

#endif