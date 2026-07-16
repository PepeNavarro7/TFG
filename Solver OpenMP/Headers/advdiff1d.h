#ifndef ADVDIFF1D_H
#define ADVDIFF1D_H

#include "Problema.h"

using namespace std;

// PROBLEMA 2
// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class advdiff1d: public Problema {

private:
    const int nx; // number of grid points at each dimension
    const double dtx_sq_inv,     // Spatial step squared
                 dtx_4_inv,        // Spatial step times 4
                 a=10.0,    // Constant scalar representing the strength of advection 
                 d=1.0;     // Constant scalar representing the strength of diffusion
            
public:
    // Constructor of the class IVP_ODE_advdiff1d    
    advdiff1d(const int &nx_points):
        Problema(nx_points, "1D_Advection-Diffusion", 1.0/nx_points),
        nx(nx_points), dtx_sq_inv( 1.0/(get_dtx()*get_dtx()) ), dtx_4_inv( 1.0/(4.0*get_dtx()) ) {};
    
    // Initialize stage vector Y0 with neqn components
    void init(double* __restrict Y0) const override;  

    // Vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double* __restrict Y, double* __restrict DY) const override; 

    // Funcion feval para un solo término
    double feval_i (const double &t, const double* __restrict Y, const int &i) const override;

    // Exportar los datos a un archivo txt
    inline void archivo (const string &n, const double* __restrict Y) const override { archivo1(n,Y); };

private:
    // Auxiliary function f
    double f(const double &x, const double &t) const;
};

#endif