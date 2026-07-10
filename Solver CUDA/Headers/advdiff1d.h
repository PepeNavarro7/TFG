#ifndef ADVDIFF1D_H
#define ADVDIFF1D_H

#include "Problema.h"

using namespace std;

// PROBLEMA 2
struct Params_advdiff1d {
    double PI;
    int neqn;
    double dtx;
    double dtx_sq_inv;
    double dtx_4_inv;
    double a;
    double d;
};

// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class advdiff1d: public Problema {

private:
    const int nx; // number of grid points at each dimension
    const double dtx_sq;     // Spatial step squared
    const double dtx_4;        // Spatial step times 4
    const double a=10.0;    // constant scalar representing the strength of advection 
    const double d=1.0;     // constant scalar representing the strength of diffusion
            
public:
    // Constructor of the class IVP_ODE_advdiff1d    
    advdiff1d(const int &nx_points, const int &threads):
        Problema(nx_points, "1D_Advection-Diffusion", (1.0/nx_points), dim3( (nx_points+threads-1)/threads, 1, 1 ), dim3(threads,1,1)),
        nx(nx_points), dtx_sq(dtx*dtx), dtx_4(4.0*dtx) { };

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;
    
    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 
    
    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override; 
    void feval (const double &h, const double *Y, double* DY, cudaStream_t stream) const override;
    
    // Exportar los datos a un archivo txt
    inline void archivo (const string &filename, const double *Y) const override { archivo1(filename,Y); };
};
#endif