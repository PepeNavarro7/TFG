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

extern __constant__ Params_advdiff1d cte_avd1; // Estructura de datos constantes para los kernel
extern __constant__ double cte_avd1_t; // Constante que utilizará el graph

// Class for the IVP-ODE representing a 1D Advection-Diffusion model 
class advdiff1d: public Problema {

private:
    const int nx;           // number of grid points at each dimension
    const double dtx_sq,    // Spatial step squared
                 dtx_4,     // Spatial step times 4
                 a=10.0,    // constant scalar representing the strength of advection 
                 d=1.0;     // constant scalar representing the strength of diffusion
            
public:
    // Constructor of the class IVP_ODE_advdiff1d    
    advdiff1d(const int &nx_points, const int &threads):
        Problema(nx_points, "1D_Advection-Diffusion", (1.0/nx_points), dim3( (nx_points+threads-1)/threads, 1, 1 ), dim3(threads,1,1)),
        nx(nx_points), dtx_sq(get_dtx()*get_dtx()), dtx_4(4.0*get_dtx()) { };

    // Definicion de los valores constantes para el kernel
    virtual void updateConstants() const override;
    
    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 
    
    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    virtual void feval (const double &t, const double *Y, double *DY) const override; 
    virtual void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;
    
    // Exportar los datos a un archivo txt
    inline void archivo (const string &filename, const double *Y) const override { archivo1(filename,Y); };

    virtual inline const void* get_t() const { return (const void*)&cte_avd1_t; }

protected:
    // Constructor para clase hija 
    advdiff1d(const int &nx_points, const int &threads, const string &name):
        Problema(nx_points, name, (1.0/nx_points), dim3( (nx_points+threads-1)/threads, 1, 1 ), dim3(threads,1,1)),
        nx(nx_points), dtx_sq(get_dtx()*get_dtx()), dtx_4(4.0*get_dtx()) { };   

public:
    inline int get_nx() const { return nx; };
    inline double get_dtx_sq() const { return dtx_sq; };
    inline double get_dtx_4() const { return dtx_4; };
    inline double get_a() const { return a; };
    inline double get_d() const { return d; };
};
#endif