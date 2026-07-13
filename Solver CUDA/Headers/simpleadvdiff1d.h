#ifndef SIMPLE_AVD_DIFF_H
#define SIMPLE_AVD_DIFF_H

#include "Problema.h"

using namespace std;

// PROBLEMA 1
struct Params_simpleadvdiff1d {
    int neqn;
    double dtx_2_inv;
    double dtx_sq_inv;
    double a;
    double d;
};

extern __constant__ Params_simpleadvdiff1d cte_savd; // Estructura de datos constantes para los kernel
extern __constant__ double cte_savd_t; // Constante que utilizará el graph

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
        nx(nx_points), dtx_2(2.0*get_dtx()), dtx_sq(get_dtx()*get_dtx()) { };

    // Definicion de los valores constantes para el kernel
    virtual void updateConstants() const override;
    
    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 

    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    virtual void feval (const double &t, const double *Y, double *DY) const override;
    virtual void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;

    inline void archivo (const string &filename, const double *Y) const override { archivo1(filename,Y); };

    virtual inline const void* get_t() const { return (const void*)&cte_savd_t; }
protected:
    simpleadvdiff1d(const int &nx_points, const int &threads, const string &name):
        Problema(nx_points, name, (1.0/nx_points), dim3( (nx_points+threads-1)/threads,1,1 ), dim3(threads,1,1)),
        nx(nx_points), dtx_2(2.0*get_dtx()), dtx_sq(get_dtx()*get_dtx()) { };
public:
    inline int get_nx () const { return nx; };
    inline double get_dtx_2 () const { return dtx_2; };
    inline double get_dtx_sq () const { return dtx_sq; };
    inline double get_a () const { return a; };
    inline double get_d () const { return d; };
};
#endif