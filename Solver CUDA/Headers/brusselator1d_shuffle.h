#ifndef BRUSSELATOR1D_SHUFFLE_H
#define BRUSSELATOR1D_SHUFFLE_H

#include "Problema.h"

using namespace std;

// PROBLEMA 3
struct Params_brusselator1d_shuffle {
    int neqn;
    int nx;
    double A;
    double B;
    double DD;
};

// Class for the IVP-ODE representing the 1D Brusselator model 
class brusselator1d_shuffle:public Problema{

private:
    const double alpha=1.0/50.0, A=1.0, B=3.0; // variables auxiliares para el calculo
    const int nx; // number of grid points at each dimension
    const double dtx_sq; // Spatial step squared
    const double DD;

public:
    // Constructor of the class 
    brusselator1d_shuffle(const int &nx_points, const int &threads):
        Problema(nx_points*2.0, "Brusselator_1D_shuffle", (1.0/(nx_points+1.0)), dim3( (nx_points*2.0+threads-1)/threads, 1, 1 ), dim3(threads,1,1)),
        nx(nx_points), dtx_sq(dtx*dtx), DD(alpha/(dtx*dtx)) { };

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;

    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 
    
    //vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override; 
    void feval (const double &h, const double *Y, double* DY, cudaStream_t stream) const override;
    
    // Exportar los datos a un archivo txt
    inline void archivo(const string &filename, const double *Y) const override { archivo2(filename,Y); };
    
private:
    // Auxiliary function f
    //inline double f(const double &y) const { return( ((y-0.7)*(y-1.3)) / ((y-0.7)*(y-1.3)+0.1) ); };
    
    // Indexation function which maps 2D spatial coordinates (i,j) to a 1D position in a vector
    inline int idx(const int &i, const int &j) const { return i * 2 + j; };
};
  
#endif 
  
  