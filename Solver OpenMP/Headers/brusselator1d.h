#ifndef BRUSSELATOR1D_H
#define BRUSSELATOR1D_H

#include "Problema.h"

using namespace std;

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 
class brusselator1d:public Problema{

private:
    int nx; // number of grid points at each dimension
    double dtx_squared; // Spatial step squared
    double DD;
    const double alpha=1.0/50.0, A=1.0, B=3.0; // variables auxiliares para el calculo

public:
    // Constructor of the class 
    brusselator1d (const int &nx_points);

    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 
    
    //vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override; 

    double feval_i (const double &t, const double *Y, const int &i) const override;

    // Exportar los datos a un archivo txt
    inline void archivo(const string &filename, const double *Y) const override { archivo2(filename,Y); };

private:
    // Auxiliary function f
    inline double f(const double &y) const { return( ((y-0.7)*(y-1.3)) / ((y-0.7)*(y-1.3)+0.1) ); };
    
    // Indexation function which maps 2D spatial coordinates (i,j) to a 1D position in a vector
    inline int idx(const int &i, const int &j) const { return i * 2 + j; };
};
  
#endif 
  
  