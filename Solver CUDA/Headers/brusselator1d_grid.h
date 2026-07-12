#ifndef BRUSSELATOR1D_GRID_H
#define BRUSSELATOR1D_GRID_H

#include "brusselator1d.h"

using namespace std;

extern __constant__ double cte_br1dg_t; // Constante que utilizará el graph
extern __constant__ Params_brusselator1d cte_br1dg; // Estructura de datos constantes para los kernel

// PROBLEMA 3
// Class for the IVP-ODE representing the 1D Brusselator model 

class brusselator1d_grid:public brusselator1d{       
public:
    // Constructor of the class 
    brusselator1d_grid(const int &nx_points, const int &threads):
        brusselator1d(nx_points, threads, "Brusselator_1D grid", dim3(threads/2,2,1)){};

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;

    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 
    
    //vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override; 
    void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;
    
    // Exportar los datos a un archivo txt
    inline void archivo(const string &filename, const double *Y) const override { archivo2v2(filename,Y); };

    inline const void* get_t() const override { return (const void*)&cte_br1dg_t; }
    
private:    
    // Indexation function which maps 2D spatial coordinates (i,j) to a 1D position in a vector
    inline int idx(const int &i, const int &j) const override { return i + j * get_nx(); };
};
  
#endif 
  
  