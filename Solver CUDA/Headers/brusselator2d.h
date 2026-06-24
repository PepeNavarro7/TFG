#ifndef BRUSSELATOR2D_H
#define BRUSSELATOR2D_H

#include "Problema.h"
#include <string>

using namespace std;


// Problema 4
// the Brusselator 2D model 
class brusselator2d: public Problema {
private:
    const double alpha = 0.002, A = 1.0, B = 3.4; // variables para el calculo de los valores
    const int nx; // number of grid points at x dimension
    const int ny; // number of grid points at y dimension
    const double dtx_sq; // Spatial step squared
    const double DD;

public:
    // Constructor of the class 
    brusselator2d(const int &nx_points, const int &threads):
        Problema(2.0*nx_points*nx_points, "Brusselator_2D", 1.0/(nx_points+1), dim3( (nx_points*nx_points*2.0+threads-1)/threads, 1, 1 ), dim3(threads,1,1)),
        nx(nx_points), ny(nx_points), dtx_sq(dtx*dtx), DD(alpha/(dtx*dtx)) { 
            updateConstants();
        };

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;

    // Initialize stage vector Y0 with neqn components
    void init(double* Y0) const override;

    //vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval(const double &t, const double* Y, double* DY) const override;

    // Exportar los datos a un archivo txt
    inline void archivo(const string &filename, const double *Y) const override { archivo3(filename,Y); };

private:
    // Auxiliary function f
    //double f(const int &i, const int &j, const double &t) const;

    // Indexation function which maps 2D spatial coordinates (i,j) to a 1D position in a vector
    inline int idx(const int &i, const int &j, const int &k) const { return 2 * (i * ny + j) + k; } 
};

#endif