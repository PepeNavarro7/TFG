#ifndef BRUSSELATOR2D_H
#define BRUSSELATOR2D_H

#include "Problema.h"

using namespace std;


// Problema 4
// the Brusselator 2D model 

class brusselator2d: public Problema {
private:
    const double alpha = 0.002, A = 1.0, B = 3.4; // variables para el calculo de los valores
    const int nx, // number of grid points at x dimension
        ny; // number of grid points at y dimension
    const double dtx_sq, // Spatial step squared
        DD;

public:
    // Constructor of the class 
    brusselator2d(const int &nx_points):
        Problema( (2*nx_points*nx_points), "Brusselator_2D", (1.0/(nx_points+1.0)) ),
        nx(nx_points), ny(nx_points), dtx_sq(get_dtx()*get_dtx()), DD( alpha/(get_dtx()*get_dtx()) ) { };

    // Initialize stage vector Y0 with neqn components
    void init(double* __restrict Y0) const override;

    // Vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval(const double &t, const double* __restrict Y, double* __restrict DY) const override;

    // Funcion feval para un solo término
    double feval_i (const double &t, const double* __restrict Y, const int &i) const override;

    // Exportar los datos a un archivo txt
    inline void archivo(const string &filename, const double* __restrict Y) const override { archivo3(filename,Y); };

private:
    // Auxiliary function f
    double f(const int &i, const int &j, const double &t) const;

    // Indexation function which maps 2D spatial coordinates (i,j) to a 1D position in a vector
    inline int idx(const int &i, const int &j, const int &k) const { return 2 * (i * ny + j) + k; }
};

#endif