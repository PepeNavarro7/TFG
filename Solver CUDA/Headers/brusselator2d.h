#ifndef BRUSSELATOR2D_H
#define BRUSSELATOR2D_H

#include "Problema.h"
#include <string>

using namespace std;

// Problema 4
struct Params_brusselator2d {
    int neqn;
    int nx;
    double A;
    double B;
    double dtx;
    double DD;
};

extern __constant__ Params_brusselator2d cte_br2d; // Estructura de datos constantes para los kernel
extern __constant__ double cte_br2d_t; // Constante que utilizará el graph

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
        nx(nx_points), ny(nx_points), dtx_sq(get_dtx()*get_dtx()), DD(alpha/(get_dtx()*get_dtx())) { };

    // Definicion de los valores constantes para el kernel
    virtual void updateConstants() const override;

    // Initialize stage vector Y0 with neqn components
    virtual void init(double* Y0) const override;

    //vector system function for the nonstiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    virtual void feval (const double &t, const double* Y, double* DY) const override;
    virtual void feval (const double &offset, const double *Y, double* DY, cudaStream_t stream) const override;

    // Exportar los datos a un archivo txt
    virtual inline void archivo(const string &filename, const double *Y) const override { archivo3(filename,Y); };

    virtual inline const void* get_t() const override{ return (const void*)&cte_br2d_t; }

protected:
    // Auxiliary function f
    //double f(const int &i, const int &j, const double &t) const;

    // Indexation function which maps 2D spatial coordinates (i,j) to a 1D position in a vector
    virtual inline int idx(const int &i, const int &j, const int &k) const { return 2 * (i * ny + j) + k; } 

    // Constructor para las hijas
    brusselator2d(const int &nx_points, const int &threads, const string &name):
        Problema(2.0*nx_points*nx_points, name, 1.0/(nx_points+1), dim3( (nx_points*nx_points*2.0+threads-1)/threads, 1, 1 ), dim3(threads,1,1)),
        nx(nx_points), ny(nx_points), dtx_sq(get_dtx()*get_dtx()), DD(alpha/(get_dtx()*get_dtx())) { };

public:
    inline double get_alpha() const { return alpha; }; 
    inline double get_A() const { return A; }; 
    inline double get_B() const { return B; }; 
    inline double get_nx() const { return nx; }; 
    inline double get_ny() const { return ny; }; 
    inline double get_dtx_sq() const { return dtx_sq; }; 
    inline double get_DD() const { return DD; };
};

#endif