#ifndef PRUEBA_H
#define PRUEBA_H

#include "Problema.h"
#include <string>

using namespace std;

class prueba: public Problema {

private:
    const int nx;
            
public:
    // Constructor of the class
    prueba(const int &nx_points, const int &threads):
        Problema (nx_points, "Prueba manual", 1.0/nx_points, threads), 
        nx(nx_points){ 
            updateConstants();
        };

    // Definicion de los valores constantes para el kernel
    void updateConstants() const override;

    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 

    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override;

    inline void archivo (const string &filename, const double *Y) const override { archivo1(filename,Y); };
};
#endif