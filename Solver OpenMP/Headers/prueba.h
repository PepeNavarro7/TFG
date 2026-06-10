#ifndef PRUEBA_H
#define PRUEBA_H

#include "Problema.h"

using namespace std;

// Problema 0, sirve de prueba
class prueba: public Problema {

private:
    int nx; // number of grid points at each dimension
            
public:
    // Constructor of the class
    prueba(const int &nx_points); 

    // Initialize stage vector Y0 with neqn components
    void init(double *Y0) const override; 

    //vector system function for the stiff term DY=G(t,Y) + the nonstiff term DY=F(t,Y)
    void feval (const double &t, const double *Y, double *DY) const override;

    inline void archivo (const string &filename, const double *Y) const override { archivo1(filename,Y); };
};
#endif