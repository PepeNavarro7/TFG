#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include "Problema.h"

using namespace std;

class RungeKutta{
private:
    const double PI=3.14159265358979;
    int neqn;  // numero de ecuaciones

public:   
    // Constructor de la clase
    RungeKutta (const int &n);

    // Aplicar Runge-Kutta el numero necesario de veces
    void aplicarRK(Problema* problema, const double &t0, const double &tf, const double &h0, const double *Y0, double *Y1);

private:
    // Escalar esc * vector X + vector Y -> Y
    void escalarPorVector(const double &esc, const double *X, double *Y);

    // Copia de X en Y
    void vectorCopia(const double *X, double *Y);
};

#endif