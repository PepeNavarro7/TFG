#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include "Problema.h"
#include "Metodo.h"

using namespace std;

class RungeKutta: public Metodo{
public:   
    // Constructor de la clase
    RungeKutta (const int &o, const int &n);

    // Aplicar Runge-Kutta el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h0, const double *Y0, double *Yf) override;

    // Aplicar Runge-Kutta una unica vez
    void aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf);
};

#endif