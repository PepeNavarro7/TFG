#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include "Problema.h"
#include "Metodo.h"

using namespace std;

struct Params_RungeKutta {
    int neqn;
    double h;
    double h2;
    double h6;
};

class RungeKutta: public Metodo{
public:   
    // Constructor de la clase
    RungeKutta (const int &n, const int &orden, const int &threads):
        Metodo(n, "Runge-Kutta", orden, threads) { };

    void updateConstants(const int &neqn, const double &h) const override;
    
    // Aplicar Runge-Kutta el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;

    // Aplicar Runge-Kutta una unica vez
    void aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf) const;
};

#endif