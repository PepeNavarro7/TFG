#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include "Problema.h"
#include "Metodo.h"

using namespace std;

class RungeKutta: public Metodo{
public:   
    // Constructor de la clase
    RungeKutta (const int &neqn, const int &orden):
        Metodo(neqn, "Runge-Kutta (paralelismo de operaciones)", orden) { };

    // Aplicar Runge-Kutta el numero necesario de veces
    void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h0, const double* __restrict Y0, double* __restrict Yf) const override;

    // Aplicar Runge-Kutta ORDEN4 una unica vez para iniciar AB y AM
    void aplicarUnidad(const Problema* problema, const double &t0, const double &h, const double* __restrict Y0, double* __restrict Yf) const;
};

#endif