#ifndef RUNGE_KUTTA_I_H
#define RUNGE_KUTTA_I_H

#include "Problema.h"
#include "Metodo.h"

using namespace std;

class RungeKutta_i: public Metodo{
public:   
    // Constructor de la clase
    RungeKutta_i (const int &neqn, const int &orden):
        Metodo(neqn, "Runge-Kutta version_i", orden) { };

    // Aplicar Runge-Kutta el numero necesario de veces
    void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h0, const double *Y0, double *Yf) const override;

    // Aplicar Runge-Kutta ORDEN4 una unica vez para iniciar AB y AM
    void aplicarUnidad(const Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf) const;
};

#endif