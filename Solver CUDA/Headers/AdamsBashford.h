#ifndef ADAMS_BASHFORD_H
#define ADAMS_BASHFORD_H

#include "Problema.h"
#include "RungeKutta.h"
#include "Metodo.h"

using namespace std;

class AdamsBashford: public Metodo{
private:
    const RungeKutta *ptr_runge;

public:   
    // Constructor de la clase
    AdamsBashford (const int &n, const int &orden, const int &threads, RungeKutta* runge):
        Metodo(n, "Adams-Bashford", orden, threads),
        ptr_runge(runge) { };

    // Aplicar Adams-Bashford el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;    

    // Aplicar Adams-Bashford una sola vez, pero aportando los pasos intermedios de antemano
    void aplicarUnidadSinRK(Problema* problema, const double &t0, const double &h, const double *Yn0, const double *Y1, const double *Y2, const double *Y3, double *Y4) const;
};

#endif