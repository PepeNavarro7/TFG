#ifndef ADAMS_BASHFORD_H
#define ADAMS_BASHFORD_H

#include "Problema.h"
#include "RungeKutta.h"
#include "Metodo.h"

using namespace std;

class AdamsBashford: public Metodo{
private:
    const RungeKutta* const ptr_runge; // Puntero constante a objeto constante
    
public:   
    // Constructor de la clase
    AdamsBashford (const int &neqn, const int &orden, const RungeKutta* runge):
        Metodo(neqn, "Adams-Bashford", orden),
        ptr_runge(runge){};

    // Aplicar Adams-Bashford el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;
};

#endif