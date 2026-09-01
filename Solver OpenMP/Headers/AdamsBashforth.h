#ifndef ADAMS_BASHFORTH_H
#define ADAMS_BASHFORTH_H

#include "Problema.h"
#include "RungeKutta.h"
#include "Metodo.h"

using namespace std;

class AdamsBashforth: public Metodo{
private:
    const RungeKutta* const ptr_runge; // Puntero constante a objeto constante
    
public:   
    // Constructor de la clase
    AdamsBashforth (const int &neqn, const int &orden, const RungeKutta* runge):
        Metodo(neqn, "Adams-Bashforth (paralelismo de operaciones)", orden),
        ptr_runge(runge){};

    // Aplicar Adams-Bashforth el numero necesario de veces
    void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const override;
};

#endif