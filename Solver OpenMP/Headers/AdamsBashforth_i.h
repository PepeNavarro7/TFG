#ifndef ADAMS_BASHFORTH_I_H
#define ADAMS_BASHFORTH_I_H

#include "Problema.h"
#include "RungeKutta_i.h"
#include "Metodo.h"

using namespace std;

class AdamsBashforth_i: public Metodo{
private:
    const RungeKutta_i* const ptr_runge; // Puntero constante a objeto constante
    
public:   
    // Constructor de la clase
    AdamsBashforth_i (const int &neqn, const int &orden, const RungeKutta_i* runge):
        Metodo(neqn, "Adams-Bashforth (paralelismo de elementos)", orden),
        ptr_runge(runge){};

    // Aplicar Adams-Bashforth el numero necesario de veces
    void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const override;
};

#endif