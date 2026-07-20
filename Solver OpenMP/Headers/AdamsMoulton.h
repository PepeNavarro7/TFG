#ifndef ADAMS_MOULTON_H
#define ADAMS_MOULTON_H

#include "Problema.h"
#include "Metodo.h"
#include "RungeKutta.h"

using namespace std;

class AdamsMoulton: public Metodo{
private:
    const RungeKutta* const ptr_runge;

public:   
    // Constructor de la clase
    AdamsMoulton (const int &neqn, const int &orden, const RungeKutta* runge):
        Metodo(neqn, "Adams-Moulton (paralelismo de operaciones)", orden),
        ptr_runge(runge) { };

    // Aplicar Adams-Moulton el numero necesario de veces
    void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const override;    
};

#endif