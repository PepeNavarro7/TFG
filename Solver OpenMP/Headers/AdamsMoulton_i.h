#ifndef ADAMS_MOULTON_I_H
#define ADAMS_MOULTON_I_H

#include "Problema.h"
#include "Metodo.h"
#include "RungeKutta_i.h"

using namespace std;

class AdamsMoulton_i: public Metodo{
private:
    const RungeKutta_i* const ptr_runge;

public:   
    // Constructor de la clase
    AdamsMoulton_i (const int &neqn, const int &orden, const RungeKutta_i* runge):
        Metodo(neqn, "Adams-Moulton version_i", orden),
        ptr_runge(runge) { };

    // Aplicar Adams-Moulton el numero necesario de veces
    void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const override;    
};

#endif