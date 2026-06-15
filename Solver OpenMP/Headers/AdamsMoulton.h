#ifndef ADAMS_MOULTON_H
#define ADAMS_MOULTON_H

#include "Problema.h"
#include "Metodo.h"
#include "RungeKutta.h"
#include "AdamsBashford.h"

using namespace std;

class AdamsMoulton: public Metodo{
private:
    const RungeKutta* const ptr_runge;
    const AdamsBashford* const ptr_bashford;

public:   
    // Constructor de la clase
    AdamsMoulton (const int &neqn, const int &orden, const RungeKutta* runge, const AdamsBashford* ab):
        Metodo(neqn, "Adams-Moulton", orden),
        ptr_runge(runge), ptr_bashford(ab) { };

    // Aplicar Adams-Moulton el numero necesario de veces
    void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;    
};

#endif