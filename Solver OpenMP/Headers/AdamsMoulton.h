#ifndef ADAMS_MOULTON_H
#define ADAMS_MOULTON_H

#include "Problema.h"
#include "RungeKutta.h"
#include "Metodo.h"

using namespace std;

class AdamsMoulton: public Metodo{
private:
    RungeKutta *ptr_runge;
public:   
    // Constructor de la clase
    AdamsMoulton (const int &n, RungeKutta* r);

    // Aplicar Adams-Bashford el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) override;    
};

#endif