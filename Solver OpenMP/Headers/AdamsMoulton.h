#ifndef ADAMS_MOULTON_H
#define ADAMS_MOULTON_H

#include "Problema.h"
#include "Metodo.h"
#include "RungeKutta.h"
#include "AdamsBashford.h"

using namespace std;

class AdamsMoulton: public Metodo{
private:
    RungeKutta *ptr_runge;
    AdamsBashford *ptr_bashford;

public:   
    // Constructor de la clase
    AdamsMoulton (const int &n, RungeKutta* r, AdamsBashford* ab);

    // Aplicar Adams-Moulton el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) override;    
};

#endif