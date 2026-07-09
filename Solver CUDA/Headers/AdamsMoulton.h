#ifndef ADAMS_MOULTON_H
#define ADAMS_MOULTON_H

#include "Problema.h"
#include "Metodo.h"
#include "RungeKutta.h"
#include "AdamsBashford.h"

using namespace std;

struct Params_AdamsMoulton {
    int neqn;
    double h;
    double h2;
    double h12;
    double h24;
    double h720;
};


class AdamsMoulton: public Metodo{
private:
    const RungeKutta *ptr_runge;
    const AdamsBashford *ptr_bashford;

public:   
    // Constructor de la clase
    AdamsMoulton (const int &n, const int &orden, const int &threads, const RungeKutta* runge, const AdamsBashford* bash):
        Metodo(n, "Adams-Moulton", orden, threads),
        ptr_runge(runge), ptr_bashford(bash) { };

    void updateConstants(const int &neqn, const double &h) const override;

    // Aplicar Adams-Moulton el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;    
};

#endif