#ifndef ADAMS_BASHFORD_H
#define ADAMS_BASHFORD_H

#include "Problema.h"
#include "RungeKutta.h"
#include "Metodo.h"

using namespace std;

struct Params_AdamsBashford {
    int neqn;
    double h;
    double h2;
    double h12;
    double h24;
};

extern __constant__ Params_AdamsBashford cte_AB;

class AdamsBashford: public Metodo{
private:
    const RungeKutta *ptr_runge;

public:   
    // Constructor de la clase
    AdamsBashford (const int &neqn, const int &orden, const int &threads, RungeKutta* runge):
        Metodo(neqn, "Adams-Bashford", orden, threads),
        ptr_runge(runge) { };

    virtual void updateConstants(const int &neqn, const double &h) const override;

    // Aplicar Adams-Bashford el numero necesario de veces
    virtual void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;    

protected:
    // Constructor para hija
    AdamsBashford (const int &neqn, const int &orden, const int &threads, RungeKutta* runge, const string &name):
        Metodo(neqn, name, orden, threads),
        ptr_runge(runge) { };

    inline const RungeKutta* get_ptr_runge() const { return ptr_runge; }
};

#endif