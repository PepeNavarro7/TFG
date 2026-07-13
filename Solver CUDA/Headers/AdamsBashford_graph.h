#ifndef ADAMS_BASHFORD_GRAPH_H
#define ADAMS_BASHFORD_GRAPH_H

#include "AdamsBashford.h"

using namespace std;

extern __constant__ Params_AdamsBashford cte_ABg;

class AdamsBashford_graph: public AdamsBashford{
private:
    const AdamsBashford *ptr_bash;
public:   
    // Constructor de la clase
    AdamsBashford_graph (const int &neqn, const int &orden, const int &threads, RungeKutta* runge, AdamsBashford* bash):
        AdamsBashford(neqn, orden, threads, runge, "Adams-Bashford graph"),
        ptr_bash(bash) { };

    void updateConstants(const int &neqn, const double &h) const override;

    // Aplicar Adams-Bashford graph el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;    
};

#endif