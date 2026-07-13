#ifndef RUNGE_KUTTA_GRAPH_H
#define RUNGE_KUTTA_GRAPH_H

#include "Problema.h"
#include "RungeKutta.h"

using namespace std;

class RungeKutta_graph: public RungeKutta{
private:
    const RungeKutta *ptr_runge;
public:   
    // Constructor de la clase
    RungeKutta_graph (const int &n, const int &orden, const int &threads, RungeKutta* runge):
        RungeKutta(n, orden, threads, "Runge-Kutta graph"),
        ptr_runge(runge) { };
    
    // Aplicar Runge-Kutta el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;
};

#endif