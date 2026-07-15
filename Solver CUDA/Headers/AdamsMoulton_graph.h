#ifndef ADAMS_MOULTON_GRAPH_H
#define ADAMS_MOULTON_GRAPH_H

#include "RungeKutta.h"
#include "AdamsBashford.h"
#include "AdamsBashford_graph.h"
#include "AdamsMoulton.h"

using namespace std;

class AdamsMoulton_graph: public AdamsMoulton{
private:
    const AdamsBashford_graph* ptr_abgraph;
    const AdamsMoulton *ptr_moulton;

public:   
    // Constructor de la clase
    AdamsMoulton_graph (const int &neqn, const int &orden, const int &threads, const RungeKutta* runge, const AdamsBashford* bash, const AdamsBashford_graph* bash_g, const AdamsMoulton* moult):
        AdamsMoulton(neqn, orden, threads, "Adams-Moulton graph", runge, bash),
        ptr_abgraph(bash_g), ptr_moulton(moult){ };

    // Aplicar Adams-Moulton el numero necesario de veces
    void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const override;    
};

#endif