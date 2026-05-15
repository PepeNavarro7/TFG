#ifndef METODO_CU
#define METODO_CU

#include "Metodo.h"

using namespace std;

void Metodo::escalarPorVector(const double &esc, const double *X, double *Y){
    for(int i=0; i<neqn; ++i){
        Y[i]+=X[i]*esc;
    }
}

void Metodo::vectorCopia(const double *X, double *Y){
    for (int i=0; i<neqn; ++i){
        Y[i]=X[i];
    }
}

#endif