#ifndef METODO_CPP
#define METODO_CPP

#include "Metodo.h"

using namespace std;

// SECUENCIAL
void Metodo::escalarPorVector(const double &esc, const double *X, double *Y){
    for(int i=0; i<neqn; ++i){
        Y[i]+=X[i]*esc;
    } 
}

// SECUENCIAL
void Metodo::vectorCopia(const double *X, double *Y){
    #pragma omp parallel for
    for (int i=0; i<neqn; ++i){
        Y[i]=X[i];
    } 
}

int Metodo::neqn = 0;

#endif