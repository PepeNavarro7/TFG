#ifndef METODO_CPP
#define METODO_CPP

#include "Metodo.h"

using namespace std;

void Metodo::escalarPorVector(const double &esc, const double* __restrict X, double* __restrict Y) const {
    #pragma omp parallel for schedule(static) default(none) shared(X, Y, esc, neqn)
    for(int i=0; i<neqn; ++i){
        Y[i]+=X[i]*esc;
    } 
}

void Metodo::vectorCopia(const double* __restrict X, double* __restrict Y) const {
    #pragma omp parallel for schedule(static) default(none) shared(X, Y,  neqn)
    for (int i=0; i<neqn; ++i){
        Y[i]=X[i];
    } 
}

void Metodo::escalarSumaMult(const double* __restrict Y0, const double &esc, const double* __restrict X, double* __restrict Yf) const {
    #pragma omp parallel for schedule(static) default(none) shared(X, Y0, Yf, esc, neqn)
    for(int i=0; i<neqn; ++i){
        Yf[i]=Y0[i] + esc * X[i];
    } 
}

void Metodo::escalar2Mult(const double &esc1, const double* __restrict X, const double &esc2, const double* __restrict Z, double* __restrict Yf) const{
    #pragma omp parallel for schedule(static) default(none) shared(esc1, X, esc2, Z, Yf)
    for(int i=0; i<neqn; ++i){
        Yf[i]=esc1*X[i] +  esc2*Z[i];
    } 
}

void Metodo::escalarSuma2Mult(const double* __restrict Y0, const double &esc1, const double* __restrict X, const double &esc2, const double* __restrict Z, double* __restrict Yf) const {
    #pragma omp parallel for schedule(static) default(none) shared(Y0, esc1, X, esc2, Z, Yf)
    for(int i=0; i<neqn; ++i){
        Yf[i]=Y0[i] + esc1*X[i] +  esc2*Z[i];
    } 
}
#endif