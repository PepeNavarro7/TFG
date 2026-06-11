#ifndef ADAMS_BASHFORD_CPP
#define ADAMS_BASHFORD_CPP

#include "AdamsBashford.h"
#include <iostream>
#include <omp.h>

using namespace std;


// Constructor de la clase
AdamsBashford::AdamsBashford (const int &n, RungeKutta* r){
    neqn = n;
    ptr_runge = r;
    nombre = "Adams-Bashford";
}

// Aplicar Adams-Bashford ORDEN 4
void AdamsBashford::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], // Vectores intermedios
           *YF0 = new double[neqn], *YF1 = new double[neqn], *YF2 = new double[neqn], *YF3 = new double[neqn]; // Vectores funcion

    vectorCopia(Y0, Yn0); // copia paralelizada
    const double h_RK = h/100.0;
        
    // Aplicamos 3 veces Runge-Kutta para obtener los primeros pasos
    ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
    ptr_runge->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
    ptr_runge->aplicarUnidad(problema, t0+h_RK*2.0, h_RK, Yn2, Yn3);
    
    // Ahora aplicamos Adams-Bashford de Orden 4
    const double h24=h/24.0;
    
    #pragma omp parallel
    {  
        #pragma omp for 
        for (int i = 0; i < neqn; ++i){
            YF0[i] = problema->feval_i(t0,          Yn0, i); // f(tn0,Yn0) -> YF0
            YF1[i] = problema->feval_i(t0+h_RK,     Yn1, i); // f(tn1,Yn1) -> YF1
            YF2[i] = problema->feval_i(t0+h_RK*2.0, Yn2, i); // f(tn2,Yn2) -> YF2
            YF3[i] = problema->feval_i(t0+h_RK*3.0, Yn3, i); // f(tn3,Yn3) -> YF3
        }
            
        for(double tn = t0+h_RK*3.0; tn<tf; tn+=h){
            #pragma omp for // Yn4 = Yn3 + h/24 * (55*YF3 - 59*YF2 + 37*YF1 - 9*YF0)
            for (int i = 0; i < neqn; ++i)
                Yn4[i] = Yn3[i] + h24 * (55.0*YF3[i] - 59.0*YF2[i] + 37.0*YF1[i] - 9.0*YF0[i]);

            // Ahora que tenemos Yn4, convertimos todos los Yn en Yn-1, y los YFn en YFn-1 para hacer la siguiente iteracion
            #pragma omp sections
            {
                swap(Yn0, Yn1); // Yn1 -> Yn0
                swap(Yn1, Yn2); // Yn2 -> Yn1
                swap(Yn2, Yn3); // Yn3 -> Yn2
                swap(Yn3, Yn4); // Yn4 -> Yn3
                swap(YF0, YF1); // YF1 -> YF0
                swap(YF1, YF2); // YF2 -> YF1
                swap(YF2, YF3); // YF3 -> YF2
            } // Tras los cambios, Yn4 y YF3 contienen basura y serán reescritos

            // Calculamos el nuevo YF3 usando el Yn4 recién creado (que ahora es Yn3)
            #pragma omp for 
            for (int i = 0; i < neqn; ++i)
                YF3[i] = problema->feval_i(tn+h, Yn3, i); // f(tn3,Yn3) -> YF3
        } // Fin del bucle for iterativo

        #pragma omp for // Yn4 -> Yf
        for (int i = 0; i < neqn; ++i)
            Yf[i] = Yn4[i];

    }  // Fin del parallel 


    delete[] Yn0; delete[] Yn1; delete[] Yn2; delete[] Yn3; delete[] Yn4;
    delete[] YF0; delete[] YF1; delete[] YF2; delete[] YF3;
}

// Aplicar Adams-Bashford una sola vez, aportando los pasos intermedios de antemano
void AdamsBashford::aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Yn0, const double *Yn1, const double *Yn2, const double *Yn3, double *Yn4) const {
    double *YF0 = new double [neqn], *YF1 = new double [neqn], *YF2 = new double [neqn], *YF3 = new double [neqn]; // vectores auxiliares para los feval
    
    for (int i = 0; i < neqn; ++i){
        YF0[i] = problema->feval_i(t0,       Yn0, i); // f(tn0,Yn0) -> YF0
        YF1[i] = problema->feval_i(t0+h,     Yn1, i); // f(tn1,Yn1) -> YF1
        YF2[i] = problema->feval_i(t0+h*2.0, Yn2, i); // f(tn2,Yn2) -> YF2
        YF3[i] = problema->feval_i(t0+h*3.0, Yn3, i); // f(tn3,Yn3) -> YF3
    }

    const double h24=h/24.0;
    // Yn4 = Yn3 + h/24 * (55*YF3 - 59*YF2 + 37*YF1 - 9*YF0)
    for (int i = 0; i < neqn; ++i)
        Yn4[i] = Yn3[i] + h24 * (55.0*YF3[i] - 59.0*YF2[i] + 37.0*YF1[i] - 9.0*YF0[i]);
     
    
    delete[] YF0, YF1, YF2, YF3;
}
#endif