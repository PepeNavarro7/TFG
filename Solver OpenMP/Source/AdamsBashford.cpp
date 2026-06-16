#ifndef ADAMS_BASHFORD_CPP
#define ADAMS_BASHFORD_CPP

#include "AdamsBashford.h"
#include <iostream>
#include <omp.h>

using namespace std;

// Aplicar Adams-Bashford
void AdamsBashford::aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], // Vectores intermedios
           *YF0 = new double[neqn], *YF1 = new double[neqn], *YF2 = new double[neqn], *YF3 = new double[neqn]; // Vectores funcion

    vectorCopia(Y0, Yn0, neqn); // copia paralelizada
    const double h_RK = h/100.0;
    
    switch(orden){ // Aplicamos orden-1 veces Runge-Kutta para obtener los primeros pasos
        case 1: break; // Orden 1 no necesita RK
        case 2: 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
        break;
        case 3: 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
        break;
        case 4: 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
            ptr_runge->aplicarUnidad(problema, t0+h_RK*2.0, h_RK, Yn2, Yn3);
        break;
    }

    switch(orden){ // Arranque generando las feval
        case 1:
            #pragma omp parallel for 
            for (int i = 0; i < neqn; ++i)
                YF0[i] = problema->feval_i(t0, Yn0, i); // f(tn0,Yn0) -> YF0
        break;
        case 2:
            #pragma omp parallel for 
            for (int i = 0; i < neqn; ++i){
                YF0[i] = problema->feval_i(t0,      Yn0, i); // f(tn0,Yn0) -> YF0
                YF1[i] = problema->feval_i(t0+h_RK, Yn1, i); // f(tn1,Yn1) -> YF1
            }
        break;
        case 3:
            #pragma omp parallel for 
            for (int i = 0; i < neqn; ++i){
                YF0[i] = problema->feval_i(t0,          Yn0, i); // f(tn0,Yn0) -> YF0
                YF1[i] = problema->feval_i(t0+h_RK,     Yn1, i); // f(tn1,Yn1) -> YF1
                YF2[i] = problema->feval_i(t0+h_RK*2.0, Yn2, i); // f(tn2,Yn2) -> YF2
            }
        break;
        case 4:
            #pragma omp parallel for 
            for (int i = 0; i < neqn; ++i){
                YF0[i] = problema->feval_i(t0,          Yn0, i); // f(tn0,Yn0) -> YF0
                YF1[i] = problema->feval_i(t0+h_RK,     Yn1, i); // f(tn1,Yn1) -> YF1
                YF2[i] = problema->feval_i(t0+h_RK*2.0, Yn2, i); // f(tn2,Yn2) -> YF2
                YF3[i] = problema->feval_i(t0+h_RK*3.0, Yn3, i); // f(tn3,Yn3) -> YF3
            }
        break;
    } // Fin del switch de arranque

    // Ahora aplicamos Adams-Bashford del orden indicado
    const double h2=h/2.0, h12=h/12.0, h24=h/24.0;
    switch(orden){ // Switch principal con el for que se trabaja
        case 1: 
            #pragma omp parallel
            {
                for(double tn = t0; tn<tf; tn+=h){
                    #pragma omp for // Yn1 = Yn0 + h * YF0
                    for (int i = 0; i < neqn; ++i)
                        Yn1[i] = Yn0[i] + h * YF0[i];

                    // Ahora que tenemos Yn1, convertimos todos los Yn en Yn-1, y los YFn en YFn-1 para hacer la siguiente iteracion
                    #pragma omp single
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                    } // Tras los cambios, Yn1 contiene basura y será reescrito

                    // Calculamos el nuevo YF0 usando el Yn1 recién creado (que ahora es Yn0)
                    #pragma omp for 
                    for (int i = 0; i < neqn; ++i)
                        YF0[i] = problema->feval_i(tn+h, Yn0, i); // f(tn1,Yn1) -> YF1
                } // Fin del bucle for iterativo
            } // Fin del parallel 1
        break;
        case 2:
            #pragma omp parallel shared(h2)
            {
                for(double tn = t0+h_RK*1.0; tn<tf; tn+=h){
                    #pragma omp for // Yn2 = Yn1 + h/2 * (3*YF1 - YF0)
                    for (int i = 0; i < neqn; ++i)
                        Yn2[i] = Yn1[i] + h2 * (3.0*YF1[i] - YF0[i]);

                    // Ahora que tenemos Yn2, convertimos todos los Yn en Yn-1, y los YFn en YFn-1 para hacer la siguiente iteracion
                    #pragma omp single
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                        swap(Yn2, Yn1); // Yn2 -> Yn1
                        swap(YF1, YF0); // YF1 -> YF0
                    } // Tras los cambios, Yn2 & YF1 contienen basura y serán reescritos

                    // Calculamos el nuevo YF1 usando el Yn2 recién creado (que ahora es Yn1)
                    #pragma omp for 
                    for (int i = 0; i < neqn; ++i)
                        YF1[i] = problema->feval_i(tn+h, Yn1, i); // f(tn2,Yn2) -> YF2
                } // Fin del bucle for iterativo
            } // Fin del parallel 2
        break;
        case 3:
            #pragma omp parallel shared(h12)
            {
                for(double tn = t0+h_RK*2.0; tn<tf; tn+=h){
                    #pragma omp for // Yn3 = Yn2 + h/12 * (23*YF2 - 16*YF1 + 5*YF0)
                    for (int i = 0; i < neqn; ++i)
                        Yn3[i] = Yn2[i] + h12 * (23.0*YF2[i] - 16.0*YF1[i] + 5.0*YF0[i]);

                    // Ahora que tenemos Yn3, convertimos todos los Yn en Yn-1, y los YFn en YFn-1 para hacer la siguiente iteracion
                    #pragma omp single
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                        swap(Yn2, Yn1); // Yn2 -> Yn1
                        swap(Yn3, Yn2); // Yn3 -> Yn2
                        swap(YF1, YF0); // YF1 -> YF0
                        swap(YF2, YF1); // YF2 -> YF1
                    } // Tras los cambios, Yn3 & YF2 contienen basura y serán reescritos

                    // Calculamos el nuevo YF2 usando el Yn3 recién creado (que ahora es Yn2)
                    #pragma omp for 
                    for (int i = 0; i < neqn; ++i)
                        YF2[i] = problema->feval_i(tn+h, Yn2, i); // f(tn3,Yn3) -> YF3
                } // Fin del bucle for iterativo
            } // Fin del parallel 3
        break;
        case 4:
            #pragma omp parallel shared(h24)
            {
                for(double tn = t0+h_RK*3.0; tn<tf; tn+=h){
                    #pragma omp for // Yn4 = Yn3 + h/24 * (55*YF3 - 59*YF2 + 37*YF1 - 9*YF0)
                    for (int i = 0; i < neqn; ++i)
                        Yn4[i] = Yn3[i] + h24 * (55.0*YF3[i] - 59.0*YF2[i] + 37.0*YF1[i] - 9.0*YF0[i]);

                    // Ahora que tenemos Yn4, convertimos todos los Yn en Yn-1, y los YFn en YFn-1 para hacer la siguiente iteracion
                    #pragma omp single
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                        swap(Yn2, Yn1); // Yn2 -> Yn1
                        swap(Yn3, Yn2); // Yn3 -> Yn2
                        swap(Yn4, Yn3); // Yn4 -> Yn3
                        swap(YF1, YF0); // YF1 -> YF0
                        swap(YF2, YF1); // YF2 -> YF1
                        swap(YF3, YF2); // YF3 -> YF2
                    } // Tras los cambios, Yn4 & YF3 contienen basura y serán reescritos

                    // Calculamos el nuevo YF3 usando el Yn4 recién creado (que ahora es Yn3)
                    #pragma omp for 
                    for (int i = 0; i < neqn; ++i)
                        YF3[i] = problema->feval_i(tn+h, Yn3, i); // f(tn4,Yn4) -> YF4
                } // Fin del bucle for iterativo
            } // Fin del parallel 4
        break;
    } // Fin del switch principal

    switch(orden){ // Switch para arrastrar el vector resultado
        case 1:
            #pragma omp parallel for // Yn0 -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn0[i];
        break;
        case 2:
            #pragma omp parallel for // Yn1 -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn1[i];
        break;
        case 3:
            #pragma omp parallel for // Yn2 -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn2[i];
        break;
        case 4:
            #pragma omp parallel for // Yn3 -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn3[i];
        break;
    } // Fin del switch resultado

    delete[] Yn0; delete[] Yn1; delete[] Yn2; delete[] Yn3; delete[] Yn4;
    delete[] YF0; delete[] YF1; delete[] YF2; delete[] YF3;
}
#endif