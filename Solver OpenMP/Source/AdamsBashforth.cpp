#ifndef ADAMS_BASHFORTH_CPP
#define ADAMS_BASHFORTH_CPP

#include "AdamsBashforth.h"
#include <iostream>
#include <omp.h>

using namespace std;

// Aplicar Adams-Bashforth con paralelismo de operaciones el numero necesaorio de veces
void AdamsBashforth::aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const {
    const int neqn = get_neqn();
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], // Vectores intermedios
           *Fn0 = new double[neqn], *Fn1 = new double[neqn], *Fn2 = new double[neqn], *Fn3 = new double[neqn], *Fn4 = new double[neqn], // Vectores funcion
           *Yaux = new double[neqn];

    vectorCopia(Y0, Yn0); // copia paralelizada
    const double h_RK = h/100.0;
    
    switch(get_orden()){ // Aplicamos orden-1 veces RK4 para obtener los primeros pasos
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

    switch(get_orden()){ // Arranque generando las feval
        case 1:
            problema->feval(t0, Yn0, Fn0); // f(tn0,Yn0) -> Fn0
        break;
        case 2:
            problema->feval(t0,      Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK, Yn1, Fn1); // f(tn1,Yn1) -> Fn1
        break;
        case 3:
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
        break;
        case 4:
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
            problema->feval(t0+h_RK*3.0, Yn3, Fn3); // f(tn3,Yn3) -> Fn3
        break;
    } // Fin del switch de arranque

    // Ahora aplicamos Adams-Bashforth del orden indicado
    const double h2=h/2.0, h12=h/12.0, h24=h/24.0;
    switch(get_orden()){ // Switch principal con el for que se trabaja
        case 1: 
            for(double tn = t0; tn<tf; tn+=h){
                escalarSumaMult(Yn0, h, Fn0, Yn1);// Yn0 + h * Fn0 -> Yn1 
                problema->feval(tn+h, Yn1, Fn1);  // f(tn1,Yn1) -> Fn1
                
                // Ahora que tenemos Yn1 & Fn1, convertimos los n en n-1 para hacer la siguiente iteracion
                swap(Yn1, Yn0); // Yn1 -> Yn0
                swap(Fn1, Fn0); // Fn1 -> Fn0
                // Tras los cambios, Yn1 & Fn1 contienen basura y serán reescritos                    
            } // Fin del bucle for iterativo
        break;
        case 2:
            for(double tn = t0+h_RK*1.0; tn<tf; tn+=h){
                // Yn2 = Yn1 + h/2 * (3*Fn1 - Fn0)
                escalar2Mult(3.0, Fn1, -1, Fn0, Yaux); // (3*Fn1 -1*Fn0) -> Yaux
                escalarSumaMult(Yn1, h2, Yaux, Yn2); // Yn1 + h2*Yaux -> Yn2
                problema->feval(tn+h, Yn2, Fn2);// f(tn2,Yn2) -> Fn2
                
                // Ahora que tenemos Yn2 & Fn2, convertimos los n en n-1 para hacer la siguiente iteracion
                swap(Yn2, Yn1); // Yn2 -> Yn1
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn2 -> Fn1
                // Tras los cambios, Yn2 & Fn2 contienen basura y serán reescritos                    
            } // Fin del bucle for iterativo
        break;
        case 3:
            for(double tn = t0+h_RK*2.0; tn<tf; tn+=h){
                // Yn3 = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
                escalar2Mult(-16.0, Fn1, 5.0, Fn0, Yaux); // -16*Fn1 + 5*Fn0 -> Yaux
                escalarPorVector(23.0, Fn2, Yaux); // Yaux+=23*Fn2
                escalarSumaMult(Yn2, h12, Yaux, Yn3);// Yn3 = Yn2 + h/12 * Yaux
                problema->feval(tn+h, Yn3, Fn3);// f(tn3,Yn3) -> Fn3

                // Ahora que tenemos Yn3 & Fn3, convertimos todos los n en n-1 para hacer la siguiente iteracion
                swap(Yn3, Yn2); // Yn3 -> Yn2
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn2 -> Fn1
                swap(Fn3, Fn2); // Fn3 -> Fn2
                // Tras los cambios, Yn3 & Fn3 contienen basura y serán reescritos                    
            } // Fin del bucle for iterativo
        break;
        case 4:
            for(double tn = t0+h_RK*3.0; tn<tf; tn+=h){
                // Yn4 = Yn3 + h/24 * (55*Fn3 - 59*Fn2 + 37*Fn1 - 9*Fn0)
                escalar2Mult(37.0, Fn1, -9.0, Fn0, Yaux); // 37*Fn1 - 9*Fn0 -> Yaux
                escalarSuma2Mult(Yaux, 55.0, Fn3, -59.0, Fn2, Yaux); // Yaux+= 55*Fn3 - 59*Fn2
                escalarSumaMult(Yn3, h24, Yaux, Yn4); // Yn3 + h/24 * Yaux -> Yn4
                problema->feval(tn+h, Yn4, Fn4);// f(tn4,Yn4) -> Fn4

                // Ahora que tenemos Yn4 & Fn4, convertimos todos los n en n-1 para hacer la siguiente iteracion
                swap(Yn4, Yn3); // Yn4 -> Yn3
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn2 -> Fn1
                swap(Fn3, Fn2); // Fn3 -> Fn2
                swap(Fn4, Fn3); // Fn4 -> Fn3
                // Tras los cambios, Yn4 & Fn4 contienen basura y serán reescritos                    
            } // Fin del bucle for iterativo
        break;
    } // Fin del switch principal

    switch(get_orden()){ // Switch para arrastrar el vector resultado
        case 1: vectorCopia(Yn0, Yf); break; // Yn0' -> Yf
        case 2: vectorCopia(Yn1, Yf); break; // Yn1' -> Yf
        case 3: vectorCopia(Yn2, Yf); break; // Yn2' -> Yf
        case 4: vectorCopia(Yn3, Yf); break; // Yn3' -> Yf
    } // Fin del switch resultado

    delete[] Yn0; delete[] Yn1; delete[] Yn2; delete[] Yn3; delete[] Yn4;
    delete[] Fn0; delete[] Fn1; delete[] Fn2; delete[] Fn3; delete[] Fn4;
    delete[] Yaux;
}
#endif