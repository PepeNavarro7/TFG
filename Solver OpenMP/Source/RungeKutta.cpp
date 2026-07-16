#ifndef RUNGE_KUTTA_CPP
#define RUNGE_KUTTA_CPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "RungeKutta.h"
#include <string>

using namespace std;

// Aplicar Runge-Kutta el numero necesario de veces
void RungeKutta::aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const {
    const int neqn = get_neqn();
    double *Yn = new double [neqn], // vector con la Y en cada iteracion
        *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar    

    vectorCopia(Y0, Yn); // Y0 -> Yn, para la primera iteración
        
    const double h2=0.5*h, h_n=-1.0*h, h_2 = 2.0*h, h6=h/6.0;
    for(double tn=t0; tn<tf; tn+=h){ // Todas las llamadas han de hacerse de forma secuencial, ya que cada una necesita de la anterior   
        switch(get_orden()){
        case 1: 
            problema->feval(tn, Yn, K1); // K1 = feval(tn, Yn)
            escalarPorVector(h, K1, Yn);// Yn+1 = Yn + h*K1
        break;
        case 2: // Heun
            problema->feval(tn, Yn, K1); // K1 = feval(tn, Yn)
            // K2 = feval(tn + h, Yn + h*K1) 
            escalarSumaMult(Yn, h, K1, Yaux); // Yn + h*K1 -> Yaux
            problema->feval(tn+h, Yaux, K2); // Definimos K2
                
            #pragma omp parallel for schedule(static) default(none) shared(Yn, h2, K1, K2, neqn)
            for (int i = 0; i < neqn; ++i) { 
                Yn[i] += + h2 * ( K1[i] + K2[i] ) ; // // Yn+1 = Yn + h/2 * (K1 + K2)
            } // Barrera implícita
        break;
        case 3: 
            problema->feval(tn, Yn, K1); // K1 = feval(tn, Yn)
            // K2 = feval(tn + h/2, Yn + h/2 * K1) 
            escalarSumaMult(Yn, h2, K1, Yaux); // Yn + h/2*K1 -> Yaux
            problema->feval(tn+h2, Yaux, K2); // Definimos K2
            // K3 = feval(tn + h, Yn -h*K1 + 2*h*K2 ) 
            escalarSuma2Mult(Yn, h_n, K1, h_2, K2, Yaux); // Yn -h*K1 + 2*h*K2 -> Yaux
            problema->feval(tn+h, Yaux, K3);// Definimos K3
                
            #pragma omp parallel for schedule(static) default(none) shared(Yn, h6, K1, K2, K3, neqn)
            for (int i = 0; i < neqn; ++i) {
                Yn[i] += h6 * ( K1[i] + 4.0*K2[i] + K3[i] ) ;// Yn+1 = Yn + h/6 * (K1 + 4*K2 + K3)
            } // Barrera implícita
        break;
        case 4: 
            problema->feval(tn, Yn, K1); // K1 = feval(tn, Yn)
            // K2 = feval(tn + h/2, Yn + K1*h/2)
            escalarSumaMult(Yn, h2, K1, Yaux); // Yn + h/2*K1 -> Yaux
            problema->feval(tn+h2, Yaux, K2); // Definimos K2
            // K3 = feval(tn + h/2, Yn + K2*h/2)  
            escalarSumaMult(Yn, h2, K2, Yaux); // Yn + h/2*K2 -> Yaux
            problema->feval(tn+h2, Yaux, K3); // Definimos K3
            // K4 = feval(tn + h, Yn + h*K3)
            escalarSumaMult(Yn, h, K3, Yaux); // Yn + h*K3 -> Yaux               
            problema->feval(tn+h, Yaux, K4); // Definimos K4

            #pragma omp parallel for schedule(static) default(none) shared(Yn, h6, K1, K2, K3, K4, neqn)
            for (int i = 0; i < neqn; ++i){
                Yn[i] += h6 * ( K1[i] + 2.0*K2[i] + 2.0*K3[i] + K4[i] ) ; // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
            } // Barrera implícita
        break; 
        } // Fin del switch
    } // Fin del bucle for
        
    // Yn es el valor que arrastramos de la ultima iteracion
    vectorCopia(Yn, Yf);

    delete[] Yn;
    delete[] K1; delete[] K2; delete[] K3; delete[] K4;
    delete[] Yaux;
}

// Aplicar Runge-Kutta ORDEN4 una unica vez
void RungeKutta::aplicarUnidad(const Problema* problema, const double &t0, const double &h, const double* __restrict Y0, double* __restrict Yf) const {
    const int neqn = get_neqn();
    double *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar
    
    const double h2=h/2.0, h6=h/6.0;
    problema->feval(t0, Y0, K1); // K1 = feval(tn, Yn)
    // K2 = feval(tn + h/2, Yn + K1*h/2)
    escalarSumaMult(Y0, h2, K1, Yaux); // Yn + h/2*K1 -> Yaux
    problema->feval(t0+h2, Yaux, K2); // Definimos K2
    // K3 = feval(tn + h/2, Yn + K2*h/2)  
    escalarSumaMult(Y0, h2, K2, Yaux); // Yn + h/2*K2 -> Yaux
    problema->feval(t0+h2, Yaux, K3); // Definimos K3
    // K4 = feval(tn + h, Yn + h*K3)
    escalarSumaMult(Y0, h, K3, Yaux); // Yn + h*K3 -> Yaux               
    problema->feval(t0+h, Yaux, K4); // Definimos K4

    #pragma omp parallel for schedule(static) default(none) shared(Yf, h6, K1, K2, K3, K4, neqn)
    for (int i = 0; i < neqn; ++i){
        Yf[i] += h6 * ( K1[i] + 2.0*K2[i] + 2.0*K3[i] + K4[i] ) ; // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
    } 
        
    delete[] K1;
    delete[] K2;
    delete[] K3;
    delete[] K4;
    delete[] Yaux;
}


#endif