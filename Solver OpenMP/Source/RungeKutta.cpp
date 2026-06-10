#ifndef RUNGE_KUTTA_CPP
#define RUNGE_KUTTA_CPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "RungeKutta.h"

using namespace std;

RungeKutta::RungeKutta (const int &n){
    neqn = n;
    nombre = "Runge-Kutta";
}

void RungeKutta::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf){
    double *Yn = new double [neqn], // vector con la Y en cada iteracion
        *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar
    
    vectorCopia(Y0, Yn); // Y0 -> Yn, para primera iteración
    
    #pragma omp parallel
    {   
        for(double tn=t0; tn<tf; tn+=h){ // Todas las llamadas han de hacerse de forma secuencial, ya que cada una necesita de la anterior            
            // K1 = feval(tn, Yn)
            problema->feval(tn,Yn,K1);              // Definimos K1

            // K2 = feval(tn + h/2, Yn + K1*h/2)
            vectorCopia(Yn, Yaux);                  // Yn -> Yaux
            escalarPorVector(0.5*h, K1, Yaux);      // Yaux += K1*h/2
            problema->feval(tn + 0.5*h, Yaux, K2);  // Definimos K2

            // K3 = feval(tn + h/2, Yn + K2*h/2)
            vectorCopia(Yn, Yaux);                  // Yn -> Yaux
            escalarPorVector(0.5*h, K2, Yaux);      // Yaux += K2*h/2
            problema->feval(tn + 0.5*h, Yaux, K3);  // Definimos K3

            // K4 = feval(tn + h, Y0 + h*K3)
            vectorCopia(Yn, Yaux);                  // Yn -> Yaux
            escalarPorVector(h, K3, Yaux);          // Yaux += h*K3
            problema->feval(tn+h, Yaux, K4);        // Definimos K4

            // Yn+1 = Yn + K1*h/6 + K2*h/3 + K3*h/3 + K4*h/6
            escalarPorVector(h/6.0, K1, Yn);     // Yn += K1*h/6
            escalarPorVector(h/3.0, K2, Yn);     // Yn += K2*h/3
            escalarPorVector(h/3.0, K3, Yn);     // Yn += K3*h/3
            escalarPorVector(h/6.0, K4, Yn);     // Yn += K4*h/6
            //#pragma omp barrier
            // Tras las sumas, el vector Yn ahora contiene Yn+1
        }
    }
    // Yn es el valor que arrastramos de la ultima iteracion
    vectorCopia(Yn, Yf); // Yn -> Yf
    delete[] K1;
    delete[] K2;
    delete[] K3;
    delete[] K4;
    delete[] Yn;
    delete[] Yaux;
}

void RungeKutta::aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf){
    double *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar
      
    // K1 = feval(tn, Yn)
    problema->feval(t0,Y0,K1);              // Definimos K1

    // K2 = feval(tn + h/2, Yn + K1*h/2)
    vectorCopia(Y0, Yaux);                  // Yn -> Yaux
    escalarPorVector(0.5*h, K1, Yaux);      // Yaux += K1*h/2
    problema->feval(t0 + 0.5*h, Yaux, K2);  // Definimos K2

    // K3 = feval(tn + h/2, Yn + K2*h/2)
    vectorCopia(Y0, Yaux);                  // Yn -> Yaux
    escalarPorVector(0.5*h, K2, Yaux);      // Y1 += K2*h/2
    problema->feval(t0 + 0.5*h, Yaux, K3);  // Definimos K3

    // K4 = feval(tn + h, Y0 + h*K3)
    vectorCopia(Y0, Yaux);                  // Yn -> Yaux
    escalarPorVector(h, K3, Yaux);          // Y1 += h*K3
    problema->feval(t0+h, Yaux, K4);        // Definimos K4

    // Yn+1 = Yn + K1*h/6 + K2*h/3 + K3*h/3 + K4*h/6
    vectorCopia(Y0, Yf);                    // Y0->Yf
    escalarPorVector(h/6.0, K1, Yf);        // Yf += K1*h/6
    escalarPorVector(h/3.0, K2, Yf);        // Yf += K2*h/3
    escalarPorVector(h/3.0, K3, Yf);        // Yf += K3*h/3
    escalarPorVector(h/6.0, K4, Yf);        // Yf += K4*h/6
        
    delete[] K1;
    delete[] K2;
    delete[] K3;
    delete[] K4;
    delete[] Yaux;
}


#endif