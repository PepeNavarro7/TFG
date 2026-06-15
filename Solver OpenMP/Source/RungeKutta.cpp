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
void RungeKutta::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn = new double [neqn], // vector con la Y en cada iteracion
        *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar    

    #pragma omp parallel
    {    
        #pragma omp for // Y0 -> Yn, para la primera iteración
        for (int i = 0; i < neqn; ++i)
            Yn[i] = Y0[i];
        
        for(double tn=t0; tn<tf; tn+=h){ // Todas las llamadas han de hacerse de forma secuencial, ya que cada una necesita de la anterior   
            // K1 = feval(tn, Yn)
            #pragma omp for
            for (int i = 0; i < neqn; ++i) // Definimos K1
                K1[i] = problema->feval_i(tn, Yn, i);

            // K2 = feval(tn + h/2, Yn + K1*h/2) 
            #pragma omp for
            for (int i = 0; i < neqn; ++i) // Yaux = Yn + K1*h/2
                Yaux[i] = Yn[i] + 0.5 * h * K1[i];

            #pragma omp for
            for (int i = 0; i < neqn; ++i) // Definimos K2
                K2[i] = problema->feval_i(tn + 0.5*h, Yaux, i);

            // K3 = feval(tn + h/2, Yn + K2*h/2)    
            #pragma omp for
            for (int i = 0; i < neqn; ++i) // Yaux = Yn + K2*h/2
                Yaux[i] = Yn[i] + 0.5 * h * K2[i];

            #pragma omp for
            for (int i = 0; i < neqn; ++i) // Definimos K3
                K3[i] = problema->feval_i(tn + 0.5*h, Yaux, i);

            // K4 = feval(tn + h, Yn + h*K3)
            #pragma omp for
            for (int i = 0; i < neqn; ++i) // Yaux = Yn + h*K3
                Yaux[i] = Yn[i] + h * K3[i];

            #pragma omp for
            for (int i = 0; i < neqn; ++i) // Definimos K4
                K4[i] = problema->feval_i(tn + h, Yaux, i);


            // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
            #pragma omp for
            for (int i = 0; i < neqn; ++i)
                Yn[i] = Yn[i] + (h/6.0) * ( K1[i] + 2*K2[i] + 2*K3[i] + K4[i] ) ;
            // Tras las sumas, el vector Yn ahora contiene Yn+1
        }
        
        // Yn es el valor que arrastramos de la ultima iteracion
        #pragma omp for
        for (int i = 0; i < neqn; ++i)
            Yf[i] = Yn[i];
    } // Se cierra el parallel

    delete[] Yn;
    delete[] K1;
    delete[] K2;
    delete[] K3;
    delete[] K4;
    delete[] Yaux;
}

// Aplicar Runge-Kutta una unica vez
void RungeKutta::aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf) const {
    double *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar
    
    #pragma omp parallel
    {
        // K1 = feval(t0, Y0)
        #pragma omp for
        for (int i = 0; i < neqn; ++i) // Definimos K1
            K1[i] = problema->feval_i(t0, Y0, i);

        // K2 = feval(t0 + h/2, Y0 + K1*h/2) 
        #pragma omp for
        for (int i = 0; i < neqn; ++i) // Yaux = Y0 + K1*h/2
            Yaux[i] = Y0[i] + 0.5 * h * K1[i];

        #pragma omp for
        for (int i = 0; i < neqn; ++i) // Definimos K2
            K2[i] = problema->feval_i(t0 + 0.5*h, Yaux, i);

        // K3 = feval(t0 + h/2, Y0 + K2*h/2)    
        #pragma omp for
        for (int i = 0; i < neqn; ++i) // Yaux = Y0 + K2*h/2
            Yaux[i] = Y0[i] + 0.5 * h * K2[i];

        #pragma omp for
        for (int i = 0; i < neqn; ++i) // Definimos K3
            K3[i] = problema->feval_i(t0 + 0.5*h, Yaux, i);

        // K4 = feval(t0 + h, Y0 + h*K3)
        #pragma omp for
        for (int i = 0; i < neqn; ++i) // Yaux = Yn + h*K3
            Yaux[i] = Y0[i] + h * K3[i];

        #pragma omp for
        for (int i = 0; i < neqn; ++i) // Definimos K4
            K4[i] = problema->feval_i(t0 + h, Yaux, i);

        // Yf = Y0 + h/6 * (K1 + 2*K2 + 2*K3 + K4)
        #pragma omp for
        for (int i = 0; i < neqn; ++i)
            Yf[i] = Y0[i] + (h/6.0) * ( K1[i] + 2*K2[i] + 2*K3[i] + K4[i] ) ;
    }
        
    delete[] K1;
    delete[] K2;
    delete[] K3;
    delete[] K4;
    delete[] Yaux;
}


#endif