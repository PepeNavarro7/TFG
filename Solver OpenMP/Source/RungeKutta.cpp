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
void RungeKutta::aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn = new double [neqn], // vector con la Y en cada iteracion
        *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar    

    #pragma omp parallel
    {    
        #pragma omp for // Y0 -> Yn, para la primera iteración
        for (int i = 0; i < neqn; ++i)
            Yn[i] = Y0[i];
        
        for(double tn=t0; tn<tf; tn+=h){ // Todas las llamadas han de hacerse de forma secuencial, ya que cada una necesita de la anterior   
            switch(orden){
            case 1: 
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    // K1 = feval(tn, Yn)
                    K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1
                }
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {    
                    // Yn+1 = Yn + h*K1
                    Yn[i] = Yn[i] + h * K1[i]; // Tras las sumas, el vector Yn ahora contiene Yn+1
                } // Barrera implícita
            break;
            case 2: // Heun
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    // K1 = feval(tn, Yn)
                    K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1

                    // K2 = feval(tn + h, Yn + h*K1) 
                    Yaux[i] = Yn[i] + h * K1[i]; // Yaux = Yn + h*K1
                } // Barrera implícita
                    
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    K2[i] = problema->feval_i(tn + h, Yaux, i); // Definimos K2

                    // Yn+1 = Yn + h/2 * (K1 + K2)
                    Yn[i] = Yn[i] + (h/2.0) * ( K1[i] + K2[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
                } // Barrera implícita
            break;
            case 3: 
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    // K1 = feval(tn, Yn)
                    K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1

                    // K2 = feval(tn + h/2, Yn + h/2 * K1) 
                    Yaux[i] = Yn[i] + 0.5 * h * K1[i]; // Yaux = Yn + h/2 * K1
                } // Barrera implícita
                    
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    K2[i] = problema->feval_i(tn + 0.5*h, Yaux, i); // Definimos K2

                    // K3 = feval(tn + h, Yn -h*K1 + 2*h*K2 )    
                    Yaux[i] = Yn[i] - h * K1[i] + 2 * h * K2[i]; // Yaux = Yn - h*K1 +2*h*K2
                } // Barrera implícita
                    
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    K3[i] = problema->feval_i(tn + h, Yaux, i);// Definimos K3

                    // Yn+1 = Yn + h/6 * (K1 + 4*K2 + K3)
                    Yn[i] = Yn[i] + (h/6.0) * ( K1[i] + 4*K2[i] + K3[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
                } // Barrera implícita
            break;
            case 4: 
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    // K1 = feval(tn, Yn)
                    K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1

                    // K2 = feval(tn + h/2, Yn + K1*h/2) 
                    Yaux[i] = Yn[i] + 0.5 * h * K1[i]; // Yaux = Yn + K1*h/2
                } // Barrera implícita
                    
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    K2[i] = problema->feval_i(tn + 0.5*h, Yaux, i); // Definimos K2

                    // K3 = feval(tn + h/2, Yn + K2*h/2)    
                    Yaux[i] = Yn[i] + 0.5 * h * K2[i]; // Yaux = Yn + K2*h/2
                } // Barrera implícita
                    
                #pragma omp for
                for (int i = 0; i < neqn; ++i) {
                    K3[i] = problema->feval_i(tn + 0.5*h, Yaux, i);// Definimos K3

                    // K4 = feval(tn + h, Yn + h*K3)
                    Yaux[i] = Yn[i] + h * K3[i]; // Yaux = Yn + h*K3
                } // Barrera implícita
                    
                #pragma omp for
                for (int i = 0; i < neqn; ++i){
                    K4[i] = problema->feval_i(tn + h, Yaux, i);// Definimos K4


                    // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
                    Yn[i] = Yn[i] + (h/6.0) * ( K1[i] + 2*K2[i] + 2*K3[i] + K4[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
                } // Barrera implícita
            break; 
            } // Fin del switch
        } // Fin del bucle for
        
        // Yn es el valor que arrastramos de la ultima iteracion
        #pragma omp for
        for (int i = 0; i < neqn; ++i)
            Yf[i] = Yn[i];
    } // Se cierra el parallel

    delete[] Yn;
    delete[] K1; delete[] K2; delete[] K3; delete[] K4;
    delete[] Yaux;
}

// Aplicar Runge-Kutta una unica vez
void RungeKutta::aplicarUnidad(const Problema* problema, const double &t0, const double &h, const double *Y0, double *Yf) const {
    double *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar
    
    #pragma omp parallel
    {    
        switch(orden){
        case 1: 
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(t0, Y0, i); // Definimos K1
            }
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {    
                // Yn+1 = Yn + h*K1
                Yf[i] = Y0[i] + h * K1[i];
            } // Barrera implícita
        break;
        case 2: // Heun
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(t0, Y0, i); // Definimos K1

                // K2 = feval(tn + h, Yn + h*K1) 
                Yaux[i] = Y0[i] + h * K1[i]; // Yaux = Yn + h*K1
            } // Barrera implícita
                
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                K2[i] = problema->feval_i(t0 + h, Yaux, i); // Definimos K2

                // Yn+1 = Yn + h/2 * (K1 + K2)
                Yf[i] = Y0[i] + (h/2.0) * ( K1[i] + K2[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
            } // Barrera implícita
        break;
        case 3: 
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(t0, Y0, i); // Definimos K1

                // K2 = feval(tn + h/2, Yn + h/2 * K1) 
                Yaux[i] = Y0[i] + 0.5 * h * K1[i]; // Yaux = Yn + h/2 * K1
            } // Barrera implícita
                
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                K2[i] = problema->feval_i(t0 + 0.5*h, Yaux, i); // Definimos K2

                // K3 = feval(tn + h, Yn -h*K1 + 2*h*K2 )    
                Yaux[i] = Y0[i] - h * K1[i] + 2 * h * K2[i]; // Yaux = Yn - h*K1 +2*h*K2
            } // Barrera implícita
                
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                K3[i] = problema->feval_i(t0 + h, Yaux, i);// Definimos K3

                // Yn+1 = Yn + h/6 * (K1 + 4*K2 + K3)
                Yf[i] = Y0[i] + (h/6.0) * ( K1[i] + 4*K2[i] + K3[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
            } // Barrera implícita
        break;
        case 4: 
        default:
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(t0, Y0, i); // Definimos K1

                // K2 = feval(tn + h/2, Yn + K1*h/2) 
                Yaux[i] = Y0[i] + 0.5 * h * K1[i]; // Yaux = Yn + K1*h/2
            } // Barrera implícita
                
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                K2[i] = problema->feval_i(t0 + 0.5*h, Yaux, i); // Definimos K2

                // K3 = feval(tn + h/2, Yn + K2*h/2)    
                Yaux[i] = Y0[i] + 0.5 * h * K2[i]; // Yaux = Yn + K2*h/2
            } // Barrera implícita
                
            #pragma omp for
            for (int i = 0; i < neqn; ++i) {
                K3[i] = problema->feval_i(t0 + 0.5*h, Yaux, i);// Definimos K3

                // K4 = feval(tn + h, Yn + h*K3)
                Yaux[i] = Y0[i] + h * K3[i]; // Yaux = Yn + h*K3
            } // Barrera implícita
                
            #pragma omp for
            for (int i = 0; i < neqn; ++i){
                K4[i] = problema->feval_i(t0 + h, Yaux, i);// Definimos K4


                // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
                Yf[i] = Y0[i] + (h/6.0) * ( K1[i] + 2*K2[i] + 2*K3[i] + K4[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
            } // Barrera implícita
        break; 
        } // Fin del switch
    } // Se cierra el parallel
        
    delete[] K1;
    delete[] K2;
    delete[] K3;
    delete[] K4;
    delete[] Yaux;
}


#endif