#ifndef RUNGE_KUTTA_I_CPP
#define RUNGE_KUTTA_I_CPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "RungeKutta_i.h"
#include <string>

using namespace std;

// Aplicar Runge-Kutta el numero necesario de veces
void RungeKutta_i::aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const {
    const int neqn = get_neqn();
    double *Yn = new double [neqn], // vector con la Y en cada iteracion
        *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar    

    vectorCopia(Y0, Yn); // Y0 -> Yn, para la primera iteración    
               
    const double h2=h/2.0, h_n=-1.0*h, h_2=2.0*h, h6=h/6.0;
    // Todas las llamadas han de hacerse de forma secuencial, ya que cada una necesita de la anterior   
    switch(get_orden()){
    case 1: 
    #pragma omp parallel default(none) shared (problema, Yn, K1, t0, tf, h, neqn)
    {
        for(double tn=t0; tn<tf; tn+=h){
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1
                // Yn+1 = Yn + h*K1
                Yn[i] += h * K1[i]; // Tras las sumas, el vector Yn ahora contiene Yn+1
            } // Barrera implícita
        }
    }
    break;
    case 2: // Heun
    #pragma omp parallel default(none) shared (problema, Yn, Yaux, K1, K2, t0, tf, h, h2, neqn)
    {
        for(double tn=t0; tn<tf; tn+=h){
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1

                // K2 = feval(tn + h, Yn + h*K1) 
                Yaux[i] = Yn[i] + h * K1[i]; // Yaux = Yn + h*K1
            } // Barrera implícita
                
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                K2[i] = problema->feval_i(tn + h, Yaux, i); // Definimos K2

                // Yn+1 = Yn + h/2 * (K1 + K2)
                Yn[i] += h2 * ( K1[i] + K2[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
            } // Barrera implícita
        }
    }
        
    break;
    case 3: 
    #pragma omp parallel default(none) shared (problema, Yn, Yaux, K1, K2, K3, t0, tf, h, h2, h_2, h_n, h6, neqn)
    {
        for(double tn=t0; tn<tf; tn+=h){
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1

                // K2 = feval(tn + h/2, Yn + h/2 * K1) 
                Yaux[i] = Yn[i] + h2 * K1[i]; // Yaux = Yn + h/2 * K1
            } // Barrera implícita
                
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                K2[i] = problema->feval_i(tn + h2, Yaux, i); // Definimos K2

                // K3 = feval(tn + h, Yn -h*K1 + 2*h*K2 )    
                Yaux[i] = Yn[i] + h_n * K1[i] + h_2 * K2[i]; // Yaux = Yn - h*K1 +2*h*K2
            } // Barrera implícita
                
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                K3[i] = problema->feval_i(tn + h, Yaux, i);// Definimos K3

                // Yn+1 = Yn + h/6 * (K1 + 4*K2 + K3)
                Yn[i] += h6 * ( K1[i] + 4.0*K2[i] + K3[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
            } // Barrera implícita
        }
    }
        
    break;
    case 4: 
    #pragma omp parallel default(none) shared (problema, Yn, Yaux, K1, K2, K3, K4, t0, tf, h, h6, h2, neqn)
    {
        for(double tn=t0; tn<tf; tn+=h){
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                // K1 = feval(tn, Yn)
                K1[i] = problema->feval_i(tn, Yn, i); // Definimos K1

                // K2 = feval(tn + h/2, Yn + K1*h/2) 
                Yaux[i] = Yn[i] + h2 * K1[i]; // Yaux = Yn + K1*h/2
            } // Barrera implícita
                
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                K2[i] = problema->feval_i(tn + h2, Yaux, i); // Definimos K2

                // K3 = feval(tn + h/2, Yn + K2*h/2)    
                Yaux[i] = Yn[i] + h2 * K2[i]; // Yaux = Yn + K2*h/2
            } // Barrera implícita
                
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i) {
                K3[i] = problema->feval_i(tn + h2, Yaux, i);// Definimos K3

                // K4 = feval(tn + h, Yn + h*K3)
                Yaux[i] = Yn[i] + h * K3[i]; // Yaux = Yn + h*K3
            } // Barrera implícita
                
            #pragma omp for schedule(static)
            for (int i = 0; i < neqn; ++i){
                K4[i] = problema->feval_i(tn + h, Yaux, i);// Definimos K4

                // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
                Yn[i] += h6 * ( K1[i] + 2.0*K2[i] + 2.0*K3[i] + K4[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
            } // Barrera implícita
        }
    } 
    break; 
    } // Fin del switch

    // Yn es el valor que arrastramos de la ultima iteracion
    vectorCopia(Yn, Yf);

    delete[] Yn;
    delete[] K1; delete[] K2; delete[] K3; delete[] K4;
    delete[] Yaux;
}

// Aplicar Runge-Kutta ORDEN4 una unica vez
void RungeKutta_i::aplicarUnidad(const Problema* problema, const double &t0, const double &h, const double* __restrict Y0, double* __restrict Yf) const {
    const int neqn = get_neqn();
    double *K1 = new double[neqn], *K2 = new double[neqn], *K3 = new double[neqn], *K4 = new double[neqn], // Vectores de cada paso
        *Yaux = new double[neqn]; // Vector auxiliar
    
    const double h2=h/2.0, h6=h/6.0;
    #pragma omp parallel default(none) shared (problema, Yaux, Y0, Yf, K1, K2, K3, K4, t0, h, h6, h2, neqn)
    {    
        #pragma omp for schedule(static)
        for (int i = 0; i < neqn; ++i) {
            // K1 = feval(tn, Yn)
            K1[i] = problema->feval_i(t0, Y0, i); // Definimos K1

            // K2 = feval(tn + h/2, Yn + K1*h/2) 
            Yaux[i] = Y0[i] + h2 * K1[i]; // Yaux = Yn + K1*h/2
        } // Barrera implícita
            
        #pragma omp for schedule(static)
        for (int i = 0; i < neqn; ++i) {
            K2[i] = problema->feval_i(t0 + h2, Yaux, i); // Definimos K2

            // K3 = feval(tn + h/2, Yn + K2*h/2)    
            Yaux[i] = Y0[i] + h2 * K2[i]; // Yaux = Yn + K2*h/2
        } // Barrera implícita
             
        #pragma omp for schedule(static)
        for (int i = 0; i < neqn; ++i) {
            K3[i] = problema->feval_i(t0 + h2, Yaux, i);// Definimos K3

            // K4 = feval(tn + h, Yn + h*K3)
            Yaux[i] = Y0[i] + h * K3[i]; // Yaux = Yn + h*K3
        } // Barrera implícita
            
        #pragma omp for schedule(static)
        for (int i = 0; i < neqn; ++i){
            K4[i] = problema->feval_i(t0 + h, Yaux, i);// Definimos K4

            // Yn+1 = Yn + h/6 * (K1 + 2*K2 + 2*K3 + K4)
            Yf[i] = Y0[i] + h6 * ( K1[i] + 2.0*K2[i] + 2.0*K3[i] + K4[i] ) ; // Tras las sumas, el vector Yn ahora contiene Yn+1
        } // Barrera implícita
    } // Se cierra el parallel
        
    delete[] K1;
    delete[] K2;
    delete[] K3;
    delete[] K4;
    delete[] Yaux;
}


#endif