#ifndef ADAMS_BASHFORD_CPP
#define ADAMS_BASHFORD_CPP

#include "AdamsBashford.h"
#include <iostream>

using namespace std;


// Constructor de la clase
AdamsBashford::AdamsBashford (const int &n, RungeKutta* r){
    neqn = n;
    ptr_runge = r;
    nombre = "Adams-Bashford";
}

// Aplicar Adams-Bashford el numero necesario de veces
void AdamsBashford::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf){
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], *Yn5 = new double[neqn]; // Vectores intermedios
    double *Yaux = new double [neqn]; // vector auxiliar
    
    #pragma omp parallel
    {
        double tn = t0;
        vectorCopia(Y0, Yn0);
        
        // Aplicamos 4 veces Runge-Kutta para obtener los primeros pasos
        ptr_runge->aplicarUnidad(problema,tn,h,Yn0,Yn1);
        tn+=h; // tn=t1
        ptr_runge->aplicarUnidad(problema,tn,h,Yn1,Yn2);
        tn+=h; // tn=t2
        ptr_runge->aplicarUnidad(problema,tn,h,Yn2,Yn3);
        tn+=h; // tn=t3
        ptr_runge->aplicarUnidad(problema,tn,h,Yn3,Yn4);
        tn+=h;  //tn=t4
        
        // Ahora aplicamos Adams-Bahsford
        const double h_aux=h/720.0;
    
        vectorCopia(Yn4,Yn5); // Yn4 -> Yn5
        for(; tn<tf; tn+=h){
            problema->feval(tn, Yn4, Yaux);                 // f(tn4,Yn4) -> Yaux
            escalarPorVector(h_aux*1901.0, Yaux,Yn5);       // Yn5 += Yaux*(h*1901/720)

            problema->feval(tn-h, Yn3, Yaux);               // f(tn3,Yn3) -> Yaux
            escalarPorVector(h_aux*(-2774.0), Yaux,Yn5);    // Yn5 += Yaux*(h*-2774/720)

            problema->feval(tn-h*2, Yn2, Yaux);             // f(tn2,Yn2) -> Yaux
            escalarPorVector(h_aux*2616.0, Yaux, Yn5);      // Yn5 += Yaux*(h*2616/720)

            problema->feval(tn-h*3,Yn1,Yaux);               // f(tn1,Yn1) -> Yaux
            escalarPorVector(h_aux*(-1274.4), Yaux,Yn5);    // Yn5 += Yaux*(h*-1274/720)

            problema->feval(tn-h*4, Yn0, Yaux);             // f(tn0,Yn0) -> Yaux
            escalarPorVector(h_aux*251.0, Yaux, Yn5);       // Yn5 += Yaux*(h*251/720)

            // Ahora que tenemos Yn5, convertimos todos los Yn en Yn-1 para hacer la siguiente iteracion
            vectorCopia(Yn1,Yn0); // Yn1 -> Yn0
            vectorCopia(Yn2,Yn1); // Yn2 -> Yn1
            vectorCopia(Yn3,Yn2); // Yn3 -> Yn2
            vectorCopia(Yn4,Yn3); // Yn4 -> Yn3
            vectorCopia(Yn5,Yn4); // Yn5 -> Yn4
        }
    }   

    vectorCopia(Yn5,Yf);
    delete[] Yn0;
    delete[] Yn1;
    delete[] Yn2;
    delete[] Yn3;
    delete[] Yn4;
    delete[] Yn5;
    delete[] Yaux;
}  

#endif