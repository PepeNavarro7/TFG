#ifndef ADAMS_MOULTON_CPP
#define ADAMS_MOULTON_CPP

#include "AdamsMoulton.h"
#include <iostream>

using namespace std;


// Constructor de la clase
AdamsMoulton::AdamsMoulton (const int &n, RungeKutta* r){
    neqn = n;
    ptr_runge = r;
    nombre = "Adams-Moulton";
}

// Aplicar Adams-Bashford el numero necesario de veces
void AdamsMoulton::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf){
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn];
    double t = t0;
    vectorCopia(Y0, Yn0);
    
    // Aplicamos 4 veces Runge-Kutta para obtener los primeros pasos
    ptr_runge->aplicarUnidad(problema,t,h,Yn0,Yn1);
    t+=h;
    ptr_runge->aplicarUnidad(problema,t,h,Yn1,Yn2);
    t+=h;
    ptr_runge->aplicarUnidad(problema,t,h,Yn2,Yn3);
    t+=h;
    ptr_runge->aplicarUnidad(problema,t,h,Yn3,Yn4);
    t+=h;
    
    // Ahora aplicamos Adams-Bahsford
    const double h_aux=h/720.0;
    //vectorCopia(Yn4,Yn5); // Yn4 -> Yn5
    #pragma omp parallel shared(h_aux)
    {
        for(double tn=t; tn<tf; tn+=h){


            // Ahora que tenemos Yn5, convertimos todos los Yn en Yn-1 para hacer la siguiente iteracion
            vectorCopia(Yn1,Yn0); // Yn1 -> Yn0
  
        }
    }

    //vectorCopia(Yn5,Yf);
    delete[] Yn0;
    delete[] Yn1;
    delete[] Yn2;
    delete[] Yn3;
    delete[] Yn4;
}  

#endif