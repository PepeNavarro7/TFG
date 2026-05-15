#ifndef ADAMS_MOULTON_CPP
#define ADAMS_MOULTON_CPP

#include "AdamsMoulton.h"
#include <iostream>

using namespace std;


// Constructor de la clase
AdamsMoulton::AdamsMoulton (const int &n, RungeKutta* r, AdamsBashford* ab){
    neqn = n;
    ptr_runge = r;
    ptr_bashford = ab;
    nombre = "Adams-Moulton";
}

// Aplicar Adams-Moulton el numero necesario de veces
void AdamsMoulton::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf){
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], // Vectores intermedios
        *Yaux = new double [neqn],      // vector auxiliar
        *Yn4_AB = new double [neqn],    // Vector para aproximar Yn4 usando Adams-Bashford
        *Yn4_AM = new double [neqn];    // Vector para aproximar Yn4 usando Adams-Moulton
    
    vectorCopia(Y0, Yn0);   // Definimos Yn0
    
    // Aplicamos 3 veces Runge-Kutta para obtener los primeros pasos
    ptr_runge->aplicarUnidad(problema,t0,h,Yn0,Yn1);        //Obtenemos Yn1 con RK
    ptr_runge->aplicarUnidad(problema,t0+h,h,Yn1,Yn2);      //Obtenemos Yn2 con RK
    ptr_runge->aplicarUnidad(problema,t0+h*2,h,Yn2,Yn3);    //Obtenemos Yn3 con RK

    for (double tn=t0+h*3; tn<tf; tn+=h){
        // Aplicamos Adams-Bashford para obtener una primera aproximación de Yn4
        ptr_bashford->aplicarUnidadSinRK(problema, tn-h*3, h, Yn0, Yn1, Yn2, Yn3, Yn4_AB);
        
        const double h_aux=h/720.0;
        // Ahora aplicamos Adams-Moulton con la aproximación de f(Yn4_AB) para obtener Yn4_AM
        vectorCopia(Yn3,Yn4_AM);                                // Yn3 -> Yn4_AM

        problema->feval(tn+h, Yn4_AB, Yaux);                      // f(tn4, Yn4_AB) -> Yaux
        escalarPorVector(h_aux*(251.0), Yaux,Yn4_AM);               // Yn4_AM += Yaux*(h*251/720)

        problema->feval(tn, Yn3, Yaux);                       // f(tn3, Yn3) -> Yaux
        escalarPorVector(h_aux*(646.0), Yaux,Yn4_AM);           // Yn4_AM += Yaux*(h*646/720)

        problema->feval(tn-h, Yn2, Yaux);                     // f(tn2, Yn2) -> Yaux
        escalarPorVector(h_aux*(-264.0), Yaux, Yn4_AM);         // Yn4_AM += Yaux*(h*-264/720)

        problema->feval(tn-h*2, Yn1, Yaux);                     // f(tn1, Yn1) -> Yaux
        escalarPorVector(h_aux*(106.0), Yaux,Yn4_AM);           // Yn4_AM += Yaux*(h*106/720)

        problema->feval(tn-h*3, Yn0, Yaux);                     // f(tn0, Yn0) -> Yaux
        escalarPorVector(h_aux*(-19.0), Yaux, Yn4_AM);          // Yn4_AM += Yaux*(h*-19/720)

        // Ahora que tenemos una segunda aproximación de Yn4, volveremos a aplicar Adams-Moulton con ella, para obtener el Yn4 definitivo
        vectorCopia(Yn3,Yn4);                                // Yn3 -> Yn4

        problema->feval(tn+h, Yn4_AM, Yaux);                   // f(tn4, Yn4_AM) -> Yaux
        escalarPorVector(h_aux*(251.0), Yaux, Yn4);          // Yn4 += Yaux*(h*251/720)

        problema->feval(tn, Yn3, Yaux);                    // f(tn3, Yn3) -> Yaux
        escalarPorVector(h_aux*(646.0), Yaux, Yn4);          // Yn4 += Yaux*(h*646/720)

        problema->feval(tn-h, Yn2, Yaux);                  // f(tn2, Yn2) -> Yaux
        escalarPorVector(h_aux*(-264.0), Yaux, Yn4);         // Yn4 += Yaux*(h*-264/720)

        problema->feval(tn-h*2, Yn1, Yaux);                  // f(tn1, Yn1) -> Yaux
        escalarPorVector(h_aux*(106.0), Yaux, Yn4);          // Yn4 += Yaux*(h*106/720)

        problema->feval(tn-h*3, Yn0, Yaux);                  // f(tn0, Yn0) -> Yaux
        escalarPorVector(h_aux*(-19.0), Yaux, Yn4);          // Yn4 += Yaux*(h*-19/720)

        // Una vez obtenido Yn4 correcto, pasamos a la siguiente iteración
        vectorCopia(Yn1,Yn0); // Yn1 -> Yn0
        vectorCopia(Yn2,Yn1); // Yn2 -> Yn1
        vectorCopia(Yn3,Yn2); // Yn3 -> Yn2
        vectorCopia(Yn4,Yn3); // Yn4 -> Yn3
    }
    
    

    
}
#endif