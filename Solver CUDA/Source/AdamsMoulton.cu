#ifndef ADAMS_MOULTON_CPP
#define ADAMS_MOULTON_CPP

#include "AdamsMoulton.h"
#include <iostream>
#include <string>

using namespace std;


// Constructor de la clase
AdamsMoulton::AdamsMoulton (const int &n, RungeKutta* r, AdamsBashford* ab){
    neqn = n;
    ptr_runge = r;
    ptr_bashford = ab;
    nombre = "Adams-Moulton";
}

void AdamsMoulton::set_threads(const int &t) { 
    THREADSPERBLOCK=t; 
    NUM_BLOCKS = ceil((double)neqn/THREADSPERBLOCK); 
    NUM_BYTES = sizeof(double) * neqn;
    ptr_bashford->set_threads(t);
}

// Aplicar Adams-Moulton el numero necesario de veces
void AdamsMoulton::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf){
    double *Yn0, *Yn1, *Yn2, *Yn3, *Yn4, // Vectores intermedios
        *Yaux,      // vector auxiliar
        *Yn4_AB,    // Vector para aproximar Yn4 usando Adams-Bashford
        *Yn4_AM;    // Vector para aproximar Yn4 usando Adams-Moulton
    cudaMalloc((void**)&Yn0,NUM_BYTES);
    cudaMalloc((void**)&Yn1,NUM_BYTES);
    cudaMalloc((void**)&Yn2,NUM_BYTES);
    cudaMalloc((void**)&Yn3,NUM_BYTES);
    cudaMalloc((void**)&Yn4,NUM_BYTES);
    cudaMalloc((void**)&Yaux,NUM_BYTES);
    cudaMalloc((void**)&Yn4_AB,NUM_BYTES);
    cudaMalloc((void**)&Yn4_AM,NUM_BYTES);
    
    const double h_aux=h/720.0; 
    cudaMemcpy(Yn0, Y0, NUM_BYTES, cudaMemcpyHostToDevice); // Definimos Yn0
    
    // Aplicamos 3 veces Runge-Kutta para obtener los primeros pasos
    ptr_runge->aplicarUnidad(problema,t0,    h,Yn0,Yn1);        //Obtenemos Yn1 con RK
    ptr_runge->aplicarUnidad(problema,t0+h,  h,Yn1,Yn2);        //Obtenemos Yn2 con RK
    ptr_runge->aplicarUnidad(problema,t0+h*2,h,Yn2,Yn3);        //Obtenemos Yn3 con RK

    int it=0;
    for (double tn=t0+h*4; tn<tf; tn+=h, ++it){
        // Aplicamos Adams-Bashford para obtener una primera aproximación de Yn4 -> Yn4_AB
        ptr_bashford->aplicarUnidadSinRK(problema, tn-h*4, h, Yn0, Yn1, Yn2, Yn3, Yn4_AB);
        
        // Aplicamos Adams-Moulton con Yn4_AB para obtener una segunda aproximación de Yn4 -> Yn4_AM
        cudaMemcpy(Yn4_AM, Yn3, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn3 -> Yn4_AM

        problema->feval(tn, Yn4_AB, Yaux);                      // f(tn4, Yn4_AB) -> Yaux
        escalarPorVector(h_aux*(251.0), Yaux,Yn4_AM);           // Yn4_AM += Yaux*(h*251/720)

        problema->feval(tn-h, Yn3, Yaux);                       // f(tn3, Yn3) -> Yaux
        escalarPorVector(h_aux*(646.0), Yaux,Yn4_AM);           // Yn4_AM += Yaux*(h*646/720)

        problema->feval(tn-h*2, Yn2, Yaux);                     // f(tn2, Yn2) -> Yaux
        escalarPorVector(h_aux*(-264.0), Yaux, Yn4_AM);         // Yn4_AM += Yaux*(h*-264/720)

        problema->feval(tn-h*3, Yn1, Yaux);                     // f(tn1, Yn1) -> Yaux
        escalarPorVector(h_aux*(106.0), Yaux,Yn4_AM);           // Yn4_AM += Yaux*(h*106/720)

        problema->feval(tn-h*4, Yn0, Yaux);                     // f(tn0, Yn0) -> Yaux
        escalarPorVector(h_aux*(-19.0), Yaux, Yn4_AM);          // Yn4_AM += Yaux*(h*-19/720)

        // Ahora que tenemos una segunda aproximación de Yn4, volveremos a aplicar Adams-Moulton con ella, para obtener el Yn4 definitivo
        cudaMemcpy(Yn4, Yn3, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn3 -> Yn4

        problema->feval(tn, Yn4_AM, Yaux);                      // f(tn4, Yn4_AM) -> Yaux
        escalarPorVector(h_aux*(251.0), Yaux, Yn4);             // Yn4 += Yaux*(h*251/720)

        problema->feval(tn-h, Yn3, Yaux);                       // f(tn3, Yn3) -> Yaux
        escalarPorVector(h_aux*(646.0), Yaux, Yn4);             // Yn4 += Yaux*(h*646/720)

        problema->feval(tn-h*2, Yn2, Yaux);                     // f(tn2, Yn2) -> Yaux
        escalarPorVector(h_aux*(-264.0), Yaux, Yn4);            // Yn4 += Yaux*(h*-264/720)

        problema->feval(tn-h*3, Yn1, Yaux);                     // f(tn1, Yn1) -> Yaux
        escalarPorVector(h_aux*(106.0), Yaux, Yn4);             // Yn4 += Yaux*(h*106/720)

        problema->feval(tn-h*4, Yn0, Yaux);                     // f(tn0, Yn0) -> Yaux
        escalarPorVector(h_aux*(-19.0), Yaux, Yn4);             // Yn4 += Yaux*(h*-19/720)

        // Una vez obtenido Yn4 correcto, pasamos a la siguiente iteración
        cudaMemcpy(Yn0, Yn1, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn1 -> Yn0
        cudaMemcpy(Yn1, Yn2, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn2 -> Yn1
        cudaMemcpy(Yn2, Yn3, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn3 -> Yn2
        cudaMemcpy(Yn3, Yn4, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn4 -> Yn3

        if(it%25000==0){
            cudaMemcpy(Yf, Yn4, NUM_BYTES, cudaMemcpyDeviceToHost);
            cout << "it="<< it << " tn=" << tn << " Yf[300]=" << Yf[300] << endl;
        }
    }

    cudaMemcpy(Yf, Yn4, NUM_BYTES, cudaMemcpyDeviceToHost); // Yn4 -> Yf
    cudaFree(Yn0);
    cudaFree(Yn1);
    cudaFree(Yn2);
    cudaFree(Yn3);
    cudaFree(Yn4);
    cudaFree(Yaux);
    cudaFree(Yn4_AB);
    cudaFree(Yn4_AM);
}
#endif