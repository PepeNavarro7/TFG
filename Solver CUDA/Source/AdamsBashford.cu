#ifndef ADAMS_BASHFORD_CU
#define ADAMS_BASHFORD_CU

#include "AdamsBashford.h"
#include <iostream>

using namespace std;

// Aplicar Adams-Bashford ORDEN 4
void AdamsBashford::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0, *Yn1, *Yn2, *Yn3, *Yn4, // Vectores intermedios
        *Yaux; // vector auxiliar
    cudaMalloc((void**)&Yn0,NUM_BYTES);
    cudaMalloc((void**)&Yn1,NUM_BYTES);
    cudaMalloc((void**)&Yn2,NUM_BYTES);
    cudaMalloc((void**)&Yn3,NUM_BYTES);
    cudaMalloc((void**)&Yn4,NUM_BYTES);
    cudaMalloc((void**)&Yaux,NUM_BYTES);

    cudaMemcpy(Yn0, Y0, NUM_BYTES, cudaMemcpyHostToDevice); // Y0 -> Yn0
    
    // Aplicamos 3 veces Runge-Kutta para obtener los primeros pasos
    ptr_runge->aplicarUnidad(problema,t0,h,Yn0,Yn1);
    ptr_runge->aplicarUnidad(problema,t0+h,h,Yn1,Yn2);
    ptr_runge->aplicarUnidad(problema,t0+h*2,h,Yn2,Yn3);
    
    // Ahora aplicamos Adams-Bashford de Orden 4
    const double h_aux=h/24.0;
    cudaMemcpy(Yn4, Yn3, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn3 -> Yn4
    for(double tn = t0+h*3; tn<tf; tn+=h){
        problema->feval(tn, Yn3, Yaux);                 // f(tn3,Yn3) -> Yaux
        escalarPorVector(h_aux*(55.0), Yaux,Yn4);       // Yn4 += Yaux*(h*55/24)

        problema->feval(tn-h, Yn2, Yaux);               // f(tn2,Yn2) -> Yaux
        escalarPorVector(h_aux*(-59.0), Yaux, Yn4);     // Yn4 += Yaux*(h*-59/24)

        problema->feval(tn-h*2,Yn1,Yaux);               // f(tn1,Yn1) -> Yaux
        escalarPorVector(h_aux*37.0, Yaux,Yn4);         // Yn4 += Yaux*(h*37/24)

        problema->feval(tn-h*3, Yn0, Yaux);             // f(tn0,Yn0) -> Yaux
        escalarPorVector(h_aux*(-9.0), Yaux, Yn4);      // Yn4 += Yaux*(h*-9/24)

        // Ahora que tenemos Yn4, convertimos todos los Yn en Yn-1 para hacer la siguiente iteracion
        cudaMemcpy(Yn0, Yn1, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn1 -> Yn0
        cudaMemcpy(Yn1, Yn2, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn2 -> Yn1
        cudaMemcpy(Yn2, Yn3, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn3 -> Yn2
        cudaMemcpy(Yn3, Yn4, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn4 -> Yn3        
    }
     

    cudaMemcpy(Yf, Yn4, NUM_BYTES, cudaMemcpyDeviceToHost); // Yn4 -> Yf
    cudaFree(Yn0);
    cudaFree(Yn1);
    cudaFree(Yn2);
    cudaFree(Yn3);
    cudaFree(Yn4);
    cudaFree(Yaux);
}

// Aplicar Adams-Bashford una sola vez, pero aportando los pasos intermedios de antemano
void AdamsBashford::aplicarUnidadSinRK(Problema* problema, const double &t0, const double &h, const double *Yn0, const double *Yn1, const double *Yn2, const double *Yn3, double *Yn4) const {
    double *Yaux, *Yf; // vector auxiliar para aplicar los feval
    cudaMalloc((void**)&Yaux,NUM_BYTES);
    cudaMalloc((void**)&Yf,NUM_BYTES);
    
    // Aplicamos Adams-Bashford de Orden 4 para obtener Yn4
    const double h_aux=h/24.0;

    cudaMemcpy(Yf, Yn3, NUM_BYTES, cudaMemcpyHostToDevice); // Yn3 -> Yf

    problema->feval(t0+h*3, Yn3, Yaux);             // f(tn3,Yn3) -> Yaux
    escalarPorVector(h_aux*(55.0), Yaux,Yn4);       // Yn4 += Yaux*(h*55/24)

    problema->feval(t0+h*2, Yn2, Yaux);             // f(tn2,Yn2) -> Yaux
    escalarPorVector(h_aux*(-59.0), Yaux, Yn4);     // Yn4 += Yaux*(h*-59/24)

    problema->feval(t0+h,Yn1,Yaux);                 // f(tn1,Yn1) -> Yaux
    escalarPorVector(h_aux*37.0, Yaux, Yn4);        // Yn4 += Yaux*(h*37/24)

    problema->feval(t0, Yn0, Yaux);                 // f(tn0,Yn0) -> Yaux
    escalarPorVector(h_aux*(-9.0), Yaux, Yn4);      // Yn4 += Yaux*(h*-9/24)
    
    cudaMemcpy(Yn4, Yf, NUM_BYTES, cudaMemcpyDeviceToHost); // Yf -> Yn4
    
    cudaFree(Yaux);
    cudaFree(Yf);
}
#endif