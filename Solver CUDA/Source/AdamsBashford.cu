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

void AdamsBashford::set_threads(const int &t) { 
    THREADSPERBLOCK=t; 
    NUM_BLOCKS = ceil((double)neqn/THREADSPERBLOCK); 
    NUM_BYTES = sizeof(double) * neqn;
    ptr_runge->set_threads(t);
}

// Aplicar Adams-Bashford ORDEN 5
/*
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
        
        // Ahora aplicamos Adams-Bashford
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
            escalarPorVector(h_aux*(-1274.0), Yaux,Yn5);    // Yn5 += Yaux*(h*-1274/720)

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
*/

// Aplicar Adams-Bashford ORDEN 4
void AdamsBashford::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf){
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
    int it=0;
    cudaMemcpy(Yn4, Yn3, NUM_BYTES, cudaMemcpyDeviceToDevice); // Yn3 -> Yn4
    for(double tn = t0+h*3; tn<tf; tn+=h, ++it){
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
}  

// Aplicar Adams-Bashford una unica vez
/*void AdamsBashford::aplicarUnidad(Problema* problema, const double &t0, const double &h, const double *Y0, double *Yn4){
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], // Vectores intermedios
        *Yaux = new double [neqn]; // vector auxiliar
    
    vectorCopia(Y0, Yn0); // Definimos Yn0
    
    // Aplicamos 3 veces Runge-Kutta para obtener los primeros pasos
    ptr_runge->aplicarUnidad(problema,t0,h,Yn0,Yn1);        // Obtenemos Yn1 con RK
    ptr_runge->aplicarUnidad(problema,t0+h,h,Yn1,Yn2);      // Obtenemos Yn2 con RK
    ptr_runge->aplicarUnidad(problema,t0+h*2,h,Yn2,Yn3);    // Obtenemos Yn3 con RK
    
    // Ahora aplicamos Adams-Bashford de Orden 4 para obtener Yn4
    const double h_aux=h/24.0;

    vectorCopia(Yn3,Yn4); // Yn3 -> Yn4

    problema->feval(t0+h*3, Yn3, Yaux);             // f(tn3,Yn3) -> Yaux
    escalarPorVector(h_aux*(55.0), Yaux,Yn4);       // Yn4 += Yaux*(h*55/24)

    problema->feval(t0+h*2, Yn2, Yaux);             // f(tn2,Yn2) -> Yaux
    escalarPorVector(h_aux*(-59.0), Yaux, Yn4);     // Yn4 += Yaux*(h*-59/24)

    problema->feval(t0+h,Yn1,Yaux);                 // f(tn1,Yn1) -> Yaux
    escalarPorVector(h_aux*37.0, Yaux, Yn4);        // Yn4 += Yaux*(h*37/24)

    problema->feval(t0, Yn0, Yaux);                 // f(tn0,Yn0) -> Yaux
    escalarPorVector(h_aux*(-9.0), Yaux, Yn4);      // Yn4 += Yaux*(h*-9/24)        
    
    delete[] Yn0;
    delete[] Yn1;
    delete[] Yn2;
    delete[] Yn3;
    delete[] Yaux;
}*/

// Aplicar Adams-Bashford una unica vez sin usar Runge Kutta dentro
void AdamsBashford::aplicarUnidadSinRK(Problema* problema, const double &t0, const double &h, const double *Yn0, const double *Yn1, const double *Yn2, const double *Yn3, double *Yn4){
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