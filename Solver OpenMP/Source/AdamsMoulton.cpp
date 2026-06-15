#ifndef ADAMS_MOULTON_CPP
#define ADAMS_MOULTON_CPP

#include "AdamsMoulton.h"
#include <iostream>
#include <omp.h>
#include <string>

using namespace std;

// Aplicar Adams-Moulton el numero necesario de veces
void AdamsMoulton::aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], // Vectores intermedios
           *YF0 = new double[neqn], *YF1 = new double[neqn], *YF2 = new double[neqn], *YF3 = new double[neqn], // Vectores feval
           *Yn4_AB = new double [neqn], *YF4_AB = new double [neqn],    // Vectores para aproximar Yn4 usando Adams-Bashford
           *Yn4_AM = new double [neqn], *YF4_AM = new double [neqn];    // Vectores para aproximar Yn4 usando Adams-Moulton
    const double h720=h/720.0, h_RK = h/100.0, h24=h/24.0;
    
    vectorCopia(Y0, Yn0, neqn);   // Definimos Yn0
        
    // Aplicamos 3 veces Runge-Kutta para obtener los primeros pasos
    ptr_runge->aplicarUnidad(problema, t0,          h, Yn0, Yn1);        //Obtenemos Yn1 con RK
    ptr_runge->aplicarUnidad(problema, t0+h_RK,     h, Yn1, Yn2);        //Obtenemos Yn2 con RK
    ptr_runge->aplicarUnidad(problema, t0+h_RK*2.0, h, Yn2, Yn3);        //Obtenemos Yn3 con RK
    
    #pragma omp parallel shared(h720)
    {
        #pragma omp for // Creamos los vectores feval
        for (int i = 0; i < neqn; ++i){
            YF0[i] = problema->feval_i(t0,          Yn0, i); // f(tn0,Yn0) -> YF0
            YF1[i] = problema->feval_i(t0+h_RK,     Yn1, i); // f(tn1,Yn1) -> YF1
            YF2[i] = problema->feval_i(t0+h_RK*2.0, Yn2, i); // f(tn2,Yn2) -> YF2
            YF3[i] = problema->feval_i(t0+h_RK*3.0, Yn3, i); // f(tn3,Yn3) -> YF3
        }
        for (double tn=t0+h_RK*3; tn<tf; tn+=h){
            // Aplicamos Adams-Bashford para obtener una primera aproximación de Yn4 -> Yn4_AB
            #pragma omp for 
            for (int i = 0; i < neqn; ++i)
                Yn4_AB[i] = Yn3[i] + h24 * (55.0*YF3[i] - 59.0*YF2[i] + 37.0*YF1[i] - 9.0*YF0[i]);            
            
            #pragma omp for // Obtenemos el feval de Yn4_AB
            for (int i = 0; i < neqn; ++i)
                YF4_AB[i] = problema->feval_i(tn+h, Yn4_AB, i); // f(tn4, Yn4_AB) -> YF4_AB
            
            // Aplicamos Adams-Moulton con Yn4_AB para obtener una segunda aproximación de Yn4 -> Yn4_AM
            #pragma omp for 
            for (int i = 0; i < neqn; ++i)
                Yn4_AM[i] = Yn3[i] + h720 * (251.0*YF4_AB[i] + 646.0*YF3[i] - 264.0*YF2[i] + 106.0*YF1[i] - 19.0*YF0[i]);

            #pragma omp for // Obtenemos el feval de Yn4_AM
            for (int i = 0; i < neqn; ++i)
                YF4_AM[i] = problema->feval_i(tn+h, Yn4_AM, i); // f(tn4, Yn4_AM) -> YF4_AM


            // Ahora que tenemos una segunda aproximación de Yn4, volveremos a aplicar Adams-Moulton con ella, para obtener el Yn4 definitivo
            #pragma omp for 
            for (int i = 0; i < neqn; ++i)
                Yn4[i] = Yn3[i] + h720 * (251.0*YF4_AM[i] + 646.0*YF3[i] - 264.0*YF2[i] + 106.0*YF1[i] - 19.0*YF0[i]);

            // Ahora que tenemos Yn4, convertimos todos los Yn en Yn-1, y los YFn en YFn-1 para hacer la siguiente iteracion
            #pragma omp single
            {
                swap(Yn1, Yn0); // Yn1 -> Yn0
                swap(Yn2, Yn1); // Yn2 -> Yn1
                swap(Yn3, Yn2); // Yn3 -> Yn2
                swap(Yn4, Yn3); // Yn4 -> Yn3
                swap(YF1, YF0); // YF1 -> YF0
                swap(YF2, YF1); // YF2 -> YF1
                swap(YF3, YF2); // YF3 -> YF2

            } // Tras los cambios, Yn4 y YF3 contienen basura y serán reescritos

            // Calculamos el nuevo YF3 usando el Yn4 recién creado (que ahora es Yn3)
            #pragma omp for 
            for (int i = 0; i < neqn; ++i)
                YF3[i] = problema->feval_i(tn+h, Yn3, i); // f(tn3,Yn3) -> YF3
        } // Fin del bucle for

        #pragma omp for 
        for (int i = 0; i < neqn; ++i)
            Yf[i] = Yn3[i];
    } // Fin del bloque parallel

    delete[] Yn0; delete[] Yn1; delete[] Yn2; delete[] Yn3; delete[] Yn4;
    delete[] YF0; delete[] YF1; delete[] YF2; delete[] YF3;
    delete[] Yn4_AB; delete[] YF4_AB;
    delete[] Yn4_AM; delete[] YF4_AM;
}
#endif