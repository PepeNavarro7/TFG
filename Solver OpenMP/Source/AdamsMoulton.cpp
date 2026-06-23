#ifndef ADAMS_MOULTON_CPP
#define ADAMS_MOULTON_CPP

#include "AdamsMoulton.h"
#include <iostream>
#include <omp.h>
#include <string>

using namespace std;

// Aplicar Adams-Moulton el numero necesario de veces
void AdamsMoulton::aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Yf) const {
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], // Vectores intermedios
           *Fn0 = new double[neqn], *Fn1 = new double[neqn], *Fn2 = new double[neqn], *Fn3 = new double[neqn], // Vectores feval
           *Y_AB = new double [neqn], *F_AB = new double [neqn],    // Vectores para aproximar Yn usando Adams-Bashford
           *Y_AM = new double [neqn], *F_AM = new double [neqn];    // Vectores para aproximar Yn usando Adams-Moulton
    
    vectorCopia(Y0, Yn0, neqn);   // Definimos Yn0
    const double h_RK = h/100.0;
    switch(orden){ // Aplicamos RK4 para obtener los primeros pasos
        case 1:
        case 2: break; // No necesitamos RK en orden 1 & 2
        case 3: // Definimos Yn1
            ptr_runge->aplicarUnidad(problema, t0, h_RK, Yn0, Yn1);
        break;
        case 4: // Definimos Yn1 & Yn2 
            ptr_runge->aplicarUnidad(problema, t0,      h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK, h_RK, Yn1, Yn2);
        break;
        case 5: // Definimos Yn1, Yn2 & Yn3 
            ptr_runge->aplicarUnidad(problema, t0,          h_RK, Yn0, Yn1);
            ptr_runge->aplicarUnidad(problema, t0+h_RK,     h_RK, Yn1, Yn2);
            ptr_runge->aplicarUnidad(problema, t0+h_RK*2.0, h_RK, Yn2, Yn3);
        break;
    } // Fin del switch de RK

    switch(orden){ // Arrancamos los feval
        case 1:
        case 2:
            #pragma omp parallel for // Definimos el vector feval Fn0
            for (int i = 0; i < neqn; ++i)
                Fn0[i] = problema->feval_i(t0, Yn0, i); // f(tn0,Yn0) -> Fn0
        break;
        case 3:
            #pragma omp parallel for // Definimos los vectores feval Fn0 & Fn1
            for (int i = 0; i < neqn; ++i){
                Fn0[i] = problema->feval_i(t0,      Yn0, i); // f(tn0,Yn0) -> Fn0
                Fn1[i] = problema->feval_i(t0+h_RK, Yn1, i); // f(tn1,Yn1) -> Fn1
            }
        break;
        case 4:
            #pragma omp parallel for // Definimos los vectores feval Fn0, Fn1 & Fn2
            for (int i = 0; i < neqn; ++i){
                Fn0[i] = problema->feval_i(t0,          Yn0, i); // f(tn0,Yn0) -> Fn0
                Fn1[i] = problema->feval_i(t0+h_RK,     Yn1, i); // f(tn1,Yn1) -> Fn1
                Fn2[i] = problema->feval_i(t0+h_RK*2.0, Yn2, i); // f(tn2,Yn2) -> Fn2
            }
        break;
        case 5:
            #pragma omp parallel for // Definimos los vectores feval Fn0, Fn1, Fn2 & Fn3
            for (int i = 0; i < neqn; ++i){
                Fn0[i] = problema->feval_i(t0,          Yn0, i); // f(tn0,Yn0) -> Fn0
                Fn1[i] = problema->feval_i(t0+h_RK,     Yn1, i); // f(tn1,Yn1) -> Fn1
                Fn2[i] = problema->feval_i(t0+h_RK*2.0, Yn2, i); // f(tn2,Yn2) -> Fn2
                Fn3[i] = problema->feval_i(t0+h_RK*3.0, Yn3, i); // f(tn3,Yn3) -> Fn3
            }
        break;
    } // Fin del switch de arranque

    const double h2=h/2.0, h12=h/12.0, h24=h/24.0, h720=h/720.0;
    switch(orden){
        case 1: // Yn+1 = Yn + h * Fn+1
            #pragma omp parallel shared (h)
            {
                for (double tn=t0; tn<tf; tn+=h){
                    #pragma omp for // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn1 -> Y_AB
                    for (int i = 0; i < neqn; ++i) // Y_AB = Yn0 + h * Fn0
                        Y_AB[i] = Yn0[i] + h * Fn0[i];            
                    
                    #pragma omp for // Obtenemos el feval de Y_AB
                    for (int i = 0; i < neqn; ++i)
                        F_AB[i] = problema->feval_i(tn+h, Y_AB, i); // f(tn1, Y_AB) -> F_AB
                    
                    #pragma omp for // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn1 -> Y_AM
                    for (int i = 0; i < neqn; ++i) // Y_AM = Yn0 + h * F_AB
                        Y_AM[i] = Yn0[i] + h * F_AB[i];

                    #pragma omp for // Obtenemos el feval de Y_AM
                    for (int i = 0; i < neqn; ++i)
                        F_AM[i] = problema->feval_i(tn+h, Y_AM, i); // f(tn1, Y_AM) -> F_AM

                    #pragma omp for // Ahora que tenemos una 2ª aproximación de Yn1, volvemos a aplicar A-M con ella, para obtener el Yn1 definitivo
                    for (int i = 0; i < neqn; ++i) // Yn1 = Yn0 + h * F_AM
                        Yn1[i] = Yn0[i] + h * F_AM[i];

                    #pragma omp single // Ahora que tenemos Yn1, convertimos los Yn en Yn-1 para hacer la siguiente iteracion
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                    } // Tras los cambios, Yn1 contiene basura y será reescrito

                    #pragma omp for // Calculamos el nuevo Fn0 usando el Yn1 recién creado (que ahora es Yn0)
                    for (int i = 0; i < neqn; ++i)
                        Fn0[i] = problema->feval_i(tn+h, Yn0, i); // f(tn1,Yn0') -> Fn0'
                } // Fin del bucle for principal
            }    
        break;
        case 2: // Yn+1 = Yn + h/2 * (Fn+1 + Fn)
            #pragma omp parallel shared(h, h2)
            {
                for (double tn=t0; tn<tf; tn+=h){
                    #pragma omp for // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn1 -> Y_AB
                    for (int i = 0; i < neqn; ++i) // Y_AB = Yn0 + h * Fn0
                        Y_AB[i] = Yn0[i] + h * Fn0[i];            
                    
                    #pragma omp for // Obtenemos el feval de Y_AB
                    for (int i = 0; i < neqn; ++i)
                        F_AB[i] = problema->feval_i(tn+h, Y_AB, i); // f(tn1, Y_AB) -> F_AB
                    
                    #pragma omp for // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn1 -> Y_AM
                    for (int i = 0; i < neqn; ++i) // Y_AM = Yn0 + h/2 * (F_AB + Fn0)
                        Y_AM[i] = Yn0[i] + h2 * (F_AB[i] + Fn0[i]);

                    #pragma omp for // Obtenemos el feval de Y_AM
                    for (int i = 0; i < neqn; ++i)
                        F_AM[i] = problema->feval_i(tn+h, Y_AM, i); // f(tn1, Y_AM) -> F_AM

                    #pragma omp for // Ahora que tenemos una 2ª aproximación de Yn1, volvemos a aplicar A-M con ella, para obtener el Yn1 definitivo
                    for (int i = 0; i < neqn; ++i) // Yn1 = Yn0 + h/2 * (F_AM + Fn0)
                        Yn1[i] = Yn0[i] + h2 * (F_AM[i] + Fn0[i]);

                    #pragma omp single // Ahora que tenemos Yn1, convertimos los Yn en Yn-1 para hacer la siguiente iteracion
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                    } // Tras los cambios, Yn1 contiene basura y será reescrito

                    #pragma omp for // Calculamos el nuevo Fn0 usando el Yn1 recién creado (que ahora es Yn0)
                    for (int i = 0; i < neqn; ++i)
                        Fn0[i] = problema->feval_i(tn+h, Yn0, i); // f(tn1,Yn0') -> Fn0'
                } // Fin del bucle for principal
            }
        break;
        case 3: // Yn+1 = Yn + h/12 * (5Fn+1 + 8Fn - 1Fn-1)
            #pragma omp parallel shared(h2, h12)
            {
                for (double tn=t0+h_RK; tn<tf; tn+=h){
                    #pragma omp for // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn2 -> Y_AB
                    for (int i = 0; i < neqn; ++i) // Y_AB = Yn1 + h/2 * (3*Fn1 - Fn0)
                        Y_AB[i] = Yn1[i] + h2 * (3.0*Fn1[i] - 1*Fn0[i]);            
                    
                    #pragma omp for // Obtenemos el feval de Y_AB
                    for (int i = 0; i < neqn; ++i)
                        F_AB[i] = problema->feval_i(tn+h, Y_AB, i); // f(tn2, Y_AB) -> F_AB
                    
                    #pragma omp for // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn2 -> Y_AM
                    for (int i = 0; i < neqn; ++i) // Y_AM = Yn1 + h/12 * (5F_AB + 8Fn1 - 1Fn0)
                        Y_AM[i] = Yn1[i] + h12 * (5*F_AB[i] + 8.0*Fn1[i] - 1*Fn0[i]);

                    #pragma omp for // Obtenemos el feval de Y_AM
                    for (int i = 0; i < neqn; ++i)
                        F_AM[i] = problema->feval_i(tn+h, Y_AM, i); // f(tn2, Y_AM) -> F_AM

                    #pragma omp for // Ahora que tenemos una 2ª aproximación de Yn2, volvemos a aplicar A-M con ella, para obtener el Yn2 definitivo
                    for (int i = 0; i < neqn; ++i) // Yn2 = Yn1 + h/12 * (5Fn_AM + 8Fn1 - 1Fn0)
                        Yn2[i] = Yn1[i] + h12 * (5*F_AM[i] + 8.0*Fn1[i] - 1*Fn0[i]);

                    #pragma omp single // Ahora que tenemos Yn2, convertimos todos los Yn en Yn-1, y los Fn en Fn-1 para hacer la siguiente iteracion
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                        swap(Yn2, Yn1); // Yn2 -> Yn1
                        swap(Fn1, Fn0); // Fn1 -> Fn0
                    } // Tras los cambios, Yn2 & Fn1 contienen basura y serán reescritos

                    #pragma omp for // Calculamos el nuevo Fn1 usando el Yn2 recién creado (que ahora es Yn1)
                    for (int i = 0; i < neqn; ++i)
                        Fn1[i] = problema->feval_i(tn+h, Yn1, i); // f(tn2,Yn1') -> Fn1'
                } // Fin del bucle for principal
            } // Fin del parallel
        break;
        case 4: // Yn+1 = Yn + h/24 * (9Fn+1 + 19Fn - 5Fn-1 + 1Fn-2)
            #pragma omp parallel shared(h12, h24)
            {
                for (double tn=t0+h_RK*2.0; tn<tf; tn+=h){
                    #pragma omp for // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn3 -> Y_AB
                    for (int i = 0; i < neqn; ++i) // Y_AB = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
                        Y_AB[i] = Yn2[i] + h12 * (23.0*Fn2[i] - 16.0*Fn1[i] + 5.0*Fn0[i]);            
                    
                    #pragma omp for // Obtenemos el feval de Y_AB
                    for (int i = 0; i < neqn; ++i)
                        F_AB[i] = problema->feval_i(tn+h, Y_AB, i); // f(tn3, Y_AB) -> F_AB
                    
                    #pragma omp for // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn3 -> Y_AM
                    for (int i = 0; i < neqn; ++i) // Y_AM = Yn2 + h/24 * (9F_AB + 19Fn2 - 5Fn1 + 1Fn0)
                        Y_AM[i] = Yn2[i] + h24 * (9*F_AB[i] + 19.0*Fn2[i] - 5.0*Fn1[i] + 1*Fn0[i]);

                    #pragma omp for // Obtenemos el feval de Y_AM
                    for (int i = 0; i < neqn; ++i)
                        F_AM[i] = problema->feval_i(tn+h, Y_AM, i); // f(tn3, Y_AM) -> F_AM

                    #pragma omp for // Ahora que tenemos una 2ª aproximación de Yn3, volvemos a aplicar A-M con ella, para obtener el Yn3 definitivo
                    for (int i = 0; i < neqn; ++i) // Yn3 = Yn2 + h/24 * (9F_AM + 19Fn2 - 5Fn1 + 1Fn0)
                        Yn3[i] = Yn2[i] + h24 * (9.0*F_AM[i] + 19.0*Fn2[i] - 5.0*Fn1[i] + 1*Fn0[i]);

                    #pragma omp single // Ahora que tenemos Yn3, convertimos todos los Yn en Yn-1, y los Fn en Fn-1 para hacer la siguiente iteracion
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                        swap(Yn2, Yn1); // Yn2 -> Yn1
                        swap(Yn3, Yn2); // Yn3 -> Yn2
                        swap(Fn1, Fn0); // Fn1 -> Fn0
                        swap(Fn2, Fn1); // Fn2 -> Fn1
                    } // Tras los cambios, Yn3 & Fn2 contienen basura y serán reescritos

                    #pragma omp for // Calculamos el nuevo Fn2 usando el Yn3 recién creado (que ahora es Yn2)
                    for (int i = 0; i < neqn; ++i)
                        Fn2[i] = problema->feval_i(tn+h, Yn2, i); // f(tn3,Yn2') -> Fn2'
                } // Fin del bucle for principal
            } // Fin del parallel
        break;
        case 5: // Yn+1 = Yn + h/720 * (251Fn+1 + 646Fn - 264Fn-1 + 106Fn-2 - 19Fn-3)
            #pragma omp parallel shared (h24, h720)
            {
                for (double tn=t0+h_RK*3.0; tn<tf; tn+=h){
                    #pragma omp for // Aplicamos Adams-Bashford para obtener una 1ª aproximación de Yn4 -> Y_AB
                    for (int i = 0; i < neqn; ++i) // Yn4 = Yn3 + h/24 * (55Fn3 - 59Fn2 + 37Fn1 - 9Fn0)
                        Y_AB[i] = Yn3[i] + h24 * (55.0*Fn3[i] - 59.0*Fn2[i] + 37.0*Fn1[i] - 9.0*Fn0[i]);            
                    
                    #pragma omp for // Obtenemos el feval de Y_AB
                    for (int i = 0; i < neqn; ++i)
                        F_AB[i] = problema->feval_i(tn+h, Y_AB, i); // f(tn4, Y_AB) -> F_AB
                    
                    #pragma omp for // Aplicamos Adams-Moulton con F_AB para obtener una 2ª aproximación de Yn4 -> Y_AM
                    for (int i = 0; i < neqn; ++i) // Y_AM = Yn3 + h/720 * (251F_AB + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)
                        Y_AM[i] = Yn3[i] + h720 * (251.0*F_AB[i] + 646.0*Fn3[i] - 264.0*Fn2[i] + 106.0*Fn1[i] - 19.0*Fn0[i]);

                    #pragma omp for // Obtenemos el feval de Y_AM
                    for (int i = 0; i < neqn; ++i)
                        F_AM[i] = problema->feval_i(tn+h, Y_AM, i); // f(tn4, Y_AM) -> F4_AM

                    #pragma omp for // Ahora que tenemos una 2ª aproximación de Yn4, volvemos a aplicar A-M con ella, para obtener el Yn4 definitivo
                    for (int i = 0; i < neqn; ++i) // Yn4 = Yn3 + h/720 * (251F_AM + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)
                        Yn4[i] = Yn3[i] + h720 * (251.0*F_AM[i] + 646.0*Fn3[i] - 264.0*Fn2[i] + 106.0*Fn1[i] - 19.0*Fn0[i]);

                    #pragma omp single // Ahora que tenemos Yn4, convertimos todos los Yn en Yn-1, y los Fn en Fn-1 para hacer la siguiente iteracion
                    {
                        swap(Yn1, Yn0); // Yn1 -> Yn0
                        swap(Yn2, Yn1); // Yn2 -> Yn1
                        swap(Yn3, Yn2); // Yn3 -> Yn2
                        swap(Yn4, Yn3); // Yn4 -> Yn3
                        swap(Fn1, Fn0); // Fn1 -> Fn0
                        swap(Fn2, Fn1); // Fn2 -> Fn1
                        swap(Fn3, Fn2); // Fn3 -> Fn2
                    } // Tras los cambios, Yn4 & Fn3 contienen basura y serán reescritos

                    #pragma omp for // Calculamos el nuevo Fn3 usando el Yn4 recién creado (que ahora es Yn3)
                    for (int i = 0; i < neqn; ++i)
                        Fn3[i] = problema->feval_i(tn+h, Yn3, i); // f(tn4,Yn3') -> Fn3'
                } // Fin del bucle for principal
            } // Fin del parallel
        break;        
    } // Fin del switch principal

    switch(orden){ // Arrastre de los valores finales
        case 1:
        case 2:
            #pragma omp parallel for // Yn0' -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn0[i];
        break;
        case 3:
            #pragma omp parallel for // Yn1' -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn1[i];
        break;
        case 4:
            #pragma omp parallel for // Yn2' -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn2[i];
        break;
        case 5:
            #pragma omp parallel for // Yn3' -> Yf
            for (int i = 0; i < neqn; ++i)
                Yf[i] = Yn3[i];
        break;
    } // Fin del switch de arrastre


    delete[] Yn0; delete[] Yn1; delete[] Yn2; delete[] Yn3; delete[] Yn4;
    delete[] Fn0; delete[] Fn1; delete[] Fn2; delete[] Fn3;
    delete[] Y_AB; delete[] F_AB;
    delete[] Y_AM; delete[] F_AM;
}
#endif