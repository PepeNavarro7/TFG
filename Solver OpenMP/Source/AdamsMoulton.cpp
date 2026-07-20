#ifndef ADAMS_MOULTON_CPP
#define ADAMS_MOULTON_CPP

#include "AdamsMoulton.h"
#include <iostream>
#include <omp.h>
#include <string>

using namespace std;

// Aplicar Adams-Bashford-Moulton con paralelismo de operaciones el numero necesario de veces
void AdamsMoulton::aplicar(const Problema* problema, const double &t0, const double &tf, const double &h, const double* __restrict Y0, double* __restrict Yf) const {
    const int neqn = get_neqn();
    double *Yn0 = new double[neqn], *Yn1 = new double[neqn], *Yn2 = new double[neqn], *Yn3 = new double[neqn], *Yn4 = new double[neqn], // Vectores intermedios
           *Fn0 = new double[neqn], *Fn1 = new double[neqn], *Fn2 = new double[neqn], *Fn3 = new double[neqn], *Fn4 = new double[neqn], // Vectores feval
           *Y_AB = new double [neqn], *F_AB = new double [neqn],    // Vectores para aproximar Yn usando Adams-Bashford
           *Y_AM = new double [neqn], *F_AM = new double [neqn],    // Vectores para aproximar Yn usando Adams-Moulton
           *Yaux = new double[neqn]; // Vector auxiliar

    vectorCopia(Y0, Yn0);   // Definimos Yn0
    const double h_RK = h/100.0;
    switch(get_orden()){ // Aplicamos RK4 para obtener los primeros pasos
        case 1: case 2: break; // No necesitamos RK en orden 1 & 2
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

    switch(get_orden()){ // Arrancamos los feval
        case 1: case 2:
            problema->feval(t0, Yn0, Fn0); // f(tn0,Yn0) -> Fn0
        break;
        case 3:
            problema->feval(t0,      Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK, Yn1, Fn1); // f(tn1,Yn1) -> Fn1
        break;
        case 4:
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
        break;
        case 5:
            problema->feval(t0,          Yn0, Fn0); // f(tn0,Yn0) -> Fn0
            problema->feval(t0+h_RK,     Yn1, Fn1); // f(tn1,Yn1) -> Fn1
            problema->feval(t0+h_RK*2.0, Yn2, Fn2); // f(tn2,Yn2) -> Fn2
            problema->feval(t0+h_RK*3.0, Yn3, Fn3); // f(tn3,Yn3) -> Fn3
        break;
    } // Fin del switch de arranque

    const double h2=h/2.0, h12=h/12.0, h24=h/24.0, h720=h/720.0;
    switch(get_orden()){
        case 1: // Yn+1 = Yn + h * Fn+1
            for (double tn=t0; tn<tf; tn+=h){
                escalarSumaMult(Yn0, h, Fn0, Y_AB);// Yn0 + h * Fn0 -> Y_AB          
                problema->feval(tn+h, Y_AB, F_AB);  // f(tn1, Y_AB) -> F_AB

                escalarSumaMult(Yn0, h, F_AB, Y_AM);// Yn0 + h * F_AB -> Y_AM  
                problema->feval(tn+h, Y_AM, F_AM);  // f(tn1, Y_AM) -> F_AM
                
                escalarSumaMult(Yn0, h, F_AM, Yn1);// Yn0 + h * F_AM -> Yn1  
                problema->feval(tn+h, Yn1, Fn1);  // f(tn1, Yn1) -> Fn1
                
                // Ahora que tenemos Yn1 & Fn1, convertimos los n en n-1 para hacer la siguiente iteracion
                swap(Yn1, Yn0); // Yn1 -> Yn0
                swap(Fn1, Fn0); // Fn1 -> Fn0
                // Tras los cambios, Yn1 & Fn1 contienen basura y serán reescritos
            } // Fin del bucle for principal
        break;
        case 2: // Yn+1 = Yn + h/2 * (Fn+1 + Fn)
            for (double tn=t0; tn<tf; tn+=h){
                escalarSumaMult(Yn0, h, Fn0, Y_AB);// Yn0 + h * Fn0 -> Y_AB  
                problema->feval(tn+h, Y_AB, F_AB);  // f(tn1, Y_AB) -> F_AB
                
                // Y_AM = Yn0 + h/2 * (F_AB + Fn0)
                escalar2Mult(1.0, F_AB, 1.0, Fn0, Yaux); // F_AB + Fn0 -> Yaux
                escalarSumaMult(Yn0, h2, Yaux, Y_AM); // Yn0 + h/2 * Yaux -> Y_AM
                problema->feval(tn+h, Y_AM, F_AM);  // f(tn1, Y_AM) -> F_AM
                
                // Yn1 = Yn0 + h/2 * (F_AM + Fn0)
                escalar2Mult(1, F_AM, 1, Fn0, Yaux); // F_AM + Fn0 -> Yaux
                escalarSumaMult(Yn0, h2, Yaux, Yn1); // Yn0 + h/2 * Yaux -> Yn1
                problema->feval(tn+h, Yn1, Fn1);  // f(tn1, Yn1) -> Fn1

                // Ahora que tenemos Yn1 & Fn1, convertimos los n en n-1 para hacer la siguiente iteracion
                swap(Yn1, Yn0); // Yn1 -> Yn0
                swap(Fn1, Fn0); // Fn1 -> Fn0
                // Tras los cambios, Yn1 & Fn1 contienen basura y serán reescritos
            } // Fin del bucle for principal
        break;
        case 3: // Yn+1 = Yn + h/12 * (5Fn+1 + 8Fn - 1Fn-1)
            for (double tn=t0+h_RK; tn<tf; tn+=h){
                // Y_AB = Yn1 + h/2 * (3*Fn1 - Fn0)
                escalar2Mult(3.0, Fn1, -1.0, Fn0, Yaux); // (3*Fn1 - Fn0) -> Yaux
                escalarSumaMult(Yn1, h2, Yaux, Y_AB); // Yn1 + h/2 * Yaux -> Y_AB
                problema->feval(tn+h, Y_AB, F_AB);  // f(tn2, Y_AB) -> F_AB
          
                // Y_AM = Yn1 + h/12 * (5F_AB + 8Fn1 - 1Fn0)
                escalar2Mult(8.0, Fn1, -1.0, Fn0, Yaux); // 8Fn1 - 1Fn0 -> Yaux
                escalarPorVector(5.0, F_AB, Yaux); // Yaux += 5*F_AB
                escalarSumaMult(Yn1, h12, Yaux, Y_AM); // Yn1 + h/12 * Yaux -> Y_AM
                problema->feval(tn+h, Y_AM, F_AM);  // f(tn2, Y_AM) -> F_AM

                // Yn2 = Yn1 + h/12 * (5Fn_AM + 8Fn1 - 1Fn0)
                escalar2Mult(8.0, Fn1, -1.0, Fn0, Yaux); // 8Fn1 - 1Fn0 -> Yaux
                escalarPorVector(5.0, F_AM, Yaux); // Yaux += 5*Fn_AM
                escalarSumaMult(Yn1, h12, Yaux, Yn2); // Yn1 + h/12 * Yaux -> Yn2
                problema->feval(tn+h, Yn2, Fn2);  // f(tn2, Yn2) -> Fn2

                // Ahora que tenemos Yn2 & Fn2, convertimos los n en n-1 para hacer la siguiente iteracion
                swap(Yn2, Yn1); // Yn2 -> Yn1
                swap(Fn1, Fn0); // Fn1 -> Fn0
                swap(Fn2, Fn1); // Fn1 -> Fn0
                // Tras los cambios, Yn2 & Fn2 contienen basura y serán reescritos
            } // Fin del bucle for principal
        break;
        case 4: // Yn+1 = Yn + h/24 * (9Fn+1 + 19Fn - 5Fn-1 + 1Fn-2)
            for (double tn=t0+h_RK*2.0; tn<tf; tn+=h){
                // Y_AB = Yn2 + h/12 * (23*Fn2 - 16*Fn1 + 5*Fn0)
                escalar2Mult(-16.0, Fn1, 5.0, Fn0, Yaux); // -16*Fn1 + 5*Fn0 -> Yaux
                escalarPorVector(23.0, Fn2, Yaux); // Yaux += 23*Fn2
                escalarSumaMult(Yn2, h12, Yaux, Y_AB); // Yn2 + h/12 * Yaux -> Y_AB
                problema->feval(tn+h, Y_AB, F_AB); // f(tn3, Y_AB) -> F_AB

                // Y_AM = Yn2 + h/24 * (9F_AB + 19Fn2 - 5Fn1 + 1Fn0)
                escalar2Mult(-5.0, Fn1, 1.0, Fn0, Yaux); // -5Fn1 + 1Fn0 -> Yaux
                escalarSuma2Mult(Yaux, 9.0, F_AB, 19.0, Fn2, Yaux); // Yaux+= 9F_AB + 19Fn2
                escalarSumaMult(Yn2, h24, Yaux, Y_AM); // Yn2 + h/24 * Yaux -> Y_AM
                problema->feval(tn+h, Y_AM, F_AM); // f(tn3, Y_AM) -> F_AM

                // Yn3 = Yn2 + h/24 * (9F_AM + 19Fn2 - 5Fn1 + 1Fn0)
                escalar2Mult(-5.0, Fn1, 1.0, Fn0, Yaux); // -5Fn1 + 1Fn0 -> Yaux
                escalarSuma2Mult(Yaux, 9.0, F_AM, 19.0, Fn2, Yaux); // Yaux+= 9F_AM + 19Fn2
                escalarSumaMult(Yn2, h24, Yaux, Yn3); // Yn2 + h/24 * Yaux -> Yn3
                problema->feval(tn+h, Yn3, Fn3); // f(tn3,Yn3) -> Fn3

                // Ahora que tenemos Yn3 & Fn3, convertimos todos los n en n-1 para hacer la siguiente iteracion
                    swap(Yn3, Yn2); // Yn3 -> Yn2
                    swap(Fn1, Fn0); // Fn1 -> Fn0
                    swap(Fn2, Fn1); // Fn2 -> Fn1
                    swap(Fn3, Fn2); // Fn2 -> Fn1
                // Tras los cambios, Yn3 & Fn3 contienen basura y serán reescritos
            } // Fin del bucle for principal
        break;
        case 5: // Yn+1 = Yn + h/720 * (251Fn+1 + 646Fn - 264Fn-1 + 106Fn-2 - 19Fn-3)
            for (double tn=t0+h_RK*3.0; tn<tf; tn+=h){
                // Y_AB = Yn3 + h/24 * (55Fn3 - 59Fn2 + 37Fn1 - 9Fn0)
                escalar2Mult(37.0, Fn1, 9.0, Fn0, Yaux); // +37Fn1 - 9Fn0 -> Yaux
                escalarSuma2Mult(Yaux, 55.0, Fn3, -59.0, Fn2, Yaux); // Yaux+= 55Fn3 - 59Fn2
                escalarSumaMult(Yn3, h24, Yaux, Y_AB); // Yn3 + h/24 * Yaux -> Y_AB
                problema->feval(tn+h, Y_AB, F_AB); // f(tn4, Y_AB) -> F_AB

                // Y_AM = Yn3 + h/720 * (251F_AB + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)
                escalar2Mult(106.0, Fn1, -19.0, Fn0, Yaux); // 106Fn1 - 19Fn0 -> Yaux
                escalarSuma2Mult(Yaux, 646.0, Fn3, -264.0, Fn2, Yaux); // Yaux+= 646Fn3 - 264Fn2
                escalarPorVector(251.0, F_AB, Yaux); // Yaux += 251*F_AB
                escalarSumaMult(Yn3, h720, Yaux, Y_AM); // Yn3 + h/720 * Yaux -> Y_AM
                problema->feval(tn+h, Y_AM, F_AM); // f(tn4, Y_AM) -> F_AM

                // Yn4 = Yn3 + h/720 * (251F_AM + 646Fn3 - 264Fn2 + 106Fn1 - 19Fn0)
                escalar2Mult(106.0, Fn1, -19.0, Fn0, Yaux); // 106Fn1 - 19Fn0 -> Yaux
                escalarSuma2Mult(Yaux, 646.0, Fn3, -264.0, Fn2, Yaux); // Yaux+= 646Fn3 - 264Fn2
                escalarPorVector(251.0, F_AM, Yaux); // Yaux += 251*F_AM
                escalarSumaMult(Yn3, h720, Yaux, Yn4); // Yn3 + h/720 * Yaux -> Yn4
                problema->feval(tn+h, Yn4, Fn4); // f(tn4, Yn4) -> Fn4
                
                // Ahora que tenemos Yn4 & Fn4, convertimos todos los n en n-1 para hacer la siguiente iteracion
                    swap(Yn4, Yn3); // Yn4 -> Yn3
                    swap(Fn1, Fn0); // Fn1 -> Fn0
                    swap(Fn2, Fn1); // Fn2 -> Fn1
                    swap(Fn3, Fn2); // Fn3 -> Fn2
                    swap(Fn4, Fn3); // Fn4 -> Fn3
                // Tras los cambios, Yn4 & Fn4 contienen basura y serán reescritos
            } // Fin del bucle for principal
        break;        
    } // Fin del switch principal

    switch(get_orden()){ // Arrastre de los valores finales
        case 1: 
        case 2: vectorCopia(Yn0, Yf); break; // Yn0' -> Yf
        case 3: vectorCopia(Yn1, Yf); break; // Yn1' -> Yf
        case 4: vectorCopia(Yn2, Yf); break; // Yn2' -> Yf
        case 5: vectorCopia(Yn3, Yf); break; // Yn3' -> Yf
    } // Fin del switch de arrastre


    delete[] Yn0; delete[] Yn1; delete[] Yn2; delete[] Yn3; delete[] Yn4;
    delete[] Fn0; delete[] Fn1; delete[] Fn2; delete[] Fn3; delete[] Fn4;
    delete[] Y_AB; delete[] F_AB; delete[] Y_AM; delete[] F_AM; delete[] Yaux;
}
#endif