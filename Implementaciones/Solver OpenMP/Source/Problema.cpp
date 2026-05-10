#ifndef PROBLEMA_CPP
#define PROBLEMA_CPP

#include <iostream>
#include <fstream>
#include "Problema.h"

using namespace std;

/*
// Exportamos al archivo los valores en dos columnas, el valor dtx y el valor correspondiente del vector
void Problema::archivo1(const string &filename, const double *Y, const double &t){
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    for (int i = 0; i < neqn; ++i){
      double x_i=(double)(i+1)*(1.0/neqn);
      file << x_i << "     " << Y[i] << endl; 
    }
    cout << "Generado " << str << endl;
    file.close();
}

// Exportamos al archivo los valores en tres columnas, el valor dtx y los valores correspondientes del vector
void Problema::archivo2(const string &filename, const double *Y, const double &t){
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    for (int i = 0; i < neqn; i+=2){
      double x_i=(double)(i+1)*(1.0/neqn);
      file << x_i << "     " << Y[i] << "     " << Y[i+1] << endl; 
    }
    cout << "Generado " << str << endl;
    file.close();
}

*/

#endif