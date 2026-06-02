#ifndef PROBLEMA_CPP
#define PROBLEMA_CPP

#include <iostream>
#include <fstream>
#include "Problema.h"

using namespace std;

// Exportamos al archivo los valores en dos columnas, el valor dtx y el valor correspondiente del vector
void Problema::archivo1(const string &filename, const double *Y) const{
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    const double aux = 1.0/neqn;
    for (int i = 0; i < neqn; ++i){
      double x_i=(double)(i+1)*(aux);
      file << x_i << '\t' << Y[i] << endl; 
    }
    cout << "Generado " << str << endl;
    file.close();
}

// Exportamos al archivo los valores en 3 columnas, el valor dtx y los 2 valores correspondientes del vector
void Problema::archivo2(const string &filename, const double *Y) const{
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    const double aux = 1.0/neqn;
    for (int i = 0; i < neqn; i+=2){
      double x_i=(double)(i+1)*(aux);
      file << x_i << '\t' << Y[i] << '\t' << Y[i+1] << endl; 
    }
    cout << "Generado " << str << endl;
    file.close();
}

// Exportamos al archivo los valores en 4 columnas, el valor dtx y los 3 valores correspondientes del vector
void Problema::archivo3(const string &filename, const double *Y) const{
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    const double aux = 1.0/neqn;
    for (int i = 0; i < neqn; i+=3){
      double x_i=(double)(i+1)*(aux);
      file << x_i << '\t' << Y[i] << '\t' << Y[i+1] << '\t' << Y[i+2] << endl; 
    }
    cout << "Generado " << str << endl;
    file.close();
}

#endif