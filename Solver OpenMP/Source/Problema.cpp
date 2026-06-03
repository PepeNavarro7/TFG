#ifndef PROBLEMA_CPP
#define PROBLEMA_CPP

#include <iostream>
#include <fstream>
#include "Problema.h"
#include <cmath>

using namespace std;

// Exportamos al archivo los valores en dos columnas, el valor dtx y el valor correspondiente del vector
void Problema::archivo1(const string &filename, const double *Y) const{
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    const double aux = 1.0/neqn; // == dtx
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

// Exportamos al archivo los valores en 2N+1 columnas, el valor dtx y los N valores correspondientes de los 2 vectores
void Problema::archivo3(const string &filename, const double *Y) const{
	string str = "./Datos/"+filename;
	ofstream file(str);
	if (!file) {
        cerr << "Error opening the file: " << str << endl;
        return;
	}
	const int n = sqrt((neqn/2));
	for(int i=0; i<n; i++){
        double x_i=(double)(i+1)*(1.0/n);
        file << x_i;
		for(int j=0; j<n; j++){
            file << '\t' << Y[i*2*n + j*2] << '\t' << Y[i*2*n + j*2 + 1]; 
		}
        file << endl;
	}
	cout << "Generado " << str << endl;
	file.close();
}

#endif