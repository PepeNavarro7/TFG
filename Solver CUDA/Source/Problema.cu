#ifndef PROBLEMA_CU
#define PROBLEMA_CU

#include <iostream>
#include <fstream>
#include <filesystem>
#include "Problema.h"

using namespace std;


// Exportamos al archivo los valores en dos columnas, el valor dtx y el valor correspondiente del vector
void Problema::archivo1(const string &filename, const double *Y) const {
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    for (int i = 0; i < neqn; ++i){
      double x_i=(double)(i+1)*(1.0/neqn);
      file << x_i << '\t' << Y[i] << endl; 
    }
    cout << "Generado " << str << endl;
    file.close();
}

// Exportamos al archivo los valores en tres columnas, el valor dtx y los valores correspondientes del vector
void Problema::archivo2(const string &filename, const double *Y) const {
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    for (int i = 0; i < neqn; i+=2){
      double x_i=(double)(i+1)*(1.0/neqn);
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

// Exportamos al archivo los valores en tres columnas, el valor dtx y los valores correspondientes del vector
void Problema::archivo2v2(const string &filename, const double *Y) const {
    string str = "./Datos/"+filename;
    ofstream file(str);
    if (!file) {
      cerr << "Error opening the file: " << str << endl;
      return;
    }
    const int n = this->neqn/2;
    for (int i = 0; i<n; i++){
      double x_i=(double)(i+1)*(1.0/neqn);
      file << x_i << '\t' << Y[i] << '\t' << Y[i+n] << endl; 
    }
    cout << "Generado " << str << endl;
    file.close();
}

// Exportamos los datos de los tiempos
void Problema::tiempos(const int &metodo, const int &orden, const int &problema, const int &hebras, const int &nx, const int &salto, const double &tiempo) const{
    // metodo orden problema hebras tamvector neqn salto tiempo
    string filename = "tiempos.csv";
    string ruta = "./Datos/"+filename;

    // Detectar si el archivo ya existe
    bool existe = filesystem::exists(ruta);

    // Abrir en modo append
    ofstream file(ruta, ios::app);

    if (!file) {
        cerr << "Error al abrir el archivo\n";
        return;
    }

    // Si el archivo es nuevo, escribir cabecera
    if (!existe) {
        file << "metodo,orden,problema,hebras,tamvector,neqn,salto,tiempo\n";
    }

    // Escribir datos de esta ejecución
    file << metodo << "," << orden << "," << problema << "," << hebras << "," << nx << "," << get_neqn() << "," << salto << "," << tiempo << "\n";
    //cout << "Generado " << ruta << endl;
    file.close();
}

#endif