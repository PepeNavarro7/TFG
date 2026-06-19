#ifndef METODO_H
#define METODO_H

using namespace std;

#include <string>
#include "Problema.h"
#include <iostream>
#include <string>

class Metodo{
protected:
    const double PI = 3.14159265358979;
    const int neqn;  // numero de ecuaciones
    const string nombre; // Nombre del método
    const int orden; // Orden
    const int THREADSPERBLOCK; // Hebras cuda en cada bloque
    const int NUM_BLOCKS; // Numero de bloques
    const int NUM_BYTES; // numero de bytes de cada vector
    
    Metodo(const int &neqn, const string &nombre, const int &orden, const int &threads):
        neqn(neqn), nombre(nombre), orden(orden), THREADSPERBLOCK(threads), 
        NUM_BLOCKS(ceil((double)neqn/(double)threads)), NUM_BYTES(sizeof(double)*neqn) { };

    virtual void updateConstants(const int &neqn, const double &h) const = 0;

    // Escalar esc * vector X + vector Y -> Y
    void escalarPorVector(const double &esc, const double *X, double *Y) const;

public:
    virtual void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Y1) const = 0;
    inline int get_neqn() const { return neqn; };
    inline string get_name() const { return nombre; };
    inline int get_orden() const { return orden; };
    inline int get_threads_per_block() const { return THREADSPERBLOCK; };
    inline int get_num_blocks() const { return NUM_BLOCKS; };
    inline int get_num_bytes() const { return NUM_BYTES; };
};



#endif