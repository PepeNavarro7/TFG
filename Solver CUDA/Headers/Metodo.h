#ifndef METODO_H
#define METODO_H

using namespace std;

#include <string>
#include "Problema.h"

class Metodo{
protected:
    const double PI = 3.14159265358979;
    int neqn;  // numero de ecuaciones
    int THREADSPERBLOCK; // hebras cuda en cada bloque
    int NUM_BLOCKS; // numero de bloques
    int NUM_BYTES; // numero de bytes de cada vector
    string nombre;

    // Escalar esc * vector X + vector Y -> Y
    void escalarPorVector(const double &esc, const double *X, double *Y);

    // Copia de X en Y
    //void vectorCopia(const double *X, double *Y);

public:
    virtual void aplicar(Problema* problema, const double &t0, const double &tf, const double &h0, const double *Y0, double *Y1) = 0;
    inline string get_name() const { return nombre; };
    inline int get_threads_per_block() const { return THREADSPERBLOCK; };
    inline int get_num_blocks() const { return NUM_BLOCKS; };
    inline int get_num_bytes() const { return NUM_BYTES; };
    virtual void set_threads(const int &t) = 0;
};



#endif