#ifndef METODO_H
#define METODO_H

using namespace std;

#include <string>
#include "Problema.h"
#include <iostream>
#include <string>

struct Params_Metodo {
    int num;
};

class Metodo{
protected:
    const double PI = 3.14159265358979;
    const int neqn;  // numero de ecuaciones
    const string nombre; // Nombre del método
    const int orden; // Orden
    const int num_blocks; // numero de bloques 1D
    const int tam_blocks; // hebras cuda en cada bloque
    const int bytes; // numero de bytes de cada vector
    
    Metodo(const int &neqn, const string &nombre, const int &orden, const int &threads):
        neqn(neqn), nombre(nombre), orden(orden), num_blocks( (neqn+threads-1)/threads ), tam_blocks(threads), bytes(sizeof(double)*neqn) {
            update();
        };

    virtual void updateConstants(const int &neqn, const double &h) const = 0;
    void update() const;

public:
    virtual void aplicar(Problema* problema, const double &t0, const double &tf, const double &h, const double *Y0, double *Y1) const = 0;
    inline int get_neqn() const { return neqn; };
    inline string get_name() const { return nombre; };
    inline int get_orden() const { return orden; };
    inline int get_num_blocks() const { return num_blocks; };
    inline int get_tam_blocks() const { return tam_blocks; };
    inline int get_bytes() const { return bytes; };
};



#endif