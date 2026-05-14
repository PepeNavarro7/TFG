#ifndef METODO_H
#define METODO_H

using namespace std;

#include <string>
#include "Problema.h"

class Metodo{
protected:
    const double PI = 3.14159265358979;
    int neqn;  // numero de ecuaciones
    string nombre;

    // Escalar esc * vector X + vector Y -> Y
    void escalarPorVector(const double &esc, const double *X, double *Y);

    // Copia de X en Y
    void vectorCopia(const double *X, double *Y);

public:
    virtual void aplicar(Problema* problema, const double &t0, const double &tf, const double &h0, const double *Y0, double *Y1) = 0;
    inline string get_name() const { return nombre; };
};



#endif