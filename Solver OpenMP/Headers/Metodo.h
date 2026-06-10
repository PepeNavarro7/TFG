#ifndef METODO_H
#define METODO_H

using namespace std;

#include <string>
#include "Problema.h"

class Metodo{
protected:
    static constexpr double PI = 3.14159265358979;
    static int neqn;       // Numero de ecuaciones
    string nombre;  // Nombre del método
    int orden;      // Orden

    // Escalar esc * vector X + vector Y -> Y
    static void escalarPorVector(const double &esc, const double *X, double *Y);

    // Copia de X en Y
    static void vectorCopia(const double *X, double *Y);

public:
    virtual void aplicar(Problema* problema, const double &t0, const double &tf, const double &h0, const double *Y0, double *Y1) = 0;
    inline int get_neqn() const { return neqn; };
    inline string get_nombre() const { return nombre; };
    inline int get_orden() const { return orden; };
};

#endif