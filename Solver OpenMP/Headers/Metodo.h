#ifndef METODO_H
#define METODO_H

using namespace std;

#include <string>
#include "Problema.h"

class Metodo{
protected:
    static constexpr double PI = 3.14159265358979;
    const int neqn;  // Numero de ecuaciones
    const string nombre;    // Nombre del método
    const int orden;        // Orden

    Metodo(const int &neqn, const string &nombre, const int &orden):
        neqn(neqn), nombre(nombre), orden(orden) {};

    // vector Y += Escalar esc * vector X 
    static void escalarPorVector(const double &esc, const double *X, double *Y, const int &neqn);

    // Copia de X en Y
    static void vectorCopia(const double *X, double *Y, const int &neqn);

public:
    virtual void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h0, const double *Y0, double *Y1) const = 0;
    inline int get_neqn() const { return neqn; };
    inline string get_nombre() const { return nombre; };
    inline int get_orden() const { return orden; };
};

#endif