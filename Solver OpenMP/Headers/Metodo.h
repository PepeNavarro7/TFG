#ifndef METODO_H
#define METODO_H

using namespace std;

#include <string>
#include "Problema.h"

class Metodo{
private:
    const double PI = 3.14159265358979;
    const int neqn;  // Numero de ecuaciones
    const string nombre;    // Nombre del método
    const int orden;        // Orden

protected:
    Metodo(const int &neqn, const string &nombre, const int &orden):
        neqn(neqn), nombre(nombre), orden(orden) {};

public:
    virtual void aplicar(const Problema* problema, const double &t0, const double &tf, const double &h0, const double* __restrict Y0, double* __restrict Y1) const = 0;
    
   
    void escalarPorVector(const double &esc, const double* __restrict X, double* __restrict Y) const; // Y += esc*X
    void vectorCopia(const double* __restrict X, double* __restrict Y) const;// Copia de X en Y
    void escalarSumaMult(const double* __restrict Y0, const double &esc, const double* __restrict X, double* __restrict Yf) const; // Y0 + esc*X -> Yf
    void escalar2Mult(const double &esc1, const double* __restrict X, const double &esc2, const double* __restrict Z, double* __restrict Yf) const; // e1*X + e2*Z -> Yf
    void escalarSuma2Mult(const double* __restrict Y0, const double &esc1, const double* __restrict X, const double &esc2, const double* __restrict Z, double* __restrict Yf) const; // Y0 + e1*X + e2*Z -> Yf

    inline double get_PI() const { return PI; };
    inline int get_neqn() const { return neqn; };
    inline string get_nombre() const { return nombre; };
    inline int get_orden() const { return orden; };
};

#endif