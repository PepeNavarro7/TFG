#ifndef PROBLEMA_H
#define PROBLEMA_H

using namespace std;

#include <string>

// Clase abstracta que sirve de base para los problemas que se implementen
class Problema{
protected:
    const double PI = 3.14159265358979;
    int neqn; // numero de ecuaciones
    string name; // nombre del problema
    double dtx; // Spacial step
        
    void archivo1(const string &filename, const double *Y) const; // Sacar vector de 1 componente por archivo
    void archivo2(const string &filename, const double *Y) const; // Sacar vector de 2 componentes por archivo
    void archivo3(const string &filename, const double *Y) const; // Sacar vector de 3 componentes por archivo

public:
    inline int get_num_ODEs() const { return neqn; };
    inline string get_name() const { return name; };
    virtual void init(double *Y0) const = 0;
    virtual void feval (const double &t, const double *Y, double *DY) const = 0; // Evaluacion de la exprexion, G+F
    virtual double feval_i (const double &t, const double *Y, const int &i) const = 0; // Evaluacion de la exprexion pero solo para 1 término
    virtual void archivo (const string &filename, const double *Y) const = 0;
};

#endif