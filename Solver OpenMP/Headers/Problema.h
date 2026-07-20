#ifndef PROBLEMA_H
#define PROBLEMA_H

using namespace std;

#include <string>

// Clase abstracta que sirve de base para los problemas que se implementen
class Problema{
private:
    const double PI = 3.14159265358979;
    const int neqn; // numero de ecuaciones
    const string nombre; // nombre del problema
    const double dtx; // Spacial step

protected:
    Problema (const int &neqn, const string &nombre, const double &dtx): 
        neqn(neqn), nombre(nombre), dtx(dtx) {};
        
public:
    virtual void init(double* __restrict Y0) const = 0; // Inicialización del vector inicial
    virtual void feval (const double &t, const double* __restrict Y, double* __restrict DY) const = 0; // Evaluacion de la exprexion, G+F
    virtual double feval_i (const double &t, const double* __restrict Y, const int &i) const = 0; // Evaluacion de la exprexion pero solo para 1 término
    virtual void archivo (const string &filename, const double* __restrict Y) const = 0; // Exportar vector a un archivo

    void archivo1(const string &filename, const double* __restrict Y) const; // Sacar vector de 1 componente por archivo
    void archivo2(const string &filename, const double* __restrict Y) const; // Sacar vector de 2 componentes por archivo
    void archivo3(const string &filename, const double* __restrict Y) const; // Sacar vector de 3 componentes por archivo
    void tiempos(const int &metodo, const int &orden, const int &problema, const int &hebras, const int &nx, const int &salto, const double &tiempo) const;

    inline double get_PI() const { return PI; };
    inline int get_neqn() const { return neqn; };
    inline string get_name() const { return nombre; };
    inline double get_dtx() const { return dtx; };
};

#endif