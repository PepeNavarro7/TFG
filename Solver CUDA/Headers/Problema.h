#ifndef PROBLEMA_H
#define PROBLEMA_H

using namespace std;

#include <string>
#include <cmath>

class Problema{
protected:
    const double PI = 3.14159265358979; 
    const int neqn; // numero de ecuaciones
    const string name; // nombre del problema
    const double dtx; // Spacial step
    const dim3 grid; // numero de bloques
    const dim3 block; // hebras cuda en cada bloque
    const double bytes; // numero de bytes
    
    Problema (const int &neqn, const string &name, const double &dtx, const dim3 grid, const dim3 block):
        neqn(neqn), name(name), dtx(dtx), grid(grid), block(block), bytes(sizeof(double)*neqn) { };
        
    void archivo1(const string &filename, const double *Y) const; // Sacar vector de 1 componente por archivo
    void archivo2(const string &filename, const double *Y) const; // Sacar vector de 2 componentes por archivo
    void archivo3(const string &filename, const double *Y) const; // Sacar vector de 3 componentes por archivo
    void archivo2v2(const string &filename, const double *Y) const; // Sacar vector de 2 componentes por archivo, reordenado

public:
    virtual void updateConstants() const = 0; // Definicion de los valores constantes para el kernel
    virtual void init(double *Y0) const = 0;
    virtual void feval (const double &t, const double *Y, double *DY) const = 0; // Evaluacion de la exprexion, G+F
    virtual void archivo (const string &filename, const double *Y) const = 0;

    inline int get_num_ODEs() const { return neqn; };
    inline string get_name() const { return name; };
    inline dim3 get_grid() const { return grid; };
    inline dim3 get_block() const { return block; };
    inline int get_bytes() const { return bytes; };
};



#endif