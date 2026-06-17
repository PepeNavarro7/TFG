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
    const int THREADSPERBLOCK; // hebras cuda en cada bloque
    const int NUM_BLOCKS; // numero de bloques
    
    Problema (const int &neqn, const string &name, const double &dtx, const int &threads):
        neqn(neqn), name(name), dtx(dtx), THREADSPERBLOCK(threads), NUM_BLOCKS( ceil((double)neqn/threads) ) {};
        
    void archivo1(const string &filename, const double *Y) const; // Sacar vector de 1 componente por archivo
    void archivo2(const string &filename, const double *Y) const; // Sacar vector de 2 componentes por archivo
    void archivo3(const string &filename, const double *Y) const; // Sacar vector de 3 componentes por archivo

public:
    virtual void updateConstants() const = 0; // Definicion de los valores constantes para el kernel
    virtual void init(double *Y0) const = 0;
    virtual void feval (const double &t, const double *Y, double *DY) const = 0; // Evaluacion de la exprexion, G+F
    virtual void archivo (const string &filename, const double *Y) const = 0;

    inline int get_num_ODEs() const { return neqn; };
    inline string get_name() const { return name; };
    inline int get_threads_per_block() const { return THREADSPERBLOCK; };
    inline int get_num_blocks() const { return NUM_BLOCKS; };
};



#endif