#ifndef PROBLEMA_H
#define PROBLEMA_H

using namespace std;

#include <string>

class Problema{
protected:
    const double PI = 3.14159265358979;
    int neqn; // numero de ecuaciones
    string name; // nombre del problema
    int THREADSPERBLOCK; // hebras cuda en cada bloque
    int NUM_BLOCKS; // numero de bloques
        
    void archivo1(const string &filename, const double *Y); // Sacar el vector por archivo
    void archivo2(const string &filename, const double *Y); // Sacar el vector de 2 componentes por archivo

public:
    inline int get_num_ODEs() const { return neqn; };
    inline string get_name() const { return name; };
    inline void set_threads(const int t){ THREADSPERBLOCK=t; NUM_BLOCKS = ceil((double)neqn/THREADSPERBLOCK); };
    //inline int get_threads_per_block() const{ return THREADSPERBLOCK; };
    //inline int get_num_blocks() const{ return NUM_BLOCKS; };
    virtual void init(double *Y0) = 0;
    virtual void feval (const double &t, const double *Y, double *DY) = 0; // Evaluacion de la exprexion, G+F
    virtual void archivo (const string &filename, const double *Y) = 0;
};



#endif