#include <iostream> // cout y cin
#include <fstream> // read y write en archivos
#include <omp.h> // OpenMP
#include <string> // strings
#include <iomanip> // cambiar precision de los double
#include <cmath> 

#include "RungeKutta.h"
#include "Problema.h"
#include "simpleadvdiff1d.h"
#include "advdiff1d.h"
#include "brusselator1d.h"
#include "prueba.h"

using namespace std;

int main(int argc, char *argv[]){ // solver problema hebras tamanio
	if (argc != 4)
		return 0;
    // Variables que usaremos en el solver
	const int num_problema = atoi(argv[1]), // Problema a ejecutar
		num_hebras = atoi(argv[2]),// numero de hebras
        num_points = atoi(argv[3]);  // tamanio del vector
	const double t0 = 0.0, tf = 1.0,  // valores de tiempo inicial, final
		h = 0.000001;	// valor de salto
	//	h = 0.1;
	const int num_iter = (tf-t0)/h; // numero total de iteraciones
	double timeIni, timeFin, tiempo; // Medidores para calcular el tiempo de procesamiento

	omp_set_num_threads(num_hebras); // Marcamos numero de hebras en regiones paralelas

    // Objetos y puntero que usaremos
	simpleadvdiff1d simpleadvdiff1d(num_points); // 1D_Simple Advection-Diffusion
	advdiff1d advdiff1d(num_points); // 1D Advection-Diffusion model 
	brusselator1d brusselator1d(num_points); // 1D Brusselator model 
	prueba prueba(num_points);
	Problema *ptr; // Puntero al problema seleccionado
	switch(num_problema){
		case 1: ptr=&simpleadvdiff1d; break;
		case 2: ptr=&advdiff1d; break;
		case 3: ptr=&brusselator1d; break;
		
		case 0: ptr=&prueba; break;
		default: ptr=NULL; break;
	}
	const int neqn = ptr->get_num_ODEs(); // Obtenemos el numero de ODEs del problema
	RungeKutta runge(neqn); // objeto para aplicar Runge-Kutta y sus operaciones asociadas

	double *Y0 = new double[neqn], *Y1 = new double[neqn]; // Vectores de entrada y salida
	cout.precision(6);
	ptr->init(Y0); // inicializamos el vector
	ptr->archivo("datos0.txt",Y0);
	timeIni = omp_get_wtime();
	runge.aplicarRK(ptr,t0,tf,h,Y0,Y1);
	timeFin = omp_get_wtime();
	ptr->archivo("datos1.txt", Y1);
	tiempo = (timeFin - timeIni)*1000.0;
	
	
	cout << "Problema " << num_problema << " -> " << ptr->get_name() << endl;
	cout << "Tamanio del vector -> " << num_points << endl;
	cout << "Numero de ecuaciones -> " << neqn << endl;
	cout << "Numero de iteraciones -> " << num_iter << endl;
	cout << "Numero de hebras -> " << num_hebras << endl;
	cout << "Tiempo tardado = "<< tiempo << " milisegundos" <<endl;
	
	delete [] Y0, Y1;
	return 0;
}