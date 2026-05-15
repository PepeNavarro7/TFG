#include <iostream> // cout y cin
#include <fstream> // read y write en archivos
#include <omp.h> // OpenMP
#include <string> // strings
#include <iomanip> // cambiar precision de los double
#include <cmath> // floor

#include "Metodo.h"
#include "RungeKutta.h"
#include "AdamsBashford.h"
#include "AdamsMoulton.h"

#include "Problema.h"
#include "simpleadvdiff1d.h"
#include "advdiff1d.h"
#include "brusselator1d.h"
#include "prueba.h"

using namespace std;

int main(int argc, char *argv[]){ // solver problema hebras tamvector salto
	if (argc != 6){
		string texto = "solverOMP metodo= problema= hebras= tamvector= salto=\n\tMetodos: 1=Runge-Kutta 2=Adams-Bashford 3=Adams-Moulton\n";
		texto += "\tProblemas: 1=simpleavdiff 2=advdiff1d 3=brusselator1d\n\tNumero de hebras en OMP[1,16]\n\tTamaño del vector[100,10000]\n\tSalto en la forma 10^-X\n";
		cout << texto;
		return 0;
	}
		

    // Variables que usaremos en el solver
	const int num_metodo = atoi(argv[1]), 	// Metodo a utilizar
		num_problema = atoi(argv[2]), 		// Problema a ejecutar
		num_hebras = atoi(argv[3]),			// numero de hebras
        num_points = atoi(argv[4]),			// tamanio del vector
		salto = atoi(argv[5]); 				// salto en la forma 10^-X
	const double t0 = 0.0, // valor de tiempo inicial
		tf = 1.0,  // valor de tiempo final
		h = pow(10,(-1*salto));	// valor de salto
	const int num_iter = (tf-t0)/h; // numero total de iteraciones
	double timeIni, timeFin, tiempo_ms, tiempo_m; // Medidores para calcular el tiempo de procesamiento
	omp_set_num_threads(num_hebras); // Marcamos numero de hebras en regiones paralelas

    // Objetos y puntero de los diferentes problemas
	simpleadvdiff1d simpleadvdiff1d(num_points); 	// 1D_Simple Advection-Diffusion
	advdiff1d advdiff1d(num_points); 				// 1D Advection-Diffusion model 
	brusselator1d brusselator1d(num_points); 		// 1D Brusselator model 
	prueba prueba(num_points);
	Problema *ptr_problema; // Puntero al problema seleccionado
	switch(num_problema){
		case 1: ptr_problema=&simpleadvdiff1d; break;
		case 2: ptr_problema=&advdiff1d; break;
		case 3: ptr_problema=&brusselator1d; break;
		case 0: ptr_problema=&prueba; break;
		default: ptr_problema=NULL; break;
	}

	// Objetos y puntero de los metodos de resolucion
	const int neqn = ptr_problema->get_num_ODEs(); 					// Obtenemos el numero de ODEs del problema
	RungeKutta RungeKutta(neqn); 									// Objeto para aplicar Runge-Kutta y sus operaciones asociadas
	AdamsBashford AdamsBashford(neqn, &RungeKutta); 				// Objeto para aplicar Adams-Bashford y sus operaciones asociadas
	AdamsMoulton AdamsMoulton(neqn, &RungeKutta, &AdamsBashford); 	// Objeto para aplicar Adams-Moulton y sus operaciones asociadas
	Metodo *ptr_metodo; 											// Puntero al metodo seleccionado
	switch(num_metodo){
		case 1: ptr_metodo=&RungeKutta; break;
		case 2: ptr_metodo=&AdamsBashford; break;
		case 3: ptr_metodo=&AdamsMoulton; break;
		default: ptr_metodo=NULL; break;
	}


	double *Y0 = new double[neqn], *Y1 = new double[neqn]; // Vectores de entrada y salida
	cout.precision(6);
	ptr_problema->init(Y0); // inicializamos el vector
	ptr_problema->archivo("datos0.txt",Y0);
	timeIni = omp_get_wtime();
	ptr_metodo->aplicar(ptr_problema,t0,tf,h,Y0,Y1);
	timeFin = omp_get_wtime();
	ptr_problema->archivo("datos1.txt", Y1);
	tiempo_ms = (timeFin - timeIni)*1000.0;
	tiempo_m = (tiempo_ms / 1000.0)/60.0;
	
	
	cout << "Problema " << num_problema << " -> " << ptr_problema->get_name() << endl;
	cout << "Metodo de resolucion -> " << ptr_metodo->get_name() << endl;
	cout << "Tamaño del vector -> " << num_points << endl;
	cout << "Numero de ecuaciones -> " << neqn << endl;
	cout << "Numero de iteraciones -> " << num_iter << endl;
	cout << "Numero de hebras -> " << num_hebras << endl;
	cout << "Tiempo -> "<< tiempo_ms << " milisegundos, es decir, " << floor(tiempo_m) << " minuto";
	if(floor(tiempo_m)!=1)
		cout << "s";
	cout << " y " << (tiempo_m-floor(tiempo_m))*60 << " segundos." << endl;
	
	delete [] Y0, Y1;
	return 0;
}