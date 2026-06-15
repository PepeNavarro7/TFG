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
#include "brusselator2d.h"
#include "prueba.h"

using namespace std;

int main(int argc, char *argv[]){ // solver metodo orden problema hebras tamvector salto
	if (argc != 7){
		string texto = "solverOMP metodo= orden= problema= hebras= tamvector= salto=\n";
		texto+= "\tMetodos: 1=Runge-Kutta 2=Adams-Bashford 3=Adams-Moulton\n\tOrden: 1-2-3-4\n";
		texto+= "\tProblemas: 1=simpleavdiff 2=advdiff1d 3=brusselator1d 4=brusselator2d\n\tNumero de hebras en OMP {1,4,16}\n";
		texto+= "\tTamaño del vector[100,10000]\n\tSalto en la forma 10^(-x)\n";
		cout << texto;
		return 0;
	}
		
    // Variables que usaremos en el solver
	const int num_metodo = atoi(argv[1]), 	// Metodo a utilizar
			  orden_metodo = atoi(argv[2]), // Orden del metodo
			  num_problema = atoi(argv[3]),	// Problema a ejecutar
			  num_hebras = atoi(argv[4]),	// Numero de hebras OpenMP
        	  num_points = atoi(argv[5]), 	// Tamaño del vector (malla espacial)
			  salto = atoi(argv[6]);		// Salto en la forma 10^-X
	const double t0 = 0.0, 					// Valor de tiempo inicial
				 tf = 1.0,  				// Valor de tiempo final
				 h = pow(10,(-1*salto));	// Valor de salto
	const int num_iter = (tf-t0)/h; // Numero total de iteraciones
	double timeIni, timeFin, tiempo_ms, tiempo_m; // Medidores para calcular el tiempo de procesamiento
	omp_set_num_threads(num_hebras); // Marcamos numero de hebras en regiones paralelas

    // Objetos y puntero de los diferentes problemas
	prueba prueba(num_points);
	simpleadvdiff1d simpleadvdiff1d(num_points); 	// 1D_Simple Advection-Diffusion
	advdiff1d advdiff1d(num_points); 				// 1D Advection-Diffusion model 
	brusselator1d brusselator1d(num_points); 		// 1D Brusselator model 
	brusselator2d brusselator2d(num_points); 		// 1D Brusselator model 
	Problema *ptr_problema; 						// Puntero al problema seleccionado
	switch(num_problema){
		case 0: ptr_problema=&prueba; break;
		case 1: ptr_problema=&simpleadvdiff1d; break;
		case 2: ptr_problema=&advdiff1d; break;
		case 3: ptr_problema=&brusselator1d; break;
		case 4: ptr_problema=&brusselator2d; break;
		default: ptr_problema=NULL; break;
	}

	// Objetos y puntero de los metodos de resolucion
	const int neqn = ptr_problema->get_num_ODEs(); 					// Obtenemos el numero de ODEs del problema
	RungeKutta RungeKutta(neqn, orden_metodo); 						// Objeto para aplicar Runge-Kutta y sus operaciones asociadas
	AdamsBashford AdamsBashford(neqn, orden_metodo, &RungeKutta); 				// Objeto para aplicar Adams-Bashford y sus operaciones asociadas
	AdamsMoulton AdamsMoulton(neqn, orden_metodo, &RungeKutta, &AdamsBashford); 	// Objeto para aplicar Adams-Moulton y sus operaciones asociadas
	Metodo *ptr_metodo; 											// Puntero al metodo seleccionado
	switch(num_metodo){
		case 1: ptr_metodo=&RungeKutta; break;
		case 2: ptr_metodo=&AdamsBashford; break;
		case 3: ptr_metodo=&AdamsMoulton; break;
		default: ptr_metodo=NULL; break;
	}

	double *Y0 = new double[neqn], // Vector de entrada
		   *Y1 = new double[neqn]; // Vector de salida
	ptr_problema->init(Y0); 					// inicializamos el vector inicial
	ptr_problema->archivo("datos0.txt", Y0);	// Guardamos en un txt los valores iniciales
	timeIni = omp_get_wtime();					// Obtenemos el tiempo antes de computar
	ptr_metodo->aplicar(ptr_problema, t0, tf, h, Y0, Y1); // Aplicamos el método
	timeFin = omp_get_wtime();					// Obtenemos el tiempo después de computar
	ptr_problema->archivo("datos1.txt", Y1);	// Guardamos en un txt los valores finales
	tiempo_ms = (timeFin - timeIni)*1000.0;		// Calculamos el tiempo en milisegundos
	tiempo_m = (tiempo_ms / 1000.0)/60.0;		// Calculamos el tiempo en minutos
	
	cout << "Problema " << num_problema << " -> " << ptr_problema->get_nombre() << endl;
	cout << "Metodo de resolucion -> " << ptr_metodo->get_nombre() << " de orden " << orden_metodo << endl;
	cout << "Tamaño del vector -> " << num_points << endl;
	cout << "Numero de ecuaciones -> " << neqn << endl;
	cout << "Salto h=" << h << " -> " << num_iter << " iteraciones" << endl;
	cout << "Numero de hebras OpenMP -> " << num_hebras << endl;
	cout << "Tiempo -> "<< tiempo_ms << " milisegundos, es decir, " << floor(tiempo_m) << " minuto";
	if(floor(tiempo_m)!=1)
		cout << "s";
	cout << " y " << (tiempo_m-floor(tiempo_m))*60 << " segundos." << endl;
	
	delete [] Y0, Y1;
	return 0;
}