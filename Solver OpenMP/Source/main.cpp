#include <iostream> // cout y cin
#include <fstream> // read y write en archivos
#include <omp.h> // OpenMP
#include <string> // strings
#include <iomanip> // cambiar precision de los double
#include <cmath> // floor
#include <cassert>
#define assertm(exp, msg) assert((void(msg), exp))

#include "Metodo.h"
#include "RungeKutta.h"
#include "RungeKutta_i.h"
#include "AdamsBashforth.h"
#include "AdamsBashforth_i.h"
#include "AdamsMoulton.h"
#include "AdamsMoulton_i.h"

#include "Problema.h"
#include "simpleadvdiff1d.h"
#include "advdiff1d.h"
#include "brusselator1d.h"
#include "brusselator2d.h"

using namespace std;

int main(int argc, char *argv[]){ // solver metodo= orden= problema= hebras= tamvector= t0= tf= salto= modo=
	if (argc != 10){
		string texto = "./solverOMP metodo= orden= problema= hebras= tamvector= t0= tf= salto= modo=\n";
		texto += "\tmetodo: 1=RK(operaciones) 2=RK(componentes) 3=AB(operaciones)\n";
		texto += "\t\t4=AB(componentes) 5=ABM(operaciones) 6=ABM(componentes)\n";
		texto += "\torden: 1 - 2 - 3 - 4 - 5(solo ABM)\n";
		texto += "\tproblema: 1=simpleavdiff 2=advdiff1d 3=brusselator1d 4=brusselator2d\n";
		texto += "\thebras: {1, 4, 16}\n\ttamvector: 100, 1000, 10000, 100000...\n";
		texto += "\tt0: Tiempo de inicio {0.0}\n\ttf: Tiempo de fin {1.0}\n";
		texto += "\tsalto: 10^-x {5,6,7}\n";
		texto += "\tmodo: Salida por pantalla si=0 no=1\n\n";
		cout << texto;
		return 0;
	}
		
	// Variables que usaremos en el solver
	const int num_metodo = atoi(argv[1]), 	// Metodo a utilizar -> [1,6]
		orden_metodo = (num_metodo<=4 && atoi(argv[2])==5 ) ? 4 : atoi(argv[2]),		// Orden del metodo -> [1,5]
		num_problema = atoi(argv[3]), 		// Problema a ejecutar -> [0,4]
		num_hebras = atoi(argv[4]),			// Numero de hebras OpenMP
        num_points = atoi(argv[5]);			// Tamaño del vector -> [100, 1000]
	const double t0 = atof(argv[6]),		// Valor de tiempo inicial
		tf = atof(argv[7]);					// Valor de tiempo final
	const int salto = atoi(argv[8]); 		// Salto en la forma 10^-X -> [5,7]
	const bool silencioso = atoi(argv[9])==1; // Modo silencioso
	const double h = pow(10,(-1*salto));	// Valor de salto h
	const int num_iter = (tf-t0)/h; 		// Numero de iteraciones que se realizarán
	assertm(num_metodo>=1 && num_metodo<=6, "Método a utilizar -> {1,2,3,4,5,6}");
	assertm(orden_metodo>=1 && orden_metodo<=5, "Orden del método -> {1,2,3,4,5}");
	assertm(num_problema>=1 && num_problema<=4, "Problema a ejecutar -> {1,2,3,4}");
	assertm(num_hebras%2 == 0 || num_hebras==1, "Numero de hebras inválido");
	assertm(atoi(argv[9])==0 || atoi(argv[9])==1, "Salida por pantalla sí/no-> {0,1}");
	
	double timeIni, timeFin, tiempo_ms, tiempo_s, tiempo_m; // Medidores para calcular el tiempo de procesamiento
	omp_set_num_threads(num_hebras); // Marcamos numero de hebras en regiones paralelas

    // Objetos y puntero de los diferentes problemas
	simpleadvdiff1d simpleadvdiff1d(num_points); 	// 1D_Simple Advection-Diffusion
	advdiff1d advdiff1d(num_points); 				// 1D Advection-Diffusion model 
	brusselator1d brusselator1d(num_points); 		// 1D Brusselator model 
	brusselator2d brusselator2d(num_points); 		// 2D Brusselator model 
	Problema *ptr_problema; 						// Puntero al problema seleccionado
	switch(num_problema){
		case 1: ptr_problema=&simpleadvdiff1d; break;
		case 2: ptr_problema=&advdiff1d; break;
		case 3: ptr_problema=&brusselator1d; break;
		case 4: ptr_problema=&brusselator2d; break;
		default: ptr_problema=NULL; break;
	}

	// Objetos y puntero de los metodos de resolución
	const int neqn = ptr_problema->get_neqn(); 			// Obtenemos el número de ODEs del problema
	RungeKutta RungeKutta(neqn, orden_metodo); 			// Objeto para aplicar Runge-Kutta (paralelismo de operaciones) y sus operaciones asociadas
	RungeKutta_i RungeKutta_i(neqn, orden_metodo); 		// Obejto para aplicar Runge-Kutta (paralelismo de elementos)
	AdamsBashforth AdamsBashforth(neqn, orden_metodo, &RungeKutta); 		// Objeto para aplicar Adams-Bashforth (paralelismo de operaciones) y sus operaciones asociadas
	AdamsBashforth_i AdamsBashforth_i(neqn, orden_metodo, &RungeKutta_i); // Objeto para aplicar Adams-Bashforth (paralelismo de componentes)
	AdamsMoulton AdamsMoulton(neqn, orden_metodo, &RungeKutta); 		// Objeto para aplicar Adams-Bashforth-Moulton (paralelismo de operaciones) y sus operaciones asociadas
	AdamsMoulton_i AdamsMoulton_i(neqn, orden_metodo, &RungeKutta_i);	// Objeto para aplicar Adams-Bashforth-Moulton (paralelismo de componentes)
	Metodo *ptr_metodo; 	// Puntero al método seleccionado
	switch(num_metodo){
		case 1: ptr_metodo=&RungeKutta; break;
		case 2: ptr_metodo=&RungeKutta_i; break;
		case 3: ptr_metodo=&AdamsBashforth; break;
		case 4: ptr_metodo=&AdamsBashforth_i; break;
		case 5: ptr_metodo=&AdamsMoulton; break;
		case 6: ptr_metodo=&AdamsMoulton_i; break;
		default: ptr_metodo=NULL; break;
	}

	double* __restrict Y0 = new double[neqn]; // Vector de entrada
	double* __restrict Y1 = new double[neqn]; // Vector de salida
	ptr_problema->init(Y0); 					// Inicializamos el vector inicial
	timeIni = omp_get_wtime();					// Obtenemos el tiempo antes de computar
	ptr_metodo->aplicar(ptr_problema, t0, tf, h, Y0, Y1); // Aplicamos el método
	timeFin = omp_get_wtime();					// Obtenemos el tiempo después de computar
	tiempo_ms = (timeFin - timeIni)*1000.0;		// Calculamos el tiempo en milisegundos
	tiempo_s = tiempo_ms/1000.0;		// Calculamos el tiempo en segundos
	tiempo_m = tiempo_s/60.0;		// Calculamos el tiempo en minutos

	if(silencioso){ // Exportamos el tiempo a un archivo csv
		ptr_problema->tiempos(num_metodo, orden_metodo, num_problema, num_hebras, num_points, salto, tiempo_s);
	} else{
		ptr_problema->archivo("datos0.txt", Y0);	// Guardamos en un txt los valores iniciales
		ptr_problema->archivo("datos1.txt", Y1);	// Guardamos en un txt los valores finales

		cout << "\nProblema " << num_problema << " -> " << ptr_problema->get_name() << endl;
		cout << "Metodo de resolucion -> " << ptr_metodo->get_nombre() << " de orden " << ptr_metodo->get_orden() << endl;
		cout << "Tamaño del vector-> " << num_points << endl;
		cout << "Numero de ecuaciones -> " << ptr_problema->get_neqn() << endl;
		cout << "T0 = " << t0 << " y tf = " << tf << endl;
		cout << "Salto h=" << h << " -> " << num_iter << " iteraciones" << endl;
		cout << "Numero de hebras OpenMP -> " << num_hebras << endl;
		cout << "Tiempo -> "<< tiempo_ms << " milisegundos, es decir, ";
		if (floor(tiempo_m)>0) {
			cout << floor(tiempo_m) << " minuto";
			if(floor(tiempo_m)!=1)
				cout << "s";
			cout << " y ";
		}
		cout << tiempo_s - (floor(tiempo_m) * 60.0) << " segundos\n" << endl;
	}
	
	delete [] Y0, Y1;
	return 0;
}