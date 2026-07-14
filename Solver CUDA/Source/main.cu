#include <iostream> // cout y cin
#include <fstream> // read y write en archivos
#include <string> // strings
#include <iomanip> // cambiar precision de los double
#include <cmath> // floor
#include <chrono> // medicion de tiempo
#include <cassert>
#define assertm(exp, msg) assert((void(msg), exp))

#include "Metodo.h"
#include "RungeKutta.h"
#include "RungeKutta_graph.h"
#include "AdamsBashford.h"
#include "AdamsBashford_graph.h"
#include "AdamsMoulton.h"

#include "Problema.h"
#include "simpleadvdiff1d.h"
#include "simpleadvdiff1d_shuffle.h"
#include "advdiff1d.h"
#include "advdiff1d_shuffle.h"
#include "brusselator1d.h"
#include "brusselator1d_shuffle.h"
#include "brusselator1d_grid.h"
#include "brusselator2d.h"
#include "brusselator2d_shuffle.h"
//#include "prueba.h"

using namespace std;

int main(int argc, char *argv[]){ // solver problema hebras tamvector salto
	if (argc != 9){
		string texto = "./solverCUDA metodo= orden= problema= hebras= tamvector= t0= tf= salto=\n\tOrden: 1 - 2 - 3 - 4 - 5(solo AM)\n";
		texto += "\tMetodos: 1=Runge-Kutta 2=Runge-Kutta graph 3=Adams-Bashford 4=Adams-Bashford graph 5=Adams-Moulton\n";
		texto += "\tProblemas: 1=simpleavdiff 2=simpleavdiff shuffle 3=advdiff1d 4=advdiff1d shuffle 5=brusselator1d";
		texto += " 6=brusselator1d shuffle 7=brusselator1d grid 8=brusselator2d 9=brusselator2d shuffle\n";
		texto += "\tHebras del bloque CUDA X%32==0\n\tTamaño del vector[100,10000]\n";
		texto += "\tTiempo inicial t0\n\tTiempo final tf\n\tSalto en la forma 10^(-x)\n";
		cout << texto;
		return 0;
	}

    // Variables que usaremos en el solver
	const int num_metodo = atoi(argv[1]), 	// Metodo a utilizar -> [1,5]
		orden_metodo = atoi(argv[2]),		// Orden del metodo -> [1,5]
		num_problema = atoi(argv[3]), 		// Problema a ejecutar -> [0,9]
		num_hebras = atoi(argv[4]),			// Numero de hebras -> X%32==0
        num_points = atoi(argv[5]),			// Tamaño del vector -> [100, 10000]
		t0 = atof(argv[6]),					// Valor de tiempo inicial
		tf = atof(argv[7]),					// Valor de tiempo final
		salto = atoi(argv[8]); 				// Salto en la forma 10^-X -> [5,7]
	const double h = pow(10,(-1*salto));	// Valor de salto h
	const int num_iter = (tf-t0)/h; 		// Numero de iteraciones que se realizarán
	assertm(num_metodo>=1 && num_metodo<=5, "Metodo a utilizar -> [1,5]");
	assertm(orden_metodo>=1 && orden_metodo<=5, "Orden del metodo -> [1,5]");
	assertm(num_problema>=0 && num_problema<=9, "Problema a ejecutar -> [0,9]");
	assertm(num_hebras%32 == 0, "Numero de hebras -> X%32==0");

    // Objetos y puntero de los diferentes problemas
	//prueba prueba(num_points, num_hebras);
	simpleadvdiff1d simpleadvdiff1d(num_points, num_hebras);// 1D_Simple Advection-Diffusion
	simpleadvdiff1d_shuffle simpleadvdiff1d_shuffle(num_points, num_hebras);
	advdiff1d advdiff1d(num_points, num_hebras); 			// 1D Advection-Diffusion model 
	advdiff1d_shuffle advdiff1d_shuffle(num_points, num_hebras);
	brusselator1d brusselator1d(num_points, num_hebras); 	// 1D Brusselator model 
	brusselator1d_shuffle brusselator1d_shuffle(num_points, num_hebras); // 1D Brusselator model con shuffle
	brusselator1d_grid brusselator1d_grid(num_points, num_hebras); // 1D Brusselator model con grid multidimensional
	brusselator2d brusselator2d(num_points, num_hebras); 	// 2D Brusselator model
	brusselator2d_shuffle brusselator2d_shuffle(num_points, num_hebras); 	// 2D Brusselator model con shuffle
	Problema *ptr_problema; 					// Puntero al problema seleccionado
	switch(num_problema){
		//case 0: ptr_problema=&prueba; break;
		case 1: ptr_problema=&simpleadvdiff1d; break;
		case 2: ptr_problema=&simpleadvdiff1d_shuffle; break;
		case 3: ptr_problema=&advdiff1d; break;
		case 4: ptr_problema=&advdiff1d_shuffle; break;
		case 5: ptr_problema=&brusselator1d; break;
		case 6: ptr_problema=&brusselator1d_shuffle; break;
		case 7: ptr_problema=&brusselator1d_grid; break;
		case 8: ptr_problema=&brusselator2d; break;
		case 9: ptr_problema=&brusselator2d_shuffle; break;
		default: ptr_problema=NULL; break;
	}

	// Objetos y puntero de los metodos de resolucion
	const int neqn = ptr_problema->get_neqn(); 					// Obtenemos el numero de ODEs del problema
	RungeKutta RungeKutta(neqn, orden_metodo, num_hebras); 						// Objeto para aplicar Runge-Kutta y sus operaciones asociadas
	RungeKutta_graph RungeKutta_graph(neqn, orden_metodo, num_hebras, &RungeKutta); 
	AdamsBashford AdamsBashford(neqn, orden_metodo, num_hebras, &RungeKutta); 	// Objeto para aplicar Adams-Bashford y sus operaciones asociadas
	AdamsBashford_graph AdamsBashford_graph(neqn, orden_metodo, num_hebras, &RungeKutta, &AdamsBashford);
	AdamsMoulton AdamsMoulton(neqn, orden_metodo, num_hebras, &RungeKutta, &AdamsBashford); 	// Objeto para aplicar Adams-Moulton y sus operaciones asociadas
	Metodo *ptr_metodo; 											// Puntero al metodo seleccionado
	switch(num_metodo){
		case 1: ptr_metodo=&RungeKutta; break;
		case 2: ptr_metodo=&RungeKutta_graph; break;
		case 3: ptr_metodo=&AdamsBashford; break;
		case 4: ptr_metodo=&AdamsBashford_graph; break;
		case 5: ptr_metodo=&AdamsMoulton; break;
		default: ptr_metodo=NULL; break;
	}

	double *Y0_host = new double[neqn], // Vector de entrada
		   *Yf_host = new double[neqn]; // Vector de salida
	ptr_problema->init(Y0_host); // Inicializamos el vector inicial
	ptr_problema->archivo("datos0.txt",Y0_host);// Guardamos en un txt los valores iniciales
	auto timeIni = std::chrono::high_resolution_clock::now(); // Obtenemos el tiempo antes de computar
	ptr_metodo->aplicar(ptr_problema,t0,tf,h,Y0_host,Yf_host); // Aplicamos el método
	auto timeFin = std::chrono::high_resolution_clock::now(); // Obtenemos el tiempo después de computar
	ptr_problema->archivo("datos1.txt", Yf_host); // Guardamos en un txt los valores finales
	double tiempo_ms = std::chrono::duration<double, std::milli>(timeFin-timeIni).count(); // Calculamos el tiempo en milisegundos
	double tiempo_m = (tiempo_ms/1000.0)/60.0; // Calculamos el tiempo en minutos
	
	cout << "\nProblema " << num_problema << " -> " << ptr_problema->get_name() << endl;
	cout << "Metodo de resolucion -> " << ptr_metodo->get_name() << " de orden " << ptr_metodo->get_orden() << endl;
	cout << "Tamaño del vector -> " << num_points << endl;
	cout << "Numero de ecuaciones -> " << ptr_problema->get_neqn() << endl;
	cout << "T0 = " << t0 << " y tf = " << tf << endl;
	cout << "Salto h=" << h << " -> " << num_iter << " iteraciones" << endl;
	cout << "Grid de " << ptr_problema->get_grid().x << "," << ptr_problema->get_grid().y << "," << ptr_problema->get_grid().z << " bloques CUDA de ";
	cout << ptr_problema->get_block().x << "," << ptr_problema->get_block().y << "," << ptr_problema->get_block().z << " hebras" << endl;
	cout << "Tiempo -> "<< tiempo_ms << " milisegundos, es decir, " << floor(tiempo_m) << " minuto";
	if(floor(tiempo_m)!=1)
		cout << "s";
	cout << " y " << (tiempo_m-floor(tiempo_m))*60 << " segundos\n" << endl;
	
	delete [] Y0_host, Yf_host;
	return 0;
}