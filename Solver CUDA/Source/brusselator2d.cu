#ifndef BRUSSELATOR2D_CU
#define BRUSSELATOR2D_CU

#include "brusselator2d.h"
#include <cmath>

using namespace std;

// Constructor of the class 
brusselator2d::brusselator2d(const int &nx_points) {
    name = "Brusselator_2D";
    ny = nx = nx_points;
    neqn = 2 * nx * ny;
    dtx = 1.0 / (nx + 1);
    dtx_squared = dtx * dtx;
    DD = alpha / dtx_squared;
}

void brusselator2d::init(double* Y0) const {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x_i = (double)(i + 1) * dtx;
            double y_i = (double)(j + 1) * dtx;
            Y0[idx(i, j, 0)] = 22 * y_i * pow(1 - y_i, 1.5);
            Y0[idx(i, j, 1)] = 27 * x_i * pow(1 - x_i, 1.5);
        }
    }
}

//vector system function DY=FG(t,Y)
__global__ void feval_brusselator2d(const double &t, const double* Y, double* DY, const int &nx, const double &dtx) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    // Aunque está almacenado todo un vector, podemos transformarlo en componentes x, y, z
    const int id_x = thread/(2*nx);
    const int id_y = (thread - id_x*2*nx) / 2 ;
    const int id_z = thread % 2;
    const double alpha = 0.002, A = 1.0, B = 3.4;
    const double DD = alpha / (dtx*dtx);

    // Tenemos thread hebras == ThreadsPerBlock * NumBlocks >= neqn == 2 * nx^2  
    // thread == id_x * 2 * nx + id_y * 2 + id_z  
    if (thread >= 0 && thread <= 2*nx*nx) {
        double u_ij, v_ij; // Y[thread] y pareja
        if(id_z==0){ // pares
            u_ij = Y[thread], v_ij = Y[thread+1];
        } else{ // impares
            v_ij = Y[thread], u_ij = Y[thread-1];
        }
        const double term = B * u_ij - u_ij * u_ij * v_ij;

        double u_im1j, v_im1j; // fila de arriba
        if (id_x==0){ // si primera fila
            u_im1j = v_im1j = Y[2*nx*(nx-1) + id_y*2 + id_z]; // ultima fila, misma columna
        } else {
            u_im1j = v_im1j = Y[thread - 2*nx]; // fila--
        }

        double u_ip1j, v_ip1j; // fila de abajo
        if (id_x==nx-1){ // si ultima fila
            u_ip1j = v_ip1j = Y[id_y*2 + id_z]; // primera fila, misma columna
        } else {
            u_ip1j = v_ip1j = Y[thread + 2*nx]; // fila++
        }

        double u_ijm1, v_ijm1; // columna a la izquierda
        if (id_y == 0){ // si primera columna
            u_ijm1 = v_ijm1 = Y[id_x*2*nx + (nx-1)*2 + id_z]; // ultima columna, misma fila
        } else{
            u_ijm1 = v_ijm1 = Y[thread - 2]; // columna--
        }

        double u_ijp1, v_ijp1; // columna a la derecha
        if(id_y == nx-1){ // si ultima columna
            u_ijp1 = v_ijp1 = Y[id_x*nx*2 + id_z]; // primera columna, misma fila
        } else {
            u_ijp1 = v_ijp1 = Y[thread + 2]; // columna++
        }

        if(id_z == 0){ // pares
            const double x = (id_x + 1) * dtx, y = (id_y + 1) * dtx;
            const double xmxc = x - 0.3, ymyc = y - 0.5;
            const double r = 0.1;
            const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;

            DY[thread] = DD * (u_im1j + u_ijm1 - 4.0 * u_ij + u_ip1j + u_ijp1) 
                       + A - term - u_ij + result;
        } else { // impares
            DY[thread] = DD * (v_im1j + v_ijm1 - 4.0 * v_ij + v_ip1j + v_ijp1)
                       + term;
        }     
    }
}

void brusselator2d::feval(const double &t, const double* Y, double* DY) const {
    feval_brusselator2d<<<NUM_BLOCKS,THREADSPERBLOCK>>>(t, Y, DY, nx, dtx);
}

// Auxiliary function f
/*double brusselator2d::f(const int &i, const int &j, const double &t) const {
    const double x = (i + 1) * dtx, y = (j + 1) * dtx;
    const double xmxc = x - 0.3, ymyc = y - 0.5;
    const double r = 0.1;
    const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;
    return result;
}*/

#endif