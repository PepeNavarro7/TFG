#ifndef BRUSSELATOR2D_CPP
#define BRUSSELATOR2D_CPP

#include "brusselator2d.h"
#include <cmath>
#include <omp.h>
#include <iostream>

using namespace std;

void brusselator2d::init(double* __restrict Y0) const {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x_i = (double)(i + 1) * get_dtx();
            double y_i = (double)(j + 1) * get_dtx();
            Y0[idx(i, j, 0)] = 22 * y_i * pow(1 - y_i, 1.5);
            Y0[idx(i, j, 1)] = 27 * x_i * pow(1 - x_i, 1.5);
        }
    }
}

//vector system function DY=FG(t,Y)
void brusselator2d::feval(const double &t, const double* __restrict Y, double* __restrict DY) const {
    // Añadimos la directiva collapse para sumar eficiencia ya que no hay dependencia de datos entre las iteraciones
    #pragma omp parallel for schedule(static) collapse(2) default(none) shared(Y, DY, nx, ny, A, B, t)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const double u_ij   = Y[idx(i, j, 0)],
                         v_ij   = Y[idx(i, j, 1)];
            const double term = B * u_ij - u_ij * u_ij * v_ij;

            const double u_im1j = (i == 0)    ? Y[idx(nx-1, j, 0)] : Y[idx(i-1, j, 0)],
                         u_ip1j = (i == nx-1) ? Y[idx(0,    j, 0)] : Y[idx(i+1, j, 0)],
                         u_ijm1 = (j == 0)    ? Y[idx(i, ny-1, 0)] : Y[idx(i, j-1, 0)],
                         u_ijp1 = (j == ny-1) ? Y[idx(i, 0   , 0)] : Y[idx(i, j+1, 0)];

            const double v_im1j = (i == 0)    ? Y[idx(nx-1, j, 1)] : Y[idx(i-1, j, 1)],
                         v_ip1j = (i == nx-1) ? Y[idx(0,    j, 1)] : Y[idx(i+1, j, 1)],
                         v_ijm1 = (j == 0)    ? Y[idx(i, ny-1, 1)] : Y[idx(i, j-1, 1)],
                         v_ijp1 = (j == ny-1) ? Y[idx(i, 0   , 1)] : Y[idx(i, j+1, 1)];

            DY[idx(i, j, 0)] = DD * (u_im1j + u_ijm1 - 4.0 * u_ij + u_ip1j + u_ijp1) 
                               + A - term - u_ij + f(i, j, t);
            DY[idx(i, j, 1)] = DD * (v_im1j + v_ijm1 - 4.0 * v_ij + v_ip1j + v_ijp1)
                               + term;
        }
    }
}
double brusselator2d::feval_i (const double &t, const double* __restrict Y, const int &i) const {
    double res;
    const int id_x = i / (2*nx); // Hago uso de la división entre enteros
    const int id_y = (i - id_x*2*nx) / 2 ;
    const int id_z = i%2;

    const double u_ij = (id_z==0) ?  (Y[i]) : (Y[i-1]), 
                 v_ij = (id_z==0) ? (Y[i+1]) : (Y[i]);
    const double term = B * u_ij - u_ij * u_ij * v_ij;

    const double im1j = (id_x == 0)    ? Y[idx(nx-1, id_y, id_z)] : Y[idx(id_x-1, id_y, id_z)], // Primera fila
                 ip1j = (id_x == nx-1) ? Y[idx(0,    id_y, id_z)] : Y[idx(id_x+1, id_y, id_z)], // Última fila
                 ijm1 = (id_y == 0)    ? Y[idx(id_x, ny-1, id_z)] : Y[idx(id_x, id_y-1, id_z)], // Primera columna
                 ijp1 = (id_y == ny-1) ? Y[idx(id_x, 0,    id_z)] : Y[idx(id_x, id_y+1, id_z)]; // Última columna


    res = DD * (im1j + ijm1 - 4.0 * Y[i] + ip1j + ijp1);
    res+= (id_z==0) ? (A - term - u_ij + f(id_x, id_y, t)) : (term);

    return res;
}

// Auxiliary function f
double brusselator2d::f(const int &i, const int &j, const double &t) const {
    const double x = (i + 1) * get_dtx(), y = (j + 1) * get_dtx();
    const double xmxc = x - 0.3, ymyc = y - 0.5;
    const double r = 0.1;
    const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;
    return result;
}

#endif