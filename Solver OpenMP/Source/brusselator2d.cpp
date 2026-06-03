#ifndef BRUSSELATOR2D_CPP
#define BRUSSELATOR2D_CPP

#include "brusselator2d.h"
#include <cmath>
#include <omp.h>

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
void brusselator2d::feval(const double &t, const double* Y, double* DY) const {
    // Añadimos la directiva collapse para sumar eficiencia ya que no hay dependencia de datos entre las iteraciones
    #pragma omp for collapse(2)
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

// Auxiliary function f
double brusselator2d::f(const int &i, const int &j, const double &t) const {
    const double x = (i + 1) * dtx, y = (j + 1) * dtx;
    const double xmxc = x - 0.3, ymyc = y - 0.5;
    const double r = 0.1;
    const double result = ((xmxc * xmxc + ymyc * ymyc) <= r * r && t >= 1.1) ? 5.0 : 0.0;
    return result;
}

#endif